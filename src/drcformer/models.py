from functools import partial
from typing import Optional

import torch
import os
import gdown

import torch.nn as nn
from torch.nn.functional import interpolate, scaled_dot_product_attention
from torch.jit import Final

import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import Mlp, DropPath, get_act_layer, get_norm_layer, use_fused_attn

from timm.models.layers import trunc_normal_

from einops import repeat, rearrange
from torch_scatter import (
    scatter_max,
    scatter_mean,
    scatter_sum,
    scatter_softmax,
    scatter_min,
)
from torch_scatter.composite import scatter_std

from functools import partial
from pathlib import Path
import os
import torch
import torch.nn as nn

from loguru import logger
# from decorators import Deprecated

# logger.remove()

__all__ = ["suit_tiny_224", "suit_small_224", "suit_base_224", "suit_base_dino"]


# Poisitional Encoding proposed in Vaswani et al., https://arxiv.org/abs/1706.03762
class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, ch, denominator=10000.0):
        super(PositionalEncoding, self).__init__()
        assert (
            ch % (pos_dim * 2) == 0
        ), "dimension of positional encoding must be equal to dim * 2."
        enc_dim = int(ch / 2)
        div_term = torch.exp(
            torch.arange(0.0, enc_dim, 2) * -(torch.log(denominator) / enc_dim)
        )
        freqs = torch.zeros([pos_dim, enc_dim])
        for i in range(pos_dim):
            freqs[i, : enc_dim // 2] = div_term
            freqs[i, enc_dim // 2 :] = div_term
        self.freqs = freqs

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos.float(), self.freqs.to(pos.device))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# Fourier Features as Positional Encoding proposed in Tancik et al., https://arxiv.org/abs/2006.10739
class FourierFeatures(nn.Module):
    def __init__(self, pos_dim, ch, sigma=10, train=False):
        super(FourierFeatures, self).__init__()
        assert ch % 2 == 0, "number of channels must be divisible by 2."
        enc_dim = int(ch / 2)
        B = torch.randn([pos_dim, enc_dim]) * sigma
        if train:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos.float(), self.B)
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


# modified from Self-attention block of ViT: timm.models.vision_transformer.Block
class EmptyMaskingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EmptyMaskingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False
    ) -> torch.Tensor:
        if return_attention:
            x_, attn = self.attn(self.norm1(x), mask, return_attention=return_attention)
        else:
            x_ = self.attn(self.norm1(x), mask)
        x = x + self.drop_path1(self.ls1(x_))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn
        else:
            return x


# modified from original attention layer of transformers: timm.models.vision_transformer.Attention
class EmptyMaskingAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False
    ) -> torch.Tensor:
        if return_attention:
            self.fused_attn = False

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:  # no masking
            x = scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # mask padded tokens (empty superpixel clusters)
            if mask is not None:
                # attn shape: [B, heads, N, N]
                # mask shape: [B, 1, 1, N]
                # mask = torch.where(mask, torch.tensor(0.0), torch.tensor(-float("inf")))
                # 保持 shape = [b, 1, 1, N] 不变，只改常量的 device/dtype
                zero = torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
                neg_inf = torch.tensor(-float("inf"), device=attn.device, dtype=attn.dtype)
                mask = torch.where(mask, zero, neg_inf)  # [B,1,1,N]
                attn = attn + mask

            attn = attn.softmax(dim=-1)
            weight = attn  # [b, n_heads, n_tokens, n_tokens]
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, weight  # [b, n_tokens, c], [b, n_heads, n_tokens, n_tokens]
        else:
            return x


class SuperpixelVisionTransformer(VisionTransformer):
    def __init__(self, return_att=False, *args, **kwargs):
        self.pe_type = kwargs.get("pe_type", "ff")
        self.pe_injection = kwargs.get("pe_injection", "concat")
        self.downsample = kwargs.get("downsample", 2)
        self.aggregate = kwargs.get("aggregate", ["max", "avg"])
        self.base_dim = kwargs.get("base_dim", 96)
        self.embed_dim = kwargs.get("embed_dim", 384)
        self.use_proj = kwargs.get("use_proj", True)

        # MCT: for mct update
        self.decay_parameter = kwargs.get("decay_parameter", 0.996)
        self.return_att = return_att
        
        # 超像素 token 数量
        self.num_sp_tokens = kwargs.get("num_sp_tokens", 196)

        # filter keywords
        suit_only_keywords = [
            "pe_type",
            "pe_injection",
            "downsample",
            "aggregate",
            "base_dim",
            "use_proj",
            "input_size",
            "num_sp_tokens",
        ]
        old_timm_keywords = ["pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"]
        keywords_to_filter = suit_only_keywords + old_timm_keywords
        for k in keywords_to_filter:
            if k in kwargs:
                kwargs.pop(k)
        super().__init__(*args, **kwargs)

        self.img_size = kwargs.get("img_size", 224)
        self.make_coords(self.img_size, self.downsample)

        token_dim = self.base_dim * len(self.aggregate)
        if self.pe_injection == "concat":
            token_dim = token_dim * 2

        # MCT: mct update: multi-class tokens and init
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        if self.use_proj:
            assert (
                self.embed_dim % len(self.aggregate) == 0
            ), "embed dim must be divisible by number of aggregation methods."
            if self.pe_injection == "concat":
                self.projection = nn.Conv2d(
                    self.base_dim * 2, int(self.embed_dim / len(self.aggregate)), 1
                )
            else:
                self.projection = nn.Conv2d(
                    self.base_dim, int(self.embed_dim / len(self.aggregate)), 1
                )

        self.get_feats = nn.Sequential(
            nn.Conv2d(
                3,
                self.base_dim,
                7,
                self.downsample,
                padding=3,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(self.base_dim),
            nn.GELU(),
        )
        if self.pe_type == "ff":
            self.pe = FourierFeatures(2, self.base_dim, train=True)
        else:
            self.pe = PositionalEncoding(2, self.base_dim)

        # re-init transformer blocks capable of masking padded tokens (empty superpixel clusters)
        norm_layer = get_norm_layer(kwargs.get("norm_layer", None)) or partial(
            nn.LayerNorm, eps=1e-6
        )
        act_layer = get_act_layer(kwargs.get("act_layer", None)) or nn.GELU
        mlp_layer = kwargs.get("mlp_layer", Mlp)
        dpr = [
            x.item()
            for x in torch.linspace(
                0, kwargs.get("drop_path_rate", 0.0), kwargs.get("depth", 12)
            )
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                EmptyMaskingBlock(
                    dim=kwargs.get("embed_dim", 384),  # embed_dim,
                    num_heads=kwargs.get("num_heads", 6),  # num_heads,
                    mlp_ratio=kwargs.get("mlp_ratio", 4.0),  # mlp_ratio,
                    qkv_bias=kwargs.get("qkv_bias", True),  # qkv_bias,
                    qk_norm=kwargs.get("qk_norm", False),  # qk_norm,
                    init_values=kwargs.get("init_values", None),  # init_values,
                    proj_drop=kwargs.get("proj_drop_rate", 0.0),  # proj_drop_rate,
                    attn_drop=kwargs.get("attn_drop_rate", 0.0),  # attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(kwargs.get("depth", 12))
            ]
        )

        # TODO: 后续可以把这部分换成 Mlp 或者 KAN，前期为了跑通模型，先保留 Conv1d
        self.head_token = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.num_classes, kernel_size=1)
        )  # 保持与网格版 Conv2d 类似的按通道映射效果，但作用在 token 维度上

        # 假设 self.head_token = nn.Sequential(nn.Conv1d(embed_dim, num_classes, 1))
        trunc_normal_(self.head_token[0].weight, std=0.02)
        nn.init.zeros_(self.head_token[0].bias)

        if kwargs.get("weight_init", "") != "skip":
            self.init_weights(kwargs.get("weight_init", ""))
        if kwargs.get("fix_init", False):
            self.fix_init_weight()

        # remove redundant components from original ViT
        del self.patch_embed, self.pos_embed

    def tokenization(self, x, spix_label=None):
        b, c, h, w = x.shape
        spix_label = spix_label.long()

        # old code
        # n_tokens = n_tokens = int(spix_label.max()) + 1  # spix_label: [b,1,h,w]
        # NOTE: Superpixel label starts from 0.
        # 使用可配置的超像素token数量
        n_tokens = self.num_sp_tokens  # 使用可配置的超像素token数量

        spix_label = spix_label.view(b, -1)  # Flatten the labels to shape (b, h * w)
        # 将标签裁剪到 [0, n_tokens-1]，避免后续 scatter_* 与索引越界
        spix_label = spix_label.clamp(min=0, max=n_tokens - 1)
        x = x.view(b, c, -1)  # Flatten the features to shape (b, c, h * w)
        # Expand labels to shape (b, c, h * w) to match features' shape
        labels_expanded = spix_label.unsqueeze(1).expand(
            -1, c, -1
        )  # Shape: (b, c, h * w)
        # Flatten the batch and channel dimensions to shape (b * c, h * w)
        labels_expanded = labels_expanded.reshape(-1, h * w).long()
        x_flat = x.reshape(-1, h * w)
        x_flat = x_flat.float()

        out_list = []
        # NOTE: MAX
        if "max" in self.aggregate:
            # Perform scatter_max
            max_out, _ = scatter_max(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            max_out = max_out.view(b, c, n_tokens).permute(
                0, 2, 1
            )  # Shape: (b, n_tokens, c)
            out_list.append(max_out)

        # NOTE: MIN
        if "min" in self.aggregate:
            # Perform scatter_min
            min_out, _ = scatter_min(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            min_out = min_out.view(b, c, n_tokens).permute(
                0, 2, 1
            )  # Shape: (b, n_tokens, c)
            out_list.append(min_out)

        # NOTE: AVG
        if "avg" in self.aggregate:
            # Perform scatter_mean
            mean_out = scatter_mean(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            mean_out = mean_out.view(b, c, n_tokens).permute(
                0, 2, 1
            )  # Shape: (b, n_tokens, c)
            out_list.append(mean_out)

        # NOTE: STD
        if "std" in self.aggregate:
            # Perform scatter_max
            std_out = scatter_std(x_flat, labels_expanded, dim=1, dim_size=n_tokens)
            std_out = std_out.view(b, c, n_tokens).permute(
                0, 2, 1
            )  # Shape: (b, n_tokens, c)
            out_list.append(std_out)

        # NOTE: SOFTMAX
        if "softmax" in self.aggregate:
            # Perform scatter_softmax
            softmax_weights = scatter_softmax(
                x_flat, labels_expanded, dim=1, dim_size=n_tokens
            )
            # Weighted features
            weighted_softmax_features = x_flat * softmax_weights
            # Perform scatter_sum to aggregate the weighted features
            softmax_out = scatter_sum(
                weighted_softmax_features, labels_expanded, dim=1, dim_size=n_tokens
            )
            softmax_out = softmax_out.view(b, c, n_tokens).permute(
                0, 2, 1
            )  # Shape: (b, n_tokens, c)
            out_list.append(softmax_out)

        # NOTE: mask padded tokens (empty superpixel clusters)
        ones = (
            torch.ones(b, 1, h, w, device=x.device, requires_grad=False)
            .reshape(-1, h * w)
            .int()
        )
        spix_labels_flat = spix_label.reshape(-1, h * w).long()
        cluster_counter = (
            scatter_sum(ones, spix_labels_flat, dim=1, dim_size=n_tokens)
            .view(b, 1, n_tokens)
            .permute(0, 2, 1)
        )
        mask = cluster_counter != 0  # mask empty tokens [B,1,1,N]
        mask = rearrange(mask, "b n 1 -> b 1 1 n")

        # Concatenate all selected pooled features
        tokens = torch.cat(out_list, dim=2)  # Concatenate along the channel dimension

        return tokens, mask

    def prepare_tokens(self, x: torch.Tensor, spix_label: torch.Tensor) -> torch.Tensor:
        x = self.get_feats(x)  # x: [b,3,h,w] -> x: [b,embed_dim,h,w]
        b, _, h_feat, w_feat = x.shape

        # 动态对齐坐标网格到特征图大小，避免多尺度输入导致的尺寸不匹配
        try:
            need_rebuild = (
                not hasattr(self, "coords")
                or self.coords is None
                or self.coords.shape[1] != w_feat
                or self.coords.shape[2] != h_feat
                or (isinstance(self.coords, torch.Tensor) and self.coords.device != x.device)
            )
        except Exception:
            need_rebuild = True
        if need_rebuild:
            # 传入 tuple(H,W) 分支，make_coords 会生成 [1,W,H,2]，后续 permute 得到 [1,C,H,W]
            self.make_coords(img_size=(h_feat, w_feat), device=x.device)

        pe = self.pe(self.coords).permute(
            0, 3, 2, 1
        )  # [1,w,h,embed_dim] -> [1,embed_dim,h,w]
        pe = repeat(pe, "1 c h w -> b c h w", b=x.shape[0])
        x = torch.cat([x, pe], dim=1) if self.pe_injection == "concat" else x + pe
        if self.use_proj:
            x = self.projection(x)

        # tokenization with superpixel labels
        # 确保 spix_label 有正确的维度用于插值
        if spix_label.dim() == 3:  # (B, H, W)
            spix_label = spix_label.unsqueeze(1)  # (B, 1, H, W)
        spix_label = interpolate(
            spix_label.to(torch.float), size=x.shape[-2:], mode="nearest"
        )
        spix_label = spix_label.squeeze(1)  # 移除 channel 维度变回 (B, H, W)
        tokens, mask = self.tokenization(x, spix_label)

        # 扩展 cls_tokens 到 batch 大小
        cls_tokens = repeat(
            self.cls_token, "1 c d -> b c d", b=b
        )  # cls_tokens: [b,num_classes,embed_dim]
        x = torch.cat((cls_tokens, tokens), dim=1)

        cls_mask = torch.ones(
            b,
            1,
            1,
            self.num_classes,
            dtype=torch.bool,
            device=x.device,
            requires_grad=False,
        )
        mask = torch.cat((cls_mask, mask), dim=-1)
        x = self.pos_drop(x)
        return x, mask  # x:[b, num_classes+num_tokens, embed_dim]

    def forward_features(
        self, x: torch.Tensor, spix_label: torch.Tensor, n=12
    ) -> torch.Tensor:
        x, mask = self.prepare_tokens(
            x, spix_label
        )  # patch_embed and _pos_embed in once
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # 保存注意力权重和 cls-tokens
        attn_weights = []
        class_embeddings = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(
                x, mask, return_attention=True
            )  # x: [b, num_classes+num_tokens, embed_dim]
            # --- debug loguru
            # logger.debug(f"shape of weights_i: {weights_i.shape}")
            attn_weights.append(weights_i)
            class_embeddings.append(x[:, 0 : self.num_classes])
        x = self.norm(x)

        return (
            x[:, 0 : self.num_classes],
            x[:, self.num_classes :],
            attn_weights,
            class_embeddings,
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        spix_label: torch.Tensor,
        n_layers=3,
        attention_type="fused",
        return_att=False,
        return_b1: bool = False,
        return_extras: bool = False,
    ) -> torch.Tensor:
        """return_extras (only with return_att): return a dict instead of the
        (x_logits, cams, patch_attn) tuple, adding the two logit branches, all three
        CAM types and the per-superpixel class scores. The selected `cams` are identical
        to the tuple's; no parameters are added, so checkpoints load unchanged.
        """

        # --- 进行特征传播 ---
        # 变量说明：
        # x_cls: (B, K, C), K 表示 num_classes
        # x_tok: (B, T, C), T 表示 sp_token 的数目
        # attn_weights: 列表，每个元素为 (B, H, T, num_classes + T)
        # all_x_cls: 列表，每个元素为 (B, K, C)
        # mask: (B, 1, 1, num_classes + T)
        x_cls, x_tok, attn_weights, all_x_cls, mask = self.forward_features(
            x, spix_label
        )

        """ 计算 x_patch_logits """
        # x_tok: (B, T, C), T 表示 sp_token 的数目
        # mask:  (B, 1, 1, num_classes + T)
        mask_tok = mask[..., self.num_classes :]  # (B, 1, 1, T)
        mask_tok = mask_tok.view(mask_tok.size(0), -1)  # (B, T), True=有效

        # 1) token级打分
        # x_token_cls = self.head_token(x_tok)    # (B, T, K), K=num_classes
        # (B, T, C) -> (B, C, T) -> Conv1d -> (B, K, T) -> (B, T, K)
        x_token_cls = self.head_token(x_tok.transpose(1, 2)).transpose(1, 2)

        # 2) 排序 + logspace 衰减权重（对每类分数按 T 维排序），并用 mask 屏蔽无效 token
        B, T, K = x_token_cls.shape
        # 把无效 token 的分数拉到 -inf，避免排序时冒到前面
        minus_inf = torch.finfo(x_token_cls.dtype).min
        x_masked = x_token_cls.masked_fill(~mask_tok.unsqueeze(-1), minus_inf)

        sorted_tok, _ = torch.sort(x_masked, dim=1, descending=True)  # (B, T, K)
        weights = torch.logspace(
            0,
            T - 1,
            steps=T,
            base=self.decay_parameter,
            device=x_token_cls.device,
            dtype=x_token_cls.dtype,
        ).view(1, T, 1)
        # 对无效部分的权重清零（-inf * w 会是 -inf，数值不稳，直接清零更安全）
        valid_weights = weights * mask_tok.unsqueeze(-1)

        x_patch_logits = (sorted_tok.clamp_min(minus_inf / 2) * valid_weights).sum(
            dim=1
        ) / valid_weights.sum(dim=1).clamp_min(
            1e-6
        )  # (B, K)

        # 3) CLS 路径（原样）
        x_cls_logits = x_cls.mean(-1)  # (B, K)

        output = []
        output.append(x_cls_logits)
        output.append(torch.stack(all_x_cls))
        output.append(x_patch_logits)

        # 当启用 B1 时，追加返回供训练阶段使用（保持默认不启用时的兼容性）
        if return_b1:
            output.append((x_token_cls, mask_tok))  # B1 bundle: (B,T,K), (B,T)

        if self.return_att or return_att:
            # === 1) 叠注意力，并得到最近 n_layers 的 head-mean ===
            # attn_weights: list[L] of (B, Hh, N, N)
            attn_weights_stacked = torch.stack(attn_weights)  # (L, B, Hh, N, N)
            attn_weights_mean = attn_weights_stacked.mean(dim=2)  # (L, B, N, N)
            attn_tail = attn_weights_mean[-n_layers:].mean(0)  # (B, N, N)

            K = self.num_classes
            # 类token → 图像token（去掉前 K 个类token，保留超像素 token）
            # mtatt_sub[b, k, t] 表示第 b 个样本中，第 k 类 token 对第 t 个超像素 token 的注意力
            mtatt_sub = attn_tail[:, :K, K:]  # (B, K, T)
            # 与网格版一致的 patch_attn（仅图像token间的注意力，跨层保留）
            patch_attn = attn_weights_mean[:, :, K:, K:]  # (L, B, T, T)

            # === 2) 构造 per-class token 激活（用于 patchcam/fused） ===
            # x_token_cls: (B, T, K)  → 取 ReLU 作为非负响应（与网格版 feature_map 的 ReLU 对齐）
            x_token_cls_act = F.relu(x_token_cls.detach())  # (B, T, K)

            # === 3) 映射到像素空间，得到 cams: (B, K, H, W) ===
            cams = []
            cams_all = {"mct": [], "patchcam": [], "fused": []}  # 仅 return_extras 时填充
            # spix_label: (B, 1, H, W) 或 (B, H, W)，值域 ∈ [0..T-1]
            if spix_label.dim() == 4:
                assignment = spix_label.squeeze(1)  # (B, H, W)
            else:
                assignment = spix_label  # (B, H, W)

            Bsz = x.shape[0]
            for b in range(Bsz):
                # 仅用有效 token，避免空簇/越界
                Ti = int(mask_tok[b].sum().item())
                assign_b = assignment[b].long()
                H_i, W_i = assign_b.shape[-2], assign_b.shape[-1]
                if Ti == 0:
                    # 没有有效 token，直接返回零 CAM，避免对空张量索引导致非法访问
                    zeros = torch.zeros(K, H_i, W_i, device=x.device, dtype=x.dtype)
                    cams.append(zeros)
                    if return_extras:
                        for kind in cams_all:
                            cams_all[kind].append(zeros)
                    continue
                # 将标签裁剪到 [0, Ti-1]，避免负索引或越界
                assign_b = assign_b.clamp(min=0, max=max(Ti - 1, 0))  # (H, W)

                # 注意力 CAM（与网格版 mct 对齐）：类→token 注意力映射到像素
                mtatt_b = mtatt_sub[b, :, :Ti]  # (K, Ti)
                mtatt_hw = mtatt_b[:, assign_b]  # (K, H, W)

                # 特征 CAM（与网格版 patchcam 对齐）：per-token 类激活映射到像素
                feat_b = x_token_cls_act[b, :Ti].transpose(0, 1)  # (K, Ti)
                feat_hw = feat_b[:, assign_b]  # (K, H, W)

                if attention_type == "fused":
                    cam_b = torch.sqrt(
                        (mtatt_hw * feat_hw).clamp_min(0)
                    )  # 与网格版一致：融合后开方
                elif attention_type == "patchcam":
                    cam_b = feat_hw
                elif attention_type == "mct":
                    cam_b = mtatt_hw
                else:
                    raise RuntimeError(f"Error! {attention_type} is not defined!")

                cams.append(cam_b)
                if return_extras:
                    cams_all["mct"].append(mtatt_hw)
                    cams_all["patchcam"].append(feat_hw)
                    cams_all["fused"].append(torch.sqrt((mtatt_hw * feat_hw).clamp_min(0)))

            cams = torch.stack(cams, dim=0)  # (B, K, H, W)

            # === 4) 与网格版一致的 x_logits 汇合 ===
            x_logits = (x_cls_logits + x_patch_logits) / 2  # (B, K)
            if return_extras:
                return {
                    "logits": x_logits,  # (B, K)，与三元组第一项相同
                    "cams": cams,  # (B, K, H, W)，attention_type 选定的 CAM
                    "patch_attn": patch_attn,  # (L, B, T, T)
                    "cls_logits": x_cls_logits,  # (B, K)，类 token 路
                    "patch_logits": x_patch_logits,  # (B, K)，超像素 token 路
                    "cams_all": {k: torch.stack(v, dim=0) for k, v in cams_all.items()},  # 各 (B, K, H, W)
                    "token_cls": x_token_cls.detach(),  # (B, T, K)，每个超像素的类别分数（未 ReLU）
                    "token_mask": mask_tok,  # (B, T)，True 为非空超像素
                }
            return (
                x_logits,
                cams,
                patch_attn,
            )  # x_logits: 图像级预测; cams: 每类像素级热力图; patch_attn: 各层图像token间注意力矩阵
        else:
            return output

    def make_coords(self, img_size=None, downsample=None, device=None):
        if device is None:
            if hasattr(self, "coords") and isinstance(getattr(self, "coords"), torch.Tensor):
                device = self.coords.device
            else:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if img_size is None:
            img_size = self.img_size
        if downsample is None:
            downsample = self.downsample

        if isinstance(img_size, tuple):
            # Coordinates for Positional Encoding
            x_coord, y_coord = torch.arange(
                0, int(img_size[1]), device=device
            ), torch.arange(
                0, int(img_size[0]), device=device
            )  # x_coord: [w], y_coord: [h]
            x_coord, y_coord = x_coord / (img_size[1] - 1), y_coord / (img_size[0] - 1)
            self.coords = torch.cartesian_prod(x_coord, y_coord).reshape(
                1, img_size[1], img_size[0], 2
            )  # [w*h,2] -> [1,w,h,2]
        else:
            img_size = int(img_size / downsample)
            # Coordinates for Positional Encoding
            x_coord, y_coord = torch.arange(0, img_size, device=device), torch.arange(
                0, img_size, device=device
            )  # x_coord: [w], y_coord: [h]
            x_coord, y_coord = x_coord / (img_size - 1), y_coord / (img_size - 1)
            self.coords = torch.cartesian_prod(x_coord, y_coord).reshape(
                1, img_size, img_size, 2
            )  # [w*h,2] -> [1,w,h,2]

    def reset_stride(self, new_stride, reset_coords=True):
        self.get_feats[0].stride = new_stride
        self.downsample = new_stride
        if reset_coords:
            self.make_coords(downsample=new_stride, device=next(self.parameters()).device)

    def reset_img_size(self, img_size, reset_coords=True):
        self.img_size = img_size
        if reset_coords:
            self.make_coords(img_size=img_size, device=next(self.parameters()).device)

    def get_last_selfattention(self, x, spix_label):
        x, mask = self.prepare_tokens(
            x, spix_label
        )  # patch_embed and _pos_embed in once
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, mask)
            else:
                _, attn = blk(x, mask, return_attention=True)

        return attn

    def get_selfattentions(self, x, spix_label):
        x, mask = self.prepare_tokens(
            x, spix_label
        )  # patch_embed and _pos_embed in once
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        attns = []
        for blk in self.blocks:
            x, attn = blk(x, mask, return_attention=True)
            attns.append(attn)

        attns = torch.stack(attns)
        return attns

    def get_intermediate_features(self, x, spix_label):
        features = []
        x, mask = self.prepare_tokens(
            x, spix_label
        )  # patch_embed and _pos_embed in once
        features.append(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x, mask)
            features.append(x)

        features = torch.stack(features)
        return features


# @Deprecated
@register_model
def suit_tiny_224(pretrained=False, **kwargs):
    model = SuperpixelVisionTransformer(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        base_dim=48,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        url = "https://drive.google.com/uc?export=download&id=1Yvje-LLkHdeAo3RXrguzV-twNH2sn4Js"
        output = "suit_tiny_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    return model


# @Deprecated
@register_model
def suit_small_224(pretrained=False, **kwargs):
    model = SuperpixelVisionTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        base_dim=96,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        url = "https://drive.google.com/uc?export=download&id=1steBEtsFYnAtTyS29jJUb2DrsN_qQPPv"
        output = "suit_small_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    return model


# @Deprecated
@register_model
def suit_base_224_old(pretrained=False, **kwargs):
    model = SuperpixelVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        url = "https://drive.google.com/uc?export=download&id=1ZwS2Ig8YL3WjOWkqYJjzjiRSh1J_s0KW"
        output = "suit_base_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    return model


# @Deprecated
@register_model
def suit_base_dino(pretrained=False, **kwargs):
    model = SuperpixelVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        url = "https://drive.google.com/uc?export=download&id=1jnr9WLEzyrv4AzKWT0U04PS6CBO9v0IH"
        output = "suit_base_dino.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def drcformer(pretrained=False, ckpt_path=None, strict=True, device="cpu", **kwargs):
    """
    ckpt_path: 本地 .pth/.pt 路径；若提供则优先从本地加载，不再下载
    strict:    是否严格匹配；若你改了 head 的维度，建议 strict=False
    device:    用于 torch.load 的 map_location（"cpu" 更通用）
    """
    # 从 kwargs 获取 num_classes，如果没有提供则使用20作为默认值（VOC）
    num_classes = kwargs.pop('num_classes', 20)
    
    model = SuperpixelVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        base_dim=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        **kwargs,
    )

    if not pretrained and ckpt_path is None:
        return model  # 不加载任何权重

    # 1) 解析权重来源：本地优先，否则走原来的下载
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    else:
        url = "https://drive.google.com/uc?export=download&id=1ZwS2Ig8YL3WjOWkqYJjzjiRSh1J_s0KW"
        output = "suit_base_224.pth"
        if not os.path.exists(output):
            print(f"{output} not found. Downloading from {url}...")
            import gdown

            gdown.download(url, output, quiet=False)
        else:
            print(f"{output} already exists. Skipping download.")
        checkpoint = torch.load(output, map_location=device, weights_only=False)

    # 2) 取出 state_dict（兼容多种格式）
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint  # 直接就是 state_dict

    # 3) 去掉 DataParallel/DistributedDataParallel 的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state_dict[k] = v

    # 4) 如分类头/多类 token 头不匹配，自动丢弃（便于迁移学习）
    model_sd = model.state_dict()
    drop_keys = []
    for k, v in new_state_dict.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            # 根据你模型命名，包含 head 层、token 头等关键字可直接丢弃
            if k.startswith(("head.", "head_token.", "aux_head.", "seg_head.")):
                drop_keys.append(k)
    for k in drop_keys:
        new_state_dict.pop(k)

    # 5) 最终加载
    missing, unexpected = model.load_state_dict(
        new_state_dict, strict=False if drop_keys else strict
    )
    if missing:
        print("[load] missing keys:", missing)
    if unexpected:
        print("[load] unexpected keys:", unexpected)
    if drop_keys:
        print("[load] dropped incompatible keys:", drop_keys)

    return model
