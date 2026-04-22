import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from skimage import transform, io
# Optional dependency: pydensecrf
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    _HAS_PY_DCRF = True
except Exception:
    dcrf = None
    unary_from_labels = None
    _HAS_PY_DCRF = False


class SuperPixelDINO(nn.Module):
    def __init__(self, model_path, device='cpu'):
        super(SuperPixelDINO, self).__init__()
        self.device = device
        
        # 加载预训练权重
        self.init_model = np.load(model_path, allow_pickle=True).item()
        
        # 确定模型配置（从权重形状推断，避免依赖文件名）
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        patch_w_np = self.init_model["patch_embed.proj.weight"]  # (P,P,3,C)
        self.P = int(patch_w_np.shape[0])  # patch大小
        self.C = int(patch_w_np.shape[3])  # 特征维度
        self.H = self.C // 64              # attention头数
        
        # 颜色归一化参数
        self.color_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.color_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # 构建模型
        self._build_model()
        
    def _build_model(self):
        # Patch Embedding层
        patch_weight = torch.from_numpy(self.init_model["patch_embed.proj.weight"]).float()
        patch_weight = patch_weight.permute(3, 2, 0, 1)  # (P,P,3,C) -> (C,3,P,P)
        patch_bias = torch.from_numpy(self.init_model["patch_embed.proj.bias"]).float()
        
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.C,
            kernel_size=self.P,
            stride=self.P,
            padding=0
        )
        self.patch_embed.weight.data = patch_weight
        self.patch_embed.bias.data = patch_bias
        
        # CLS token和位置编码
        self.cls_token = nn.Parameter(torch.from_numpy(self.init_model["cls_token"]).float())
        self.pos_embed = nn.Parameter(torch.from_numpy(self.init_model["pos_embed"]).float())
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(12):  # DINO通常有12个transformer block
            block = TransformerBlock(self.C, self.H, self.init_model, str(i))
            self.blocks.append(block)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 图像归一化
        x = (x - self.color_mean.to(x.device)) / self.color_std.to(x.device)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, C, H//P, W//P]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        pos_embed = self._interpolate_pos_embed(self.pos_embed, H//self.P, W//self.P)
        x = x + pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return x[:, 1:, :]  # 移除CLS token，返回patch特征
    
    def _interpolate_pos_embed(self, pos_embed, H, W):
        """插值位置编码以适应不同的图像尺寸"""
        N = pos_embed.shape[1] - 1  # 减去CLS token
        if N == H * W:
            return pos_embed
        
        cls_pos_embed = pos_embed[:, 0:1]
        patch_pos_embed = pos_embed[:, 1:]
        
        # 重塑为2D
        dim = int(np.sqrt(N))
        patch_pos_embed = patch_pos_embed.reshape(1, dim, dim, -1).permute(0, 3, 1, 2)
        
        # 双线性插值
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False
        )
        
        # 重塑回1D
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        
        return torch.cat([cls_pos_embed, patch_pos_embed], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, init_model, block_id):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Layer Norm 1
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm1.weight.data = torch.from_numpy(init_model[f"blocks.{block_id}.norm1.weight"]).float()
        self.norm1.bias.data = torch.from_numpy(init_model[f"blocks.{block_id}.norm1.bias"]).float()
        
        # QKV projection
        qkv_weight = torch.from_numpy(init_model[f"blocks.{block_id}.attn.qkv.weight"]).float().T
        qkv_bias = torch.from_numpy(init_model[f"blocks.{block_id}.attn.qkv.bias"]).float()
        self.qkv = nn.Linear(dim, dim * 3)
        self.qkv.weight.data = qkv_weight
        self.qkv.bias.data = qkv_bias
        
        # Output projection
        proj_weight = torch.from_numpy(init_model[f"blocks.{block_id}.attn.proj.weight"]).float().T
        proj_bias = torch.from_numpy(init_model[f"blocks.{block_id}.attn.proj.bias"]).float()
        self.proj = nn.Linear(dim, dim)
        self.proj.weight.data = proj_weight
        self.proj.bias.data = proj_bias
        
        # Layer Norm 2
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2.weight.data = torch.from_numpy(init_model[f"blocks.{block_id}.norm2.weight"]).float()
        self.norm2.bias.data = torch.from_numpy(init_model[f"blocks.{block_id}.norm2.bias"]).float()
        
        # MLP
        fc1_weight = torch.from_numpy(init_model[f"blocks.{block_id}.mlp.fc1.weight"]).float().T
        fc1_bias = torch.from_numpy(init_model[f"blocks.{block_id}.mlp.fc1.bias"]).float()
        fc2_weight = torch.from_numpy(init_model[f"blocks.{block_id}.mlp.fc2.weight"]).float().T
        fc2_bias = torch.from_numpy(init_model[f"blocks.{block_id}.mlp.fc2.bias"]).float()
        
        self.mlp_fc1 = nn.Linear(dim, fc1_weight.shape[0])
        self.mlp_fc1.weight.data = fc1_weight
        self.mlp_fc1.bias.data = fc1_bias
        
        self.mlp_fc2 = nn.Linear(fc2_weight.shape[0], dim)
        self.mlp_fc2.weight.data = fc2_weight
        self.mlp_fc2.bias.data = fc2_bias
    
    def forward(self, x):
        # Multi-head attention
        residual = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = F.gelu(x)
        x = self.mlp_fc2(x)
        x = residual + x
        
        return x


class SuperPixelGenerator:
    def __init__(self, model_path="model/dino22416s", device='cpu'):
        self.device = device
        self.model = SuperPixelDINO(model_path, device).to(device)
        self.model.eval()
        
    def load_image(self, img_file, scale=1):
        """加载并预处理图像"""
        # 使用PIL加载图像
        img = Image.open(img_file).convert('RGB')
        img_array = np.array(img)
        
        h0, w0 = img_array.shape[:2]
        P = self.model.P
        
        # 计算新尺寸
        h = int((h0 * scale - 1) // P + 1)
        w = int((w0 * scale - 1) // P + 1)
        
        # 使用skimage resize替代cv2
        resized_img = transform.resize(
            img_array, 
            (h * P, w * P), 
            order=3,  # 三次插值，等价于cv2.INTER_CUBIC
            preserve_range=True,
            anti_aliasing=True
        ) / 255.0
        
        # 转换为PyTorch tensor，调整维度顺序
        img_tensor = torch.from_numpy(resized_img).float().permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, h, w, img_array, h0, w0
    
    def crf_inference_label(self, img, labels, t=20, n_labels=21, gt_prob=0.7):
        """CRF后处理"""
        if not _HAS_PY_DCRF:
            raise RuntimeError("pydensecrf is not installed, cannot run CRF post-processing")
        h, w = img.shape[:2]
        
        d = dcrf.DenseCRF2D(w, h, n_labels)
        
        unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
        
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=10, compat=10)
        d.addPairwiseBilateral(sxy=20, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)
        
        q = d.inference(t)
        
        return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
    
    def get_voc_palette(self, num_cls):
        """生成调色板"""
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while lab > 0:
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
        return palette
    
    def get_mask_palette(self, npimg):
        """应用调色板到mask"""
        voc_palette = self.get_voc_palette(256)
        out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
        out_img.putpalette(voc_palette)
        return out_img
    
    def generate_superpixel(self, img_file, tau=0.2, out_dir="superpixels/", use_crf=False, crf_only=False, if_tip=True):
        """生成超像素分割结果"""
        # 创建输出目录
        os.makedirs(out_dir, exist_ok=True)
        if use_crf:
            if not _HAS_PY_DCRF:
                print("[WARN] pydensecrf not installed, disabling CRF post-processing.")
                use_crf = False
            else:
                # CRF结果保存在同一个输出目录中，使用不同的文件名前缀区分
                crf_dir = out_dir
                os.makedirs(crf_dir, exist_ok=True)
        
        # 加载图像
        img_tensor, h, w, img0, h0, w0 = self.load_image(img_file)
        img_tensor = img_tensor.to(self.device)
        
        # 模型推理
        with torch.no_grad():
            features = self.model(img_tensor)  # [1, H*W, C]
            
        # 转换为numpy并归一化特征
        features = features.cpu().numpy()[0]  # [H*W, C]
        f = features / np.sqrt(np.sum(features**2, axis=1, keepdims=True))
        
        # 计算相似度矩阵
        apq = f @ f.T
        bpq = apq > 0
        
        # 超像素聚类算法
        pset = np.ones([h*w, 1], dtype=bool)
        selset = np.zeros([0, h*w], dtype=np.int32)
        count = 0
        
        while np.sum(pset) != 0:
            bpq_m = bpq * pset * pset.T
            
            dp = np.sum(bpq_m, 0)
            dp[dp == 0] = h * w + 1
            
            pstar = np.argmin(dp)  # 种子像素索引
            
            S = apq[pstar]
            set1 = (S > tau) & (pset[:, 0] == 1)
            
            selset = np.concatenate([selset, set1.reshape(1, h*w)], axis=0)
            pset[set1 == True] = 0
            count += 1
        
        if if_tip:
            print(f"模型: DINO {self.model.model_name}")
            print(f"tau = {tau}, 超像素数量: {count}")
        
        # 生成超像素标签
        npimg = np.argmax(selset, axis=0).reshape(h, w)
        
        # 使用skimage resize替代cv2
        npimg_resized = transform.resize(
            npimg, 
            (h0, w0), 
            order=0,  # 最近邻插值，等价于cv2.INTER_NEAREST
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.int32)
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        
        # 只在非crf_only模式下保存原始结果
        if not crf_only:
            result_img = self.get_mask_palette(npimg_resized + 1)
            result_img.save(os.path.join(out_dir, f"{base_name}.png"))
        
        # CRF后处理（如果需要）
        if use_crf:
            npimg_crf = self.crf_inference_label(img0, npimg_resized, n_labels=np.max(npimg_resized)+1)
            result_img_crf = self.get_mask_palette(npimg_crf + 1)
            
            # 在crf_only模式下，保存为普通文件名；否则保存为.crf.png
            if crf_only:
                result_img_crf.save(os.path.join(crf_dir, f"{base_name}.png"))
            else:
                result_img_crf.save(os.path.join(crf_dir, f"{base_name}.crf.png"))
            
            return npimg_crf, count  # 返回CRF处理后的结果
        
        return npimg_resized, count


def main():
    """
    使用示例
    """
    from pathlib import Path
    # init all params
    model_path = Path("assets/saved_models/sp_generator/dino22416b.npy")
    input_path = Path("assets/examples/sp_gen_eg.jpg")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化超像素生成器
    generator = SuperPixelGenerator(str(model_path), device)
    
    # 输入图像和输出路径
    img_file = os.path.join(os.path.dirname(__file__), str(input_path))  # 使用脚本同目录下的示例图
    out_path = "assets/outputs/sp-generator/superpixels/"
    use_crf = True  # 设置为True来使用CRF后处理（需要安装pydensecrf）
    
    # 生成不同tau值的超像素
    tau_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for tau in tau_values:
        print(f"\n生成 tau={tau} 的超像素...")
        model_name = os.path.basename(str(model_path.stem))
        out_dir = f"{out_path}{model_name}-{int(tau*10):02d}/"
        
        try:
            result, count = generator.generate_superpixel(
                img_file, 
                tau=tau, 
                out_dir=out_dir, 
                use_crf=use_crf
            )
            print(f"成功生成 {count} 个超像素")
        except Exception as e:
            print(f"生成超像素时出错: {e}")


if __name__ == "__main__":
    main()