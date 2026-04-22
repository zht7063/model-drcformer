import math
import sys
from typing import Iterable
from collections import defaultdict

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
from skimage import io as skio
from skimage.segmentation import find_boundaries
from matplotlib import cm
import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# 定义 JET 颜色映射
JET_CMAP = cm.get_cmap('jet')

# VOC 风格调色板 -> 用于可视化语义类别的伪掩码/注意力图。
palette = [ 0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 255, 255, 255, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, ]


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    set_training_mode=True,
    args=None,
    writer=None,
):
    model.train(set_training_mode)  # 设置训练模式

    # MetricLogger：包装 data_loader，每 print_freq 个 iteration 打印滑动平均的指标（这里额外注册了 lr）。
    metric_logger = utils.MetricLogger(delimiter="  ")  # 包装 dataloader
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    # --- 读取“老权重” ---
    base_w_mct   = getattr(args, "mct_loss_weight",   1.0)
    base_w_attn  = getattr(args, "attn_loss_weight",  1.0)
    base_w_patch = getattr(args, "patch_loss_weight", 1.0)

    # 目标“新权重”（若未提供则回退为老权重，保证兼容）
    tgt_w_mct   = getattr(args, "new_mct_loss_weight",   None)
    tgt_w_attn  = getattr(args, "new_attn_loss_weight",  None)
    tgt_w_patch = getattr(args, "new_patch_loss_weight", None)
    tgt_w_mct   = base_w_mct   if (tgt_w_mct   is None) else float(tgt_w_mct)
    tgt_w_attn  = base_w_attn  if (tgt_w_attn  is None) else float(tgt_w_attn)
    tgt_w_patch = base_w_patch if (tgt_w_patch is None) else float(tgt_w_patch)

    # --- 是否启用 & 何时切换 ---
    swap_enabled = bool(getattr(args, "loss_swap_enable", False))
    swap_epoch   = int(getattr(args, "loss_swap_epoch", 0))
    # 可选：平滑过渡的周期（未在 CLI 显式声明也不影响，默认 0 表示瞬时切换）
    warm_epochs  = int(getattr(args, "loss_swap_warm_epochs", 0))

    if swap_enabled and (epoch >= swap_epoch):
        if warm_epochs > 0:
            # 线性插值：从 swap_epoch 开始，warm_epochs 内从 base -> target 平滑过渡
            # 例如 warm_epochs=4：在 swap_epoch 时 α≈0.25，随后 0.5、0.75、1.0
            alpha = (epoch - swap_epoch + 1) / float(warm_epochs)
            alpha = max(0.0, min(1.0, alpha))
        else:
            alpha = 1.0

        w_mct   = (1.0 - alpha) * base_w_mct   + alpha * tgt_w_mct
        w_attn  = (1.0 - alpha) * base_w_attn  + alpha * tgt_w_attn
        w_patch = (1.0 - alpha) * base_w_patch + alpha * tgt_w_patch

        # 首次进入切换阶段时记录一次摘要（含平滑信息）
        if not hasattr(args, "_loss_swap_notified"):
            try:
                is_main = utils.is_main_process()
            except Exception:
                is_main = True
            if is_main:
                warm_msg = f" with warm_epochs={warm_epochs}" if warm_epochs > 0 else " (instant)"
                print(
                    f"[LossSwap] Epoch {epoch} reached. Target -> mct:{tgt_w_mct} attn:{tgt_w_attn} patch:{tgt_w_patch};"
                    f" base -> mct:{base_w_mct} attn:{base_w_attn} patch:{base_w_patch}{warm_msg}"
                )
            setattr(args, "_loss_swap_notified", True)
    else:
        # 未开启或尚未到切换点时，维持老权重
        w_mct, w_attn, w_patch = base_w_mct, base_w_attn, base_w_patch

    logger.info(
        f"[LossSwap] Epoch {epoch} weights -> mct:{w_mct:.4f} attn:{w_attn:.4f} patch:{w_patch:.4f}"
    )

    # 计算全局步数基数，用于 TensorBoard 记录
    try:
        data_len = len(data_loader)
    except Exception:
        data_len = 0
    global_step_base = epoch * data_len
    iter_idx = 0

    for samples, spix_labels, dino_labels, targets in metric_logger.log_every(
        data_loader, print_freq, header
    ):
        samples = samples.to(device, non_blocking=True)
        spix_labels = spix_labels.to(device, non_blocking=True)
        # dino_labels可能为None，需要特殊处理
        if dino_labels is not None:
            dino_labels = dino_labels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        c_outputs = None
        b1_bundle = None
        with torch.amp.autocast('cuda'):
            outputs = model(samples, spix_labels, return_b1=(getattr(args, "b1_enable", False) or getattr(args, "sp_align_enable", False)))

            if len(outputs) == 2:
                outputs, patch_outputs = outputs
            elif len(outputs) == 3:
                outputs, c_outputs, patch_outputs = outputs
            elif len(outputs) == 4:
                outputs, c_outputs, patch_outputs, b1_bundle = outputs

            # --- 主分类损失（MCT） ---
            mct_loss  = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(mct_loss=mct_loss.item())

            total_loss  = w_mct * mct_loss 

            # --- 注意力一致性/正则损失（ATTN） ---
            if c_outputs is not None:
                # 取后 num_cct 层的 class tokens: c_outputs: [L, B, C, D]
                c_outputs = c_outputs[-args.num_cct:]
                # 归一化 + 相似度（一次 matmul 即可），classes 在最后一维
                emb = F.normalize(c_outputs, dim=-1)                        # [L,B,C,D]
                scores = torch.matmul(emb, emb.transpose(-1, -2))           # [L,B,C,C]
                tau = max(getattr(args, "cct_temp", 1.0), 1e-6)
                scores = scores / tau                                       # [L,B,C,C]

                # 构造目标：每行 i 的目标就是 i
                C = targets.size(-1)
                gt = torch.arange(C, device=device).view(1, 1, C)           # [1,1,C]
                gt = gt.expand(scores.size(0), scores.size(1), C)           # [L,B,C]

                # 展平到 (N,C) 做 CE，类别维=最后一维 C(col)
                ce_per = F.cross_entropy(
                    scores.reshape(-1, C),                                  # [L*B*C, C]
                    gt.reshape(-1),                                         # [L*B*C]
                    reduction="none",
                )                                                           # [L*B*C]
                ce_per = ce_per.view(scores.size(0), scores.size(1), C)     # [L,B,C]

                # 只在“图像中存在的类”上取平均（按每张图的正类数归一）
                mask = targets.bool().view(1, targets.size(0), C)           # [1,B,C]
                mask = mask.expand_as(ce_per)                               # [L,B,C]
                num_pos = mask.sum(dim=2).clamp_min(1)                      # [L,B]
                regularizer_loss = (ce_per * mask.float()).sum(dim=2) / num_pos  # [L,B]
                regularizer_loss = regularizer_loss.mean()                  # 标量
                # 数值稳定性：若出现非有限，置零并继续，避免影响总 loss
                if not torch.isfinite(regularizer_loss):
                    logger.warning("[ATTN] non-finite regularizer_loss detected; replacing with 0.0 to continue")
                    regularizer_loss = torch.zeros((), device=ce_per.device, dtype=ce_per.dtype)

                metric_logger.update(attn_loss=regularizer_loss.item())
                total_loss = total_loss + w_attn * regularizer_loss


            # --- Patch-level 分类损失（PATCH） ---
            if patch_outputs is not None:
                pat_loss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                metric_logger.update(pat_loss=pat_loss.item())
                total_loss = total_loss + w_patch * pat_loss

            # --- B1: Token-level 结构性监督 ---
            if getattr(args, "b1_enable", False) and b1_bundle is not None:
                x_token_cls, mask_tok = b1_bundle  # (B,T,K), (B,T)
                eps = float(getattr(args, "b1_prob_eps", 1e-6))
                w_ent = float(getattr(args, "w_ent", 0.05))
                w_neg = float(getattr(args, "w_neg", 0.02))
                w_var = float(getattr(args, "w_var", 0.0))
                margin = float(getattr(args, "b1_margin", 0.1))

                # 概率与掩码
                valid_tok = mask_tok.unsqueeze(-1)   # (B,T,1) bool
                valid_tok_f = valid_tok.float()

                # 若上游产生了 NaN/Inf 的 token logits，先替换为 0，避免 sigmoid 后继续传播
                if iter_idx == 0 and torch.isnan(x_token_cls).any():
                    logger.warning("[B1] NaN detected in raw token logits; replacing with 0 for stability")
                x_token_cls = torch.nan_to_num(x_token_cls, nan=0.0, posinf=0.0, neginf=0.0)

                p = torch.sigmoid(x_token_cls)
                p = p.clamp(min=eps, max=1.0 - eps)  # (B,T,K)

                # 正类熵项：在 FP32 下计算 log，且对无效 token 使用 p=0.5（最大熵）以避免 0*NaN 传播
                with torch.amp.autocast('cuda', enabled=False):
                    p32 = p.float()
                    p32 = torch.where(valid_tok_f > 0, p32, torch.full_like(p32, 0.5))
                    ent32 = -(p32 * torch.log(p32) + (1.0 - p32) * torch.log(1.0 - p32))  # (B,T,K)
                    ent32 = torch.nan_to_num(ent32, nan=0.0, posinf=0.0, neginf=0.0)
                ent = ent32.to(p.dtype) # (B,T,K)

                # Token 维度均值（仅有效 token） -> (B,K)
                denom_tok = valid_tok_f.sum(dim=1).clamp_min(1.0)           # (B,1)
                ent_tok_mean = (ent * valid_tok_f).sum(dim=1) / denom_tok   # (B,K)
                # 类维度：仅正类 -> (B,)
                tgt_f = targets.float()
                pos_cnt = tgt_f.sum(dim=1).clamp_min(1.0)                   # (B,)
                L_ent = ((ent_tok_mean * tgt_f).sum(dim=1) / pos_cnt).mean()
                # 数值稳定性：若 L_ent 非有限，置零
                if not torch.isfinite(L_ent):
                    logger.warning("[B1] non-finite L_ent detected; replacing with 0.0")
                    L_ent = torch.zeros((), device=ent_tok_mean.device, dtype=ent_tok_mean.dtype)

                # 负类边际：抑制负类的 token 最高概率超过 margin
                p_masked = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).masked_fill(~valid_tok, 0.0)
                p_hat = p_masked.max(dim=1).values                          # (B,K)
                neg_mask = (targets <= 0).float()                            # (B,K)
                L_neg = ((F.relu(p_hat - margin) * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1.0)).mean()
                # 数值稳定性：若 L_neg 非有限，置零
                if not torch.isfinite(L_neg):
                    logger.warning("[B1] non-finite L_neg detected; replacing with 0.0")
                    L_neg = torch.zeros((), device=p_hat.device, dtype=p_hat.dtype)

                # 区域方差项：按规划先置 0
                L_var = torch.zeros((), device=p.device, dtype=p.dtype)

                L_B1 = w_ent * L_ent + w_neg * L_neg + w_var * L_var

                # 守护：若 B1 分支数值出现非有限，置零并给出警告，避免触发全局 early-exit
                if not torch.isfinite(L_B1):
                    logger.warning("[B1] non-finite L_B1 detected; replacing with 0.0 to continue")
                    L_B1 = torch.zeros((), device=p.device, dtype=p.dtype)

                # warmup 调度：前 n 个 epoch 线性 0->1
                warm_ep = int(getattr(args, "b1_warmup_epochs", 1))
                if warm_ep > 0:
                    alpha = min(1.0, max(0.0, float(epoch) / float(warm_ep)))
                else:
                    alpha = 1.0
                total_loss = total_loss + alpha * L_B1

                # 记录日志
                metric_logger.update(b1_ent=L_ent.item(), b1_neg=L_neg.item(), b1_var=L_var.item(), b1_loss=L_B1.item())

                # TensorBoard: 仅在每个 epoch 的第一个 iteration 记录一次直方图，避免过多 I/O
                if writer is not None and iter_idx == 0:
                    try:
                        global_step = global_step_base + iter_idx
                        # 仅正类的均值/最大概率分布
                        pos_mask = (targets > 0).bool()  # (B,K)
                        # token 维度的最大概率 (B,K)
                        p_hat_pos = p_hat[pos_mask]
                        # token 维度的均值概率 (B,K)
                        p_mean_tok = (p * valid_tok_f).sum(dim=1) / denom_tok  # (B,K)
                        p_mean_pos = p_mean_tok[pos_mask]
                        # token 熵（逐 token），仅有效 token & 正类
                        pos_mask_broadcast = pos_mask.unsqueeze(1)  # (B,1,K)
                        ent_pos_all = ent[valid_tok.expand_as(ent) & pos_mask_broadcast.expand_as(ent)]
                        # 写入直方图（若为空则跳过）
                        if ent_pos_all.numel() > 0:
                            writer.add_histogram("B1/entropy_pos_tokens", ent_pos_all.detach().float().cpu().numpy(), global_step)
                        if p_hat_pos.numel() > 0:
                            writer.add_histogram("B1/p_max_pos", p_hat_pos.detach().float().cpu().numpy(), global_step)
                        if p_mean_pos.numel() > 0:
                            writer.add_histogram("B1/p_mean_pos", p_mean_pos.detach().float().cpu().numpy(), global_step)
                        # 有效 token 比例
                        valid_ratio = valid_tok_f.mean().item()
                        writer.add_scalar("B1/valid_token_ratio", valid_ratio, global_step)
                    except Exception as _e:
                        logger.warning(f"[tensorboard] histogram logging failed: {_e}")

            # --- L_sp: Superpixel alignment loss (region consistency) ---
            if getattr(args, "sp_align_enable", False) and getattr(args, "sp_align_loss_weight", 0.0) > 0.0:
                # 获取参数
                base_w = float(getattr(args, "sp_align_loss_weight", 0.0))
                apply_to = str(getattr(args, "sp_align_apply_to", "pos")).lower()
                warmup = int(getattr(args, "sp_align_warmup_epochs", 0))
                start_ep = int(getattr(args, "sp_align_start_epoch", 0))
                tau = float(getattr(args, "sp_overlap_tau", 0.4))
                min_area = int(getattr(args, "sp_min_area", 0))
                
                # 计算调度权重
                if epoch < start_ep:
                    sched = 0.0
                elif warmup > 0 and epoch < start_ep + warmup:
                    sched = float(epoch - start_ep + 1) / float(warmup)
                else:
                    sched = 1.0
                eff_w = base_w * sched
                
                # 初始化L_sp损失
                L_sp = torch.tensor(0.0, device=device)
                
                if eff_w > 0.0:
                    # 获取token概率 - 优先使用B1的token logits，否则需要模型支持返回token
                    if getattr(args, "sp_align_enable", False) and b1_bundle is not None:
                        x_token_cls, mask_tok = b1_bundle  # (B,T,K), (B,T)
                        p_tok = torch.sigmoid(x_token_cls)  # (B,T,K)
                    else:
                        # 如果没有B1，需要模型支持返回token logits
                        logger.warning("[L_sp] B1 not enabled, L_sp loss requires token logits. Skipping L_sp computation.")
                        p_tok = None
                    
                    if p_tok is not None:
                        tgt_f = targets.float()  # (B,K)
                        
                        L_sp_total = p_tok.new_zeros(())
                        valid_batches = 0
                        
                        # 处理每个batch
                        for b in range(p_tok.size(0)):
                            # 检查DINO标签是否存在
                            if dino_labels is None or dino_labels[b] is None:
                                continue
                                
                            spix_b = spix_labels[b]  # (H,W)
                            dino_b = dino_labels[b]  # (H,W)
                            
                            if spix_b.dim() != 2 or dino_b.dim() != 2:
                                continue
                            
                            # 计算重合权重矩阵
                            try:
                                W_masked = compute_overlap_weights(spix_b, dino_b, tau=tau, min_area=min_area)
                                if W_masked is None:
                                    continue
                                
                                # 获取token概率
                                probs_b = p_tok[b]  # (T,K)
                                
                                # 应用类别掩码
                                if apply_to == "pos":
                                    cls_mask_b = (tgt_f[b] > 0).float()
                                else:
                                    cls_mask_b = torch.ones(probs_b.size(1), 
                                                          device=probs_b.device, 
                                                          dtype=probs_b.dtype)
                                
                                # 计算区域一致性损失
                                l_b = compute_region_consistency_loss(probs_b, W_masked, cls_mask_b)
                                
                                if torch.isfinite(l_b):
                                    L_sp_total = L_sp_total + l_b
                                    valid_batches += 1
                                    
                            except Exception as e:
                                logger.warning(f"[L_sp] Error computing loss for batch {b}: {e}")
                                continue
                        
                        if valid_batches > 0:
                            L_sp = L_sp_total / float(valid_batches)
                            
                            # 数值稳定性检查
                            if not torch.isfinite(L_sp):
                                logger.warning("[L_sp] non-finite L_sp detected; replacing with 0.0")
                                L_sp = torch.zeros((), device=L_sp_total.device, dtype=L_sp_total.dtype)
                
                # 添加到总损失
                total_loss = total_loss + eff_w * L_sp
                
                # 记录L_sp损失（只在启用时记录）
                metric_logger.update(sp_loss=L_sp.item())
                
                # TensorBoard记录
                if writer is not None and iter_idx == 0:
                    try:
                        global_step = global_step_base + iter_idx
                        writer.add_scalar("Loss/sp_alignment", L_sp.item(), global_step)
                        writer.add_scalar("Loss/sp_weight", eff_w, global_step)
                    except Exception as e:
                        logger.warning(f"[L_sp] TensorBoard logging failed: {e}")

        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            logger.warning(f"[train] non-finite total loss {loss_value} at epoch={epoch}, iter={iter_idx}; skipping this step")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()

        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            total_loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        iter_idx += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, writer=None, epoch=None):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    # 用于累积验证指标：正类平均熵和负类最大响应
    entropy_pos_sum = 0.0
    entropy_pos_count = 0
    pmax_neg_sum = 0.0
    pmax_neg_count = 0

    # 检查是否需要计算B1指标
    compute_b1_metrics = (args is not None and 
                          (getattr(args, "b1_enable", False) or getattr(args, "sp_align_enable", False)))

    # for images, spix_labels, dino_labels, target in metric_logger.log_every(data_loader, 10, header):
    for images, spix_labels, _, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        spix_labels = spix_labels.to(device, non_blocking=True)
        # dino_labels在评估时不需要，但需要处理数据格式
        # if dino_labels is not None:
        #     dino_labels = dino_labels.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.amp.autocast('cuda'):
            # 如果需要计算B1指标，传入return_b1参数
            if compute_b1_metrics:
                output = model(images, spix_labels, return_b1=True)
            else:
                output = model(images, spix_labels)

            patch_output = None
            c_output = None
            b1_bundle = None
            
            if len(output) == 2:
                output, patch_output = output
            elif len(output) == 3:
                output, c_output, patch_output = output
            elif len(output) == 4:
                output, c_output, patch_output, b1_bundle = output

            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters["mAP"].update(np.mean(mAP_list), n=batch_size)

            # 计算B1指标：正类平均熵和负类最大响应
            if compute_b1_metrics and b1_bundle is not None:
                try:
                    x_token_cls, mask_tok = b1_bundle  # (B,T,K), (B,T)
                    eps = 1e-8
                    
                    # 概率与掩码
                    valid_tok = mask_tok.unsqueeze(-1)   # (B,T,1) bool
                    valid_tok_f = valid_tok.float()
                    
                    # 处理NaN/Inf
                    x_token_cls = torch.nan_to_num(x_token_cls, nan=0.0, posinf=0.0, neginf=0.0)
                    p = torch.sigmoid(x_token_cls)
                    p = p.clamp(min=eps, max=1.0 - eps)  # (B,T,K)
                    
                    # 计算熵：正类平均熵 \bar{H}
                    with torch.amp.autocast('cuda', enabled=False):
                        p32 = p.float()
                        p32 = torch.where(valid_tok_f > 0, p32, torch.full_like(p32, 0.5))
                        ent32 = -(p32 * torch.log(p32) + (1.0 - p32) * torch.log(1.0 - p32))  # (B,T,K)
                        ent32 = torch.nan_to_num(ent32, nan=0.0, posinf=0.0, neginf=0.0)
                    ent = ent32.to(p.dtype)  # (B,T,K)
                    
                    # Token维度均值（仅有效token） -> (B,K)
                    denom_tok = valid_tok_f.sum(dim=1).clamp_min(1.0)  # (B,1)
                    ent_tok_mean = (ent * valid_tok_f).sum(dim=1) / denom_tok  # (B,K)
                    
                    # 类维度：仅正类 -> (B,)
                    tgt_f = target.float()
                    pos_mask = (tgt_f > 0)  # (B,K) bool
                    pos_cnt = tgt_f.sum(dim=1).clamp_min(1.0)  # (B,)
                    
                    # 计算每个batch的正类平均熵
                    batch_entropy_pos = ((ent_tok_mean * tgt_f).sum(dim=1) / pos_cnt)  # (B,)
                    batch_entropy_pos = batch_entropy_pos[torch.isfinite(batch_entropy_pos)]
                    
                    if batch_entropy_pos.numel() > 0:
                        entropy_pos_sum += batch_entropy_pos.sum().item()
                        entropy_pos_count += batch_entropy_pos.numel()
                    
                    # 计算负类最大响应 \bar{P}_{max}^{-}
                    p_masked = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).masked_fill(~valid_tok, 0.0)
                    p_hat = p_masked.max(dim=1).values  # (B,K) 每个类在所有token上的最大概率
                    
                    neg_mask = (tgt_f <= 0).float()  # (B,K) float mask for negative classes
                    neg_cnt = neg_mask.sum(dim=1).clamp_min(1.0)  # (B,) 每个样本的负类数量
                    
                    # 计算每个batch的负类平均最大响应：对负类的p_hat求平均
                    batch_pmax_neg = ((p_hat * neg_mask).sum(dim=1) / neg_cnt)  # (B,)
                    batch_pmax_neg = batch_pmax_neg[torch.isfinite(batch_pmax_neg)]
                    
                    if batch_pmax_neg.numel() > 0:
                        pmax_neg_sum += batch_pmax_neg.sum().item()
                        pmax_neg_count += batch_pmax_neg.numel()
                        
                except Exception as e:
                    logger.warning(f"[evaluate] Failed to compute B1 metrics: {e}")

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        "* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            mAP=metric_logger.mAP, losses=metric_logger.loss
        )
    )

    # 计算epoch级别的指标平均值
    result_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if compute_b1_metrics:
        if entropy_pos_count > 0:
            entropy_pos_mean = entropy_pos_sum / entropy_pos_count
            result_dict['entropy_pos_mean'] = entropy_pos_mean
            if writer is not None and epoch is not None:
                writer.add_scalar("val/entropy_pos_mean", entropy_pos_mean, epoch)
        else:
            result_dict['entropy_pos_mean'] = 0.0
            
        if pmax_neg_count > 0:
            pmax_neg_mean = pmax_neg_sum / pmax_neg_count
            result_dict['pmax_neg_mean'] = pmax_neg_mean
            if writer is not None and epoch is not None:
                writer.add_scalar("val/pmax_neg_mean", pmax_neg_mean, epoch)
        else:
            result_dict['pmax_neg_mean'] = 0.0

    return result_dict


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Generating attention maps:"
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # Prepare CRF export dirs if needed
    export_crf = getattr(args, "out_crf", None) is not None
    crf_temps = list(getattr(args, "crf_save_temps", [1.0, 12.0])) if export_crf else []
    crf_out_dirs = {}
    if export_crf:
        base = Path(args.out_crf)
        for t in crf_temps:
            # suffix rule, e.g., _1, _12
            try:
                t_int = int(round(float(t)))
                suffix = f"_{t_int}"
            except Exception:
                suffix = f"_{str(t).replace('.', '_')}"
            out_dir_t = Path(str(base) + suffix)
            out_dir_t.mkdir(parents=True, exist_ok=True)
            crf_out_dirs[suffix] = out_dir_t

    model.eval()

    # 扫描已存在的 CAM 文件（断点续传支持）
    existing_cams = set()
    if args.cam_npy_dir is not None:
        cam_npy_path = Path(args.cam_npy_dir)
        if cam_npy_path.exists():
            for npy_file in cam_npy_path.glob("*.npy"):
                existing_cams.add(npy_file.stem)  # 文件名（不含扩展名）
            logger.info(f"Found {len(existing_cams)} existing CAM files, will skip them")

    # 获取数据集和图像列表
    dataset = data_loader.dataset
    # 直接从 dataset 获取图像列表，兼容 VOC12 和 COCO
    if hasattr(dataset, 'img_name_list'):
        img_list_data = dataset.img_name_list
    else:
        # 降级方案：从文件读取（仅用于旧版数据集）
        img_list_raw = open(os.path.join(args.img_list, "train_aug_id.txt")).readlines()
        img_list_data = [x.strip() for x in img_list_raw]
    total_images = len(img_list_data)
    skipped_count = sum(1 for name in img_list_data if name in existing_cams)
    to_process_count = total_images - skipped_count
    index = 0  # 用于追踪当前处理的图像索引
    
    # 使用 tqdm 进度条显示处理进度
    progress_interval = int(getattr(args, "cam_progress_interval", 50))
    logger.info(f"Starting CAM generation with batch processing (total images: {total_images})")
    logger.info(f"Total images: {total_images}, Already processed: {skipped_count}, To process: {to_process_count}")
    
    pbar = tqdm(
        data_loader,
        total=total_images,
        desc="CAM Generation",
        unit="img",
        miniters=max(1, progress_interval),
        dynamic_ncols=True
    )

    for msf_img_list, msf_sp_list, target, annotation in pbar:

        # ---- 选择一个参考尺度的图像与超像素（默认第一个：未翻转的最初尺度） ----
        # 处理批处理的多尺度数据：msf_img_list[0] 可能是堆叠张量或张量列表
        img_ref_raw = msf_img_list[0]
        
        # 如果是列表（不同尺寸），则堆叠；如果已经是张量，直接使用
        if isinstance(img_ref_raw, list):
            # 列表中的每个元素是一张图像，需要堆叠
            img_ref = torch.stack([img.to(device, non_blocking=True) for img in img_ref_raw], dim=0)
        else:
            img_ref = img_ref_raw.to(device, non_blocking=True)
        
        # 支持批处理：img_ref 现在是 (B,C,H,W)
        if img_ref.dim() == 3:
            images1 = img_ref.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        elif img_ref.dim() == 4:
            images1 = img_ref  # [B,C,H,W]
        else:
            raise RuntimeError(f"Unexpected img_ref.dim()={img_ref.dim()}")

        spix_ref_raw = msf_sp_list[0]
        
        # 处理超像素标签（同样可能是列表或张量）
        if isinstance(spix_ref_raw, list):
            spix_ref = torch.stack([sp.to(device, non_blocking=True) for sp in spix_ref_raw], dim=0)
        else:
            spix_ref = spix_ref_raw.to(device, non_blocking=True)
        
        # 规范到 (B,H,W)
        if spix_ref.dim() == 4 and spix_ref.size(1) == 1:
            spix_ref = spix_ref.squeeze(1)
        elif spix_ref.dim() == 2:
            spix_ref = spix_ref.unsqueeze(0)
        elif spix_ref.dim() != 3:
            raise RuntimeError(f"Unexpected spix_ref shape: {tuple(spix_ref.shape)}")

        target = target.to(device, non_blocking=True)
        
        # 处理 annotation（可能是堆叠的张量或列表）
        # 如果是列表（尺寸不同），保持为列表；如果是张量（尺寸相同），保持为张量
        if isinstance(annotation, list):
            # 列表格式：每个元素是一张图像的 annotation，保持为列表
            annotation_processed = [ann.to(device, non_blocking=True) for ann in annotation]
        else:
            # 已经是张量（所有图像尺寸相同），直接转到 GPU
            annotation_processed = annotation.to(device, non_blocking=True)

        # 批处理：获取实际批次大小和对应的图像名称列表
        batch_size = images1.shape[0]
        img_name_list = [img_list_data[index + i] for i in range(batch_size)]
        
        # 断点续传：检查是否需要跳过整个批次
        skip_batch = all(name in existing_cams for name in img_name_list)
        if skip_batch:
            index += batch_size
            continue
        
        index += batch_size

        # 批处理：反归一化所有图像（(B,H,W,3)）
        img_temp = images1.permute(0,2,3,1).detach().cpu().numpy()  # (B,H,W,3)
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.0
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.0
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.0

        H, W = orig_images.shape[1], orig_images.shape[2]
        
        # 获取超像素标签（用于边界绘制）
        sp_labels = None
        if getattr(args, 'draw_sp_boundaries', False):
            # 将超像素标签转换为 numpy 数组 (B,H,W)
            sp_labels = spix_ref.detach().cpu().numpy()
            # 如果尺寸不匹配，需要调整（但通常应该匹配，因为 spix_ref 和 images1 来自同一尺度）

        with torch.amp.autocast('cuda'):
            cam_list = []
            vitattn_list = []
            cam_maps = None

            # ---- 多尺度 CAM ----
            # 遍历 msf_img_list：偶数索引=原图，奇数索引=水平翻转图
            for s_idx in range(len(msf_img_list)):
                # 处理可能是列表或张量的数据
                img_s_raw = msf_img_list[s_idx]
                if isinstance(img_s_raw, list):
                    img_s = torch.stack([img.to(device, non_blocking=True) for img in img_s_raw], dim=0)
                else:
                    img_s = img_s_raw.to(device, non_blocking=True)
                
                # 标准化为 [B,C,H,W]
                if img_s.dim() == 3:
                    img_s = img_s.unsqueeze(0)
                assert img_s.dim() == 4, f"img_s dim={img_s.dim()}, shape={tuple(img_s.shape)}"

                spix_s_raw = msf_sp_list[s_idx]
                if isinstance(spix_s_raw, list):
                    spix_s = torch.stack([sp.to(device, non_blocking=True) for sp in spix_s_raw], dim=0)
                else:
                    spix_s = spix_s_raw.to(device, non_blocking=True)
                
                # 允许 (H,W), (1,H,W), (B,H,W), (B,1,H,W)
                if spix_s.dim() == 2:
                    spix_s = spix_s.unsqueeze(0)  # (1,H,W)
                elif spix_s.dim() == 4 and spix_s.size(1) == 1:
                    pass  # (B,1,H,W)
                elif spix_s.dim() == 3:
                    pass  # (1,H,W) 或 (B,H,W)
                else:
                    raise RuntimeError(f"Unexpected spix_s shape: {tuple(spix_s.shape)}")

                _w = img_s.shape[2] - img_s.shape[2] % args.patch_size
                _h = img_s.shape[3] - img_s.shape[3] % args.patch_size
                w_featmap = _w // args.patch_size
                h_featmap = _h // args.patch_size

                # 前向
                output, cls_attentions, patch_attn = model(
                    img_s,
                    spix_label=spix_s,
                    return_att=True,
                    n_layers=args.layer_index,
                    attention_type=args.attention_type,
                )
                patch_attn = torch.sum(patch_attn, dim=0)

                # p2p 亲和力细化（可选）
                if args.patch_attn_refine:
                    A = patch_attn.squeeze(0).squeeze(0)  # (T,T)
                    spix_label_ref = spix_s
                    if spix_label_ref.dim() == 2:
                        spix_label_ref = spix_label_ref.unsqueeze(0)
                    if spix_label_ref.shape[-2:] != cls_attentions.shape[-2:]:
                        spix_label_ref = (
                            F.interpolate(spix_label_ref.unsqueeze(0).float(),
                                          size=cls_attentions.shape[-2:],
                                          mode="nearest")
                            .squeeze(0)
                            .long()
                        )
                    else:
                        spix_label_ref = spix_label_ref.long()

                    cls_attentions = refine_cam_with_spatch_affinity(
                        cls_attentions, spix_label_ref, A,
                        output=output, pos_thr=0.5, alpha=0.85, steps=10, is_logit=True,
                    )

                # 插值到参考原图尺寸（H,W）
                cls_attentions = F.interpolate(
                    cls_attentions, size=(H, W), mode="bilinear", align_corners=False
                )  # (B,C,H,W) - 支持批处理

                # 保持在 GPU 上做逐尺度屏蔽与融合（批处理版本）
                target_mask = target.to(cls_attentions.device).view(batch_size, args.nb_classes, 1, 1).float()
                cls_att_t = cls_attentions * target_mask  # (B,C,H,W)

                # 若该尺度为翻转分支（奇数 index），再沿宽度反转回来（GPU 上 torch.flip）
                if s_idx % 2 == 1:
                    cls_att_t = torch.flip(cls_att_t, dims=[-1])
                cam_list.append(cls_att_t)
                vitattn_list.append(cam_maps)

            # ---- 多尺度融合 ----
            # 批处理：sum_cam 的形状是 (B, C, H, W)
            sum_cam = torch.stack(cam_list, dim=0).sum(dim=0)   # (B,C,H,W) torch
            # 注意：cam_list 中每个元素已经是 (B,C,H,W)，stack 后是 (num_scales, B, C, H, W)
            # sum 后变为 (B,C,H,W)

            # --- 在下面加入增强处理 ---

            # 批处理：对每张图像单独执行超像素增强处理
            # 依据 超像素-gt 重叠 对 CAM 做按类增强/抑制（超像素尺度）
            enable_amp = bool(getattr(args, "cam_enable_amp", True))   # 是否进行增强（ratio>=gamma）
            enable_suppr = bool(getattr(args, "cam_enable_suppr", True))  # 是否进行抑制（ratio<=theta）
            gamma = float(getattr(args, "cam_sp_gamma", 0.45))    # 重叠上阈值
            theta = float(getattr(args, "cam_sp_theta", 0.38))    # 重叠下阈值
            alpha = float(getattr(args, "cam_amp_alpha", 0.06))  # 提升比例（乘以 1+alpha）
            beta  = float(getattr(args, "cam_amp_beta", 0.08))  # 抑制比例（乘以 1-beta）
            # 安全裁剪
            alpha = max(0.0, min(1.0, alpha))
            beta  = max(0.0, min(1.0, beta))
            gamma = max(0.0, min(1.0, gamma))
            theta = max(0.0, min(1.0, theta))
            if theta > gamma:
                theta, gamma = gamma, theta  # 兜底：确保 theta <= gamma

            Hc, Wc = sum_cam.shape[-2:]
            ann_label_offset = int(getattr(args, "ann_label_offset", 1))

            # 批次循环：对每张图像单独处理超像素增强
            for b in range(batch_size):
                # 对齐 annotation 到 (Hc,Wc)
                # 处理列表或张量格式的 annotation
                if isinstance(annotation_processed, list):
                    ann = annotation_processed[b]  # 列表：直接取元素（已在 GPU 上）
                else:
                    ann = annotation_processed[b]  # 张量：索引取元素
                if ann.dim() == 2:
                    ann = ann  # (H,W)
                elif ann.dim() == 3 and ann.shape[0] == 1:
                    ann = ann.squeeze(0)
                else:
                    raise RuntimeError(f"Unexpected annotation dim for batch {b}: {ann.dim()}")
                
                if (ann.shape[-2], ann.shape[-1]) != (Hc, Wc):
                    ann = F.interpolate(ann.unsqueeze(0).unsqueeze(0).float(), size=(Hc, Wc), mode="nearest").squeeze(0).squeeze(0).long()
                else:
                    ann = ann.long()

                # 对齐超像素图到 (Hc,Wc)
                spix = spix_ref[b]  # (H,W)
                if (spix.shape[-2], spix.shape[-1]) != (Hc, Wc):
                    spix = F.interpolate(spix.unsqueeze(0).unsqueeze(0).float(), size=(Hc, Wc), mode="nearest").squeeze(0).squeeze(0).long()
                else:
                    spix = spix.long()

                # 展平为向量，统计每个超像素大小
                sp_flat = spix.view(-1)
                T = int(sp_flat.max().item()) + 1
                cnt_total = torch.bincount(sp_flat, minlength=T).to(sum_cam.dtype)
                cnt_total = cnt_total.clamp_min(1.0)

                # 当前样本存在的类别
                target_vec = target[b]
                target_vec = target_vec.to(sum_cam.device)

                # 对每个类别计算增强系数
                for cls_ind in range(args.nb_classes):
                    if target_vec[cls_ind] <= 0:
                        continue
                    seg_id = cls_ind + ann_label_offset
                    pos_mask = (ann == seg_id).view(-1)  # 与该类重叠的像素

                    # 计算每个超像素与该类的重叠比例 ratio ∈ [0,1]
                    if pos_mask.any():
                        cnt_pos = torch.bincount(sp_flat[pos_mask], minlength=T).to(sum_cam.dtype)
                    else:
                        cnt_pos = torch.zeros(T, dtype=sum_cam.dtype, device=sum_cam.device)
                    ratio = cnt_pos / cnt_total

                    # 根据阈值确定每个超像素的缩放系数
                    scale_sp = torch.ones(T, dtype=sum_cam.dtype, device=sum_cam.device)
                    if enable_amp and alpha > 0:
                        scale_sp = torch.where(ratio >= gamma, 1.0 + alpha, scale_sp)
                    if enable_suppr and beta > 0:
                        scale_sp = torch.where(ratio <= theta, 1.0 - beta, scale_sp)

                    # 回投到像素尺度并应用到对应类别 CAM
                    scale_map = scale_sp[sp_flat].view(Hc, Wc)  # (Hc,Wc)
                    sum_cam[b:b+1, cls_ind:cls_ind+1, :, :] = sum_cam[b:b+1, cls_ind:cls_ind+1, :, :] * scale_map.unsqueeze(0).unsqueeze(0)

            # --- 在上面加入增强处理 ---

            # 三个实现功能开关：
            # cam_smooth: 光滑（默认关，可用 args 开关覆盖）
            # cam_edge_aware: 保边（可选）
            # cam_crf: CRF 后处理（可选）
            args.cam_smooth = False
            args.cam_edge_aware = False
            args.cam_crf = False

            # ---- A: 光滑（默认关，可用 args 开关覆盖）----
            if getattr(args, "cam_smooth", False):
                target_res = getattr(args, "cam_target_res", 14)
                gauss_ks   = getattr(args, "cam_gauss_ks", 7)
                gauss_sig  = getattr(args, "cam_gauss_sigma", 2.0)
                with torch.no_grad():
                    sum_cam = _smooth_like_mctformer_4d(sum_cam, target=target_res, ks=gauss_ks, sigma=gauss_sig)

            # ---- B: 保边（可选）----
            if getattr(args, "cam_edge_aware", False):
                img4d = torch.from_numpy(orig_images/255.0).to(sum_cam.device, dtype=sum_cam.dtype)
                img4d = img4d.permute(0,3,1,2).contiguous()
                sum_cam = edge_aware_smooth_4d(
                    sum_cam, img4d,
                    iters=getattr(args,"cam_edge_iters",5),
                    lam=getattr(args,"cam_edge_lam",0.2),
                    alpha=getattr(args,"cam_edge_alpha",10.0)
                )

            # ---- 分类分数（sigmoid）----
            output = torch.sigmoid(output)

        # ======================= 可视化与保存 =======================
        if args.visualize_cls_attn:
            # 批处理：对每张图像单独处理保存和可视化
            for b in range(batch_size):
                # 获取当前图像的文件名、CAM、输出和annotation
                img_name = img_name_list[b]
                
                # 断点续传：跳过已处理的图像
                if img_name in existing_cams:
                    continue
                
                # 将 target 压缩成 1D 类别向量
                target_vec = target[b]
                if (target_vec.sum()) > 0:
                    cam_dict = {}
                    norm_cam = np.zeros((args.nb_classes, H, W))

                    # ----------（可选）一次性初始化 CRF ----------
                    use_crf = getattr(args, "cam_crf", False)
                    # ensure CRF is initialized if we need to export CRF seeds
                    need_crf = bool(use_crf) or bool(export_crf)
                    d = None
                    crfW = getattr(args, "crf_width",  W)
                    crfH = getattr(args, "crf_height", H)
                    if need_crf:
                        try:
                            img_np = orig_images[b].astype(np.uint8)  # (H,W,3)
                            if (crfW, crfH) != (W, H):
                                img_np_crf = _resize_img_uint8(img_np, crfW, crfH)
                            else:
                                img_np_crf = img_np
                            d = _init_densecrf_2d(
                                img_np_crf,
                                sxy1=getattr(args,"crf_sxy1",80),
                                srgb1=getattr(args,"crf_srgb1",3),
                                w1=getattr(args,"crf_w1",3),
                                sxy2=getattr(args,"crf_sxy2",3),
                                w2=getattr(args,"crf_w2",5),
                            )
                            crf_iters = getattr(args, "crf_iters", 5)
                            crf_temp  = getattr(args, "crf_temp", 1.0)
                        except Exception as e:
                            print(f"[WARN] CRF init failed, disabled: {e}")
                            need_crf = False
                            use_crf = False
                            export_crf = False
                        # ------------------------------------------------

                    # prepare export dicts per temperature
                    export_dicts = {}
                    if export_crf and need_crf:
                        bg_thr = float(getattr(args, "seed_bg_thr", 0.39))
                        for t in crf_temps:
                            try:
                                t_int = int(round(float(t)))
                                suffix = f"_{t_int}"
                            except Exception:
                                suffix = f"_{str(t).replace('.', '_')}"
                            export_dicts[suffix] = {}
                            # Insert background channel (key 0) with constant map first (preserve insertion order)
                            export_dicts[suffix][0] = np.full((H, W), bg_thr, dtype=np.float32)

                    for cls_ind in range(args.nb_classes):
                        if target_vec[cls_ind] > 0:  # 只对存在的类别处理/保存
                            # 基础 CAM 概率（未做可视化 CRF 之前）
                            prob_base = sum_cam[b, cls_ind, :]  # (H,W) torch
                            prob_base = (prob_base - prob_base.min()) / (prob_base.max() - prob_base.min() + 1e-8)

                            cls_attention = prob_base.clone()

                            # ——（可选）CRF（仅用于可视化） —— #
                            if use_crf and need_crf and d is not None:
                                prob = cls_attention.detach().cpu().numpy()  # (H,W)
                                prob_refined = _crf_refine_prob(
                                    d, prob,
                                    out_W=W, out_H=H,
                                    crf_W=crfW, crf_H=crfH,
                                    iters=crf_iters, temp=crf_temp, normalize=True, only_scale_if_gt1=True
                                )
                                cls_attention = torch.from_numpy(prob_refined).to(sum_cam.device, dtype=sum_cam.dtype)
                                cls_attention = (cls_attention - cls_attention.min()) / (
                                    cls_attention.max() - cls_attention.min() + 1e-8
                                )

                            cls_attention = cls_attention.detach().cpu().numpy()

                            # 保存/可视化
                            cls_score = format(output[b, cls_ind].cpu().numpy(), ".3f")
                            cam_dict[cls_ind] = cls_attention
                            norm_cam[cls_ind] = cls_attention

                            # 导出 CRF seeds（不同温度）
                            if export_crf and need_crf and d is not None:
                                base_prob = prob_base.detach().cpu().numpy()
                                for t in crf_temps:
                                    try:
                                        t_int = int(round(float(t)))
                                        suffix = f"_{t_int}"
                                    except Exception:
                                        suffix = f"_{str(t).replace('.', '_')}"
                                    pr = _crf_refine_prob(
                                        d, base_prob,
                                        out_W=W, out_H=H,
                                        crf_W=crfW, crf_H=crfH,
                                        iters=crf_iters, temp=float(t), normalize=True
                                    )
                                    export_dicts[suffix][cls_ind + 1] = pr

                            # 只有在指定了 --save-cam-png 参数时才保存 PNG 图片
                            if args.attention_dir is not None and getattr(args, 'save_cam_png', False):
                                fname = os.path.join(
                                    args.attention_dir,
                                    img_name + "_" + str(cls_ind) + "_" + str(cls_score) + ".png",
                                )
                                # 获取当前图像对应的超像素标签
                                sp_label = sp_labels[b] if sp_labels is not None else None
                                show_cam_on_image(
                                    orig_images[b], 
                                    cls_attention, 
                                    fname,
                                    sp_label=sp_label,
                                    draw_boundaries=getattr(args, 'draw_sp_boundaries', False)
                                )

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + ".npy"), cam_dict)

                    # 写出各温度 CRF 字典
                    if export_crf and export_dicts:
                        for suffix, dic in export_dicts.items():
                            out_dir = crf_out_dirs[suffix]
                            np.save(os.path.join(out_dir, img_name + ".npy"), dic)

    pbar.close()
    logger.info(f"Completed CAM generation. Total processed: {total_images}/{total_images}")
    metric_logger.synchronize_between_processes()
    return


@torch.no_grad()
def generate_attention_maps_ms_batch(data_loader, model, device, args):
    """
    分组批处理版本的 CAM 生成函数
    
    实现按尺寸分组的批处理推理，提升 GPU 利用率。
    参考 dino-sp-generator 的分组批处理方案。
    """
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # Prepare CRF export dirs if needed
    export_crf = getattr(args, "out_crf", None) is not None
    crf_temps = list(getattr(args, "crf_save_temps", [1.0, 12.0])) if export_crf else []
    crf_out_dirs = {}
    if export_crf:
        base = Path(args.out_crf)
        for t in crf_temps:
            try:
                t_int = int(round(float(t)))
                suffix = f"_{t_int}"
            except Exception:
                suffix = f"_{str(t).replace('.', '_')}"
            out_dir_t = Path(str(base) + suffix)
            out_dir_t.mkdir(parents=True, exist_ok=True)
            crf_out_dirs[suffix] = out_dir_t

    model.eval()

    # 扫描已存在的 CAM 文件（断点续传支持）
    existing_cams = set()
    if args.cam_npy_dir is not None:
        cam_npy_path = Path(args.cam_npy_dir)
        if cam_npy_path.exists():
            for npy_file in cam_npy_path.glob("*.npy"):
                existing_cams.add(npy_file.stem)  # 文件名（不含扩展名）
            logger.info(f"Found {len(existing_cams)} existing CAM files, will skip them")

    # 获取数据集和图像列表
    dataset = data_loader.dataset
    # 直接从 dataset 获取图像列表，兼容 VOC12 和 COCO
    if hasattr(dataset, 'img_name_list'):
        img_name_list = dataset.img_name_list
        total_images = len(img_name_list)
    else:
        # 降级方案：从文件读取（仅用于旧版数据集）
        img_list = open(os.path.join(args.img_list, "train_aug_id.txt")).readlines()
        img_name_list = [x.strip() for x in img_list]
        total_images = len(img_name_list)
    skipped_count = sum(1 for name in img_name_list if name in existing_cams)
    to_process_count = total_images - skipped_count
    cam_batch_size = getattr(args, "cam_batch_size", 8)
    
    logger.info(f"Starting CAM generation with grouped batch processing (total images: {total_images}, batch_size: {cam_batch_size})")
    logger.info(f"Total images: {total_images}, Already processed: {skipped_count}, To process: {to_process_count}")
    
    # 使用 tqdm 进度条显示处理进度
    progress_interval = int(getattr(args, "cam_progress_interval", 50))
    pbar = tqdm(
        total=total_images,
        desc="CAM Generation",
        unit="img",
        miniters=max(1, progress_interval),
        dynamic_ncols=True
    )
    
    # 批量处理：每次处理 cam_batch_size 张图像
    for batch_start in range(0, total_images, cam_batch_size):
        batch_end = min(batch_start + cam_batch_size, total_images)
        batch_indices = list(range(batch_start, batch_end))
        
        # 断点续传：过滤掉已处理的图像
        batch_indices_filtered = [idx for idx in batch_indices 
                                   if img_name_list[idx] not in existing_cams]
        
        # 如果批次全部已处理，跳过
        if len(batch_indices_filtered) == 0:
            pbar.update(batch_end - batch_start)
            continue
        
        # === 步骤1: 批量加载图像并按尺寸分组 ===
        batch_data = []
        for idx in batch_indices_filtered:
            try:
                msf_img_list, msf_sp_list, target, annotation = dataset[idx]
                
                # 获取参考尺度的图像尺寸（第一个尺度，未翻转）
                img_ref_raw = msf_img_list[0]
                if isinstance(img_ref_raw, list):
                    img_ref_sample = img_ref_raw[0] if len(img_ref_raw) > 0 else None
                else:
                    img_ref_sample = img_ref_raw
                
                if img_ref_sample is None:
                    logger.warning(f"Skipping image {idx}: invalid reference image")
                    continue
                
                # 获取图像尺寸 (H, W)
                if img_ref_sample.dim() == 3:
                    H, W = img_ref_sample.shape[1], img_ref_sample.shape[2]
                elif img_ref_sample.dim() == 4:
                    H, W = img_ref_sample.shape[2], img_ref_sample.shape[3]
                else:
                    logger.warning(f"Skipping image {idx}: unexpected image dim {img_ref_sample.dim()}")
                    continue
                
                batch_data.append({
                    'idx': idx,
                    'msf_img_list': msf_img_list,
                    'msf_sp_list': msf_sp_list,
                    'target': target,
                    'annotation': annotation,
                    'shape': (H, W),
                    'name': img_name_list[idx]
                })
            except Exception as e:
                logger.error(f"Failed to load image {idx}: {e}")
                continue
        
        if len(batch_data) == 0:
            continue
        
        # === 步骤2: 按尺寸分组 ===
        shape_groups = defaultdict(list)
        for data_idx, data in enumerate(batch_data):
            shape_groups[data['shape']].append(data_idx)
        
        # === 步骤3: 对每个尺寸组分别进行批处理推理 ===
        for shape, group_indices in shape_groups.items():
            H, W = shape
            
            # 收集该尺寸组的所有数据
            group_msf_img_list = [batch_data[i]['msf_img_list'] for i in group_indices]
            group_msf_sp_list = [batch_data[i]['msf_sp_list'] for i in group_indices]
            group_targets = torch.stack([batch_data[i]['target'] for i in group_indices])
            group_annotations = [batch_data[i]['annotation'] for i in group_indices]
            group_names = [batch_data[i]['name'] for i in group_indices]
            
            group_batch_size = len(group_indices)
            
            # 处理参考尺度的图像和超像素
            img_ref_list = []
            spix_ref_list = []
            for i in group_indices:
                img_ref_raw = batch_data[i]['msf_img_list'][0]
                spix_ref_raw = batch_data[i]['msf_sp_list'][0]
                
                if isinstance(img_ref_raw, list):
                    img_ref = img_ref_raw[0] if len(img_ref_raw) > 0 else None
                else:
                    img_ref = img_ref_raw
                
                if isinstance(spix_ref_raw, list):
                    spix_ref = spix_ref_raw[0] if len(spix_ref_raw) > 0 else None
                else:
                    spix_ref = spix_ref_raw
                
                if img_ref is None or spix_ref is None:
                    continue
                
                img_ref_list.append(img_ref)
                spix_ref_list.append(spix_ref)
            
            if len(img_ref_list) == 0:
                continue
            
            # Stack 相同尺寸的图像
            images1 = torch.stack([img.to(device, non_blocking=True) for img in img_ref_list], dim=0)
            
            # Stack 超像素标签
            spix_ref_stacked = []
            for sp in spix_ref_list:
                if sp.dim() == 2:
                    sp = sp.unsqueeze(0)
                elif sp.dim() == 4 and sp.size(1) == 1:
                    sp = sp.squeeze(1)
                spix_ref_stacked.append(sp)
            spix_ref = torch.stack([sp.to(device, non_blocking=True) for sp in spix_ref_stacked], dim=0)
            
            group_targets = group_targets.to(device, non_blocking=True)
            
            # 处理 annotation
            annotation_processed = []
            for ann in group_annotations:
                if isinstance(ann, torch.Tensor):
                    annotation_processed.append(ann.to(device, non_blocking=True))
                else:
                    annotation_processed.append(ann)
            
            # 反归一化图像
            img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
            orig_images = np.zeros_like(img_temp)
            orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.0
            orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.0
            orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.0
            
            # 获取超像素标签（用于边界绘制）
            sp_labels = None
            if getattr(args, 'draw_sp_boundaries', False):
                # 将超像素标签转换为 numpy 数组 (B,H,W)
                sp_labels = spix_ref.detach().cpu().numpy()
            
            with torch.amp.autocast('cuda'):
                # 对每个尺度进行推理
                cam_list_all_scales = []
                
                num_scales = len(group_msf_img_list[0])
                for s_idx in range(num_scales):
                    # 收集该尺度下所有图像
                    scale_img_list = []
                    scale_sp_list = []
                    
                    for i in range(group_batch_size):
                        img_s_raw = group_msf_img_list[i][s_idx]
                        spix_s_raw = group_msf_sp_list[i][s_idx]
                        
                        if isinstance(img_s_raw, list):
                            img_s = img_s_raw[0] if len(img_s_raw) > 0 else None
                        else:
                            img_s = img_s_raw
                        
                        if isinstance(spix_s_raw, list):
                            spix_s = spix_s_raw[0] if len(spix_s_raw) > 0 else None
                        else:
                            spix_s = spix_s_raw
                        
                        if img_s is None or spix_s is None:
                            continue
                        
                        scale_img_list.append(img_s)
                        scale_sp_list.append(spix_s)
                    
                    if len(scale_img_list) == 0:
                        continue
                    
                    # Stack 为 batch
                    img_s = torch.stack([img.to(device, non_blocking=True) for img in scale_img_list], dim=0)
                    
                    # 处理超像素标签
                    spix_s_list = []
                    for sp in scale_sp_list:
                        if sp.dim() == 2:
                            sp = sp.unsqueeze(0)
                        elif sp.dim() == 4 and sp.size(1) == 1:
                            sp = sp.squeeze(1)
                        spix_s_list.append(sp)
                    spix_s = torch.stack([sp.to(device, non_blocking=True) for sp in spix_s_list], dim=0)
                    
                    # 计算特征图尺寸
                    _w = img_s.shape[2] - img_s.shape[2] % args.patch_size
                    _h = img_s.shape[3] - img_s.shape[3] % args.patch_size
                    
                    # 前向推理
                    output, cls_attentions, patch_attn = model(
                        img_s,
                        spix_label=spix_s,
                        return_att=True,
                        n_layers=args.layer_index,
                        attention_type=args.attention_type,
                    )
                    patch_attn = torch.sum(patch_attn, dim=0)
                    
                    # p2p 亲和力细化（可选）
                    if args.patch_attn_refine:
                        A = patch_attn.squeeze(0).squeeze(0)
                        spix_label_ref = spix_s
                        if spix_label_ref.dim() == 2:
                            spix_label_ref = spix_label_ref.unsqueeze(0)
                        if spix_label_ref.shape[-2:] != cls_attentions.shape[-2:]:
                            spix_label_ref = (
                                F.interpolate(spix_label_ref.unsqueeze(0).float(),
                                              size=cls_attentions.shape[-2:],
                                              mode="nearest")
                                .squeeze(0)
                                .long()
                            )
                        else:
                            spix_label_ref = spix_label_ref.long()
                        
                        cls_attentions = refine_cam_with_spatch_affinity(
                            cls_attentions, spix_label_ref, A,
                            output=output, pos_thr=0.5, alpha=0.85, steps=10, is_logit=True,
                        )
                    
                    # 插值到参考原图尺寸
                    cls_attentions = F.interpolate(
                        cls_attentions, size=(H, W), mode="bilinear", align_corners=False
                    )
                    
                    # 应用类别掩码
                    target_mask = group_targets.view(group_batch_size, args.nb_classes, 1, 1).float()
                    cls_att_t = cls_attentions * target_mask
                    
                    # 若该尺度为翻转分支，反转回来
                    if s_idx % 2 == 1:
                        cls_att_t = torch.flip(cls_att_t, dims=[-1])
                    
                    cam_list_all_scales.append(cls_att_t)
                
                # 多尺度融合
                if len(cam_list_all_scales) > 0:
                    sum_cam = torch.stack(cam_list_all_scales, dim=0).sum(dim=0)
                else:
                    logger.warning(f"No valid CAM for shape {shape}, skipping")
                    continue
                
                # 超像素增强处理
                enable_amp = bool(getattr(args, "cam_enable_amp", True))
                enable_suppr = bool(getattr(args, "cam_enable_suppr", True))
                gamma = float(getattr(args, "cam_sp_gamma", 0.45))
                theta = float(getattr(args, "cam_sp_theta", 0.38))
                alpha_amp = float(getattr(args, "cam_amp_alpha", 0.06))
                beta = float(getattr(args, "cam_amp_beta", 0.08))
                
                alpha_amp = max(0.0, min(1.0, alpha_amp))
                beta = max(0.0, min(1.0, beta))
                gamma = max(0.0, min(1.0, gamma))
                theta = max(0.0, min(1.0, theta))
                if theta > gamma:
                    theta, gamma = gamma, theta
                
                Hc, Wc = sum_cam.shape[-2:]
                ann_label_offset = int(getattr(args, "ann_label_offset", 1))
                
                # 对每张图像单独处理超像素增强
                for b in range(group_batch_size):
                    ann = annotation_processed[b]
                    if ann.dim() == 2:
                        ann = ann
                    elif ann.dim() == 3 and ann.shape[0] == 1:
                        ann = ann.squeeze(0)
                    
                    if (ann.shape[-2], ann.shape[-1]) != (Hc, Wc):
                        ann = F.interpolate(ann.unsqueeze(0).unsqueeze(0).float(), size=(Hc, Wc), mode="nearest").squeeze(0).squeeze(0).long()
                    else:
                        ann = ann.long()
                    
                    spix = spix_ref[b]
                    if (spix.shape[-2], spix.shape[-1]) != (Hc, Wc):
                        spix = F.interpolate(spix.unsqueeze(0).unsqueeze(0).float(), size=(Hc, Wc), mode="nearest").squeeze(0).squeeze(0).long()
                    else:
                        spix = spix.long()
                    
                    sp_flat = spix.view(-1)
                    T = int(sp_flat.max().item()) + 1
                    cnt_total = torch.bincount(sp_flat, minlength=T).to(sum_cam.dtype)
                    cnt_total = cnt_total.clamp_min(1.0)
                    
                    target_vec = group_targets[b]
                    target_vec = target_vec.to(sum_cam.device)
                    
                    for cls_ind in range(args.nb_classes):
                        if target_vec[cls_ind] <= 0:
                            continue
                        seg_id = cls_ind + ann_label_offset
                        pos_mask = (ann == seg_id).view(-1)
                        
                        if pos_mask.any():
                            cnt_pos = torch.bincount(sp_flat[pos_mask], minlength=T).to(sum_cam.dtype)
                        else:
                            cnt_pos = torch.zeros(T, dtype=sum_cam.dtype, device=sum_cam.device)
                        ratio = cnt_pos / cnt_total
                        
                        scale_sp = torch.ones(T, dtype=sum_cam.dtype, device=sum_cam.device)
                        if enable_amp and alpha_amp > 0:
                            scale_sp = torch.where(ratio >= gamma, 1.0 + alpha_amp, scale_sp)
                        if enable_suppr and beta > 0:
                            scale_sp = torch.where(ratio <= theta, 1.0 - beta, scale_sp)
                        
                        scale_map = scale_sp[sp_flat].view(Hc, Wc)
                        sum_cam[b:b+1, cls_ind:cls_ind+1, :, :] = sum_cam[b:b+1, cls_ind:cls_ind+1, :, :] * scale_map.unsqueeze(0).unsqueeze(0)
                
                # 分类分数
                output = torch.sigmoid(output)
                
                # 保存结果
                if args.visualize_cls_attn:
                    for b in range(group_batch_size):
                        img_name = group_names[b]
                        
                        # 断点续传：再次检查（双重保险）
                        if img_name in existing_cams:
                            continue
                        
                        target_vec = group_targets[b]
                        
                        if (target_vec.sum()) > 0:
                            cam_dict = {}
                            norm_cam = np.zeros((args.nb_classes, H, W))
                            
                            use_crf = getattr(args, "cam_crf", False)
                            need_crf = bool(use_crf) or bool(export_crf)
                            d = None
                            crfW = getattr(args, "crf_width", W)
                            crfH = getattr(args, "crf_height", H)
                            
                            if need_crf:
                                try:
                                    img_np = orig_images[b].astype(np.uint8)
                                    if (crfW, crfH) != (W, H):
                                        img_np_crf = _resize_img_uint8(img_np, crfW, crfH)
                                    else:
                                        img_np_crf = img_np
                                    d = _init_densecrf_2d(
                                        img_np_crf,
                                        sxy1=getattr(args, "crf_sxy1", 80),
                                        srgb1=getattr(args, "crf_srgb1", 3),
                                        w1=getattr(args, "crf_w1", 3),
                                        sxy2=getattr(args, "crf_sxy2", 3),
                                        w2=getattr(args, "crf_w2", 5),
                                    )
                                    crf_iters = getattr(args, "crf_iters", 5)
                                    crf_temp = getattr(args, "crf_temp", 1.0)
                                except Exception as e:
                                    logger.warning(f"CRF init failed for {img_name}, disabled: {e}")
                                    need_crf = False
                                    use_crf = False
                                    export_crf = False
                            
                            export_dicts = {}
                            if export_crf and need_crf:
                                bg_thr = float(getattr(args, "seed_bg_thr", 0.39))
                                for t in crf_temps:
                                    try:
                                        t_int = int(round(float(t)))
                                        suffix = f"_{t_int}"
                                    except Exception:
                                        suffix = f"_{str(t).replace('.', '_')}"
                                    export_dicts[suffix] = {}
                                    export_dicts[suffix][0] = np.full((H, W), bg_thr, dtype=np.float32)
                            
                            for cls_ind in range(args.nb_classes):
                                if target_vec[cls_ind] > 0:
                                    prob_base = sum_cam[b, cls_ind, :].detach().cpu().numpy()
                                    prob_base = (prob_base - prob_base.min()) / (prob_base.max() - prob_base.min() + 1e-8)
                                    
                                    cls_attention = prob_base.copy()
                                    
                                    if use_crf and need_crf and d is not None:
                                        prob_refined = _crf_refine_prob(
                                            d, cls_attention,
                                            out_W=W, out_H=H,
                                            crf_W=crfW, crf_H=crfH,
                                            iters=crf_iters, temp=crf_temp, normalize=True, only_scale_if_gt1=True
                                        )
                                        cls_attention = prob_refined
                                        cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                                    
                                    cls_score = format(output[b, cls_ind].cpu().numpy(), ".3f")
                                    cam_dict[cls_ind] = cls_attention
                                    norm_cam[cls_ind] = cls_attention
                                    
                                    if export_crf and need_crf and d is not None:
                                        base_prob = prob_base.copy()
                                        for t in crf_temps:
                                            try:
                                                t_int = int(round(float(t)))
                                                suffix = f"_{t_int}"
                                            except Exception:
                                                suffix = f"_{str(t).replace('.', '_')}"
                                            pr = _crf_refine_prob(
                                                d, base_prob,
                                                out_W=W, out_H=H,
                                                crf_W=crfW, crf_H=crfH,
                                                iters=crf_iters, temp=float(t), normalize=True
                                            )
                                            export_dicts[suffix][cls_ind + 1] = pr
                                    
                                    if args.attention_dir is not None and getattr(args, 'save_cam_png', False):
                                        fname = os.path.join(
                                            args.attention_dir,
                                            img_name + "_" + str(cls_ind) + "_" + str(cls_score) + ".png",
                                        )
                                        # 获取当前图像对应的超像素标签
                                        sp_label = sp_labels[b] if sp_labels is not None else None
                                        show_cam_on_image(
                                            orig_images[b], 
                                            cls_attention, 
                                            fname,
                                            sp_label=sp_label,
                                            draw_boundaries=getattr(args, 'draw_sp_boundaries', False)
                                        )
                            
                            if args.cam_npy_dir is not None:
                                np.save(os.path.join(args.cam_npy_dir, img_name + ".npy"), cam_dict)
                            
                            if export_crf and export_dicts:
                                for suffix, dic in export_dicts.items():
                                    out_dir = crf_out_dirs[suffix]
                                    np.save(os.path.join(out_dir, img_name + ".npy"), dic)
        
        # 更新 tqdm 进度条
        pbar.update(len(batch_data))
    
    pbar.close()
    logger.info(f"Completed CAM generation with batch processing. Total processed: {total_images}/{total_images}")
    return

def show_cam_on_image(img, mask, save_path, sp_label=None, draw_boundaries=False):
    # 保持与原逻辑一致：
    # 1) 将 img 归一化到 [0,1]
    # 2) 使用 "jet" 颜色映射为 mask 生成热力图（与 OpenCV 的 uint8 量化行为对齐）
    # 3) 将热力图(模拟 BGR 顺序)与原图相加并归一化
    # 4) 保存到磁盘（保持与 cv2.imwrite 等价的可视显示效果）
    img = np.asarray(img, dtype=np.float32) / 255.0
    mask = np.asarray(mask, dtype=np.float32)
    
    # 添加数据验证和清理
    if np.any(np.isnan(mask)) or np.any(np.isinf(mask)):
        print(f"警告：mask 中包含无效值，正在清理...")
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    
    mask = np.clip(mask, 0.0, 1.0)

    # OpenCV 的 applyColorMap 会对 0..1 的 mask 先乘以 255 并量化到 uint8，再做映射。
    # 为了保持一致，我们也先做量化再映射（向下取整，与 OpenCV 保持一致）。
    quant = (mask * 255.0).astype(np.uint8)
    
    # 确保 quant/255.0 在有效范围内
    normalized_quant = quant / 255.0
    if np.any(np.isnan(normalized_quant)) or np.any(np.isinf(normalized_quant)):
        print(f"警告：归一化后的 quant 包含无效值，正在修复...")
        normalized_quant = np.nan_to_num(normalized_quant, nan=0.0, posinf=1.0, neginf=0.0)
    
    heatmap_rgb = JET_CMAP(normalized_quant)[..., :3].astype(np.float32)  # [H, W, 3], RGB, 0..1

    # OpenCV 返回 BGR，而原代码未做通道转换，直接与 RGB 的 img 相加后再用 cv2.imwrite(BGR) 写盘。
    # 为复现该行为，这里先把热力图转为 BGR 再与 img 相加；保存前再从 BGR 转回 RGB。
    heatmap_bgr = heatmap_rgb[..., ::-1]

    cam = heatmap_bgr + img
    max_val = cam.max()
    if max_val > 0:
        cam = cam / max_val
    cam = np.clip(cam, 0.0, 1.0)

    cam_uint8 = (cam * 255.0).astype(np.uint8)
    cam_rgb = cam_uint8[..., ::-1]
    
    # 如果启用边界绘制且提供了超像素标签，则绘制边界
    if draw_boundaries and sp_label is not None:
        # 确保超像素标签是 numpy 数组
        sp_label = np.asarray(sp_label, dtype=np.int32)
        
        # 确保超像素标签与图像尺寸匹配
        if sp_label.shape[:2] != cam_rgb.shape[:2]:
            # 使用最近邻插值调整尺寸
            try:
                from scipy.ndimage import zoom
                h_ratio = cam_rgb.shape[0] / sp_label.shape[0]
                w_ratio = cam_rgb.shape[1] / sp_label.shape[1]
                sp_label = zoom(sp_label, (h_ratio, w_ratio), order=0, mode='nearest').astype(np.int32)
            except ImportError:
                # 如果 scipy 不可用，使用简单的最近邻插值
                from PIL import Image
                sp_pil = Image.fromarray(sp_label.astype(np.uint16), mode='I;16')
                sp_pil = sp_pil.resize((cam_rgb.shape[1], cam_rgb.shape[0]), Image.NEAREST)
                sp_label = np.array(sp_pil, dtype=np.int32)
        
        # 提取边界（使用 thick 模式生成较粗的边界线）
        boundaries = find_boundaries(sp_label, mode='thick')
        
        # 将边界绘制到图像上（白色边界）
        cam_rgb[boundaries] = [255, 255, 255]

    skio.imsave(save_path, cam_rgb)


def _row_normalize(A, eps=1e-8):
    return A / (A.sum(dim=-1, keepdim=True) + eps)


def refine_cam_with_spatch_affinity(
    cls_attention,  # (B, C, H, W)  通常 B=1
    spix_label,  # (B, H, W) 或 (H, W)，long，id ∈ [0, T-1]
    patch_attn,  # (T, T) 或 (B, T, T)；B=1 时传 (T,T) 最简
    output=None,  # (B, C)；若是 logits，就设 is_logit=True
    pos_thr=0.5,
    alpha=0.85,
    steps=10,
    add_self_loop=True,
    is_logit=True,
):
    """
    注入思路：像素→超像素均值池化 → 在 spatch 图上做 RWR 扩散 → 回投到像素。
    返回同形状 (B, C, H, W)。
    """
    if spix_label.dim() == 2:
        spix_label = spix_label.unsqueeze(0)
    spix_label = spix_label.long()  # 确保是 long

    B, C, H, W = cls_attention.shape
    P = H * W

    # --- 亲和力 A: (B,T,T)
    if patch_attn.dim() == 2:  # (T,T)
        A = patch_attn.unsqueeze(0).to(cls_attention.dtype)
    elif patch_attn.dim() == 3:  # (B,T,T)
        A = patch_attn.to(cls_attention.dtype)
    else:  # 兼容 [1,1,T,T]
        A = patch_attn.squeeze(0).squeeze(0).unsqueeze(0).to(cls_attention.dtype)
    T = A.size(-1)
    if add_self_loop:
        I = torch.eye(T, device=A.device, dtype=A.dtype)
        A = A + I.unsqueeze(0)
    K = _row_normalize(A)  # (B,T,T)

    # --- 展平像素
    cam_px = cls_attention.flatten(2).permute(0, 2, 1)  # (B,P,C)
    sp = spix_label.view(B, -1)  # (B,P)
    # 安全检查：sp id 不越界
    assert int(sp.max().item()) < T, "spix_label 的 id 超出 patch_attn 的 T"

    # 1) 像素→token：对每个超像素做均值池化
    sum_tc = torch.zeros(B, T, C, device=cam_px.device, dtype=cam_px.dtype)
    cnt_t = torch.zeros(B, T, device=cam_px.device, dtype=cam_px.dtype)
    for b in range(B):
        sum_tc[b].index_add_(0, sp[b], cam_px[b])  # (T,C) += (P,C)
        cnt = torch.bincount(sp[b], minlength=T).to(cam_px.dtype)
        cnt_t[b] = cnt.clamp_min(1.0)
    M = sum_tc / cnt_t.unsqueeze(-1)  # (B,T,C)
    M0 = M.clone()

    # 只扩散阳性类（用于最终回退 gating）
    if output is not None:
        prob = output.sigmoid() if is_logit else output.clamp(0, 1)
        pos_mask = prob > pos_thr  # (B,C)
    else:
        pos_mask = None

    # 2) token 图上随机游走带重启（RWR）
    for _ in range(steps):
        M = alpha * torch.bmm(K, M) + (1 - alpha) * M0  # (B,T,C)

    # 3) token→像素 回投
    refined_px = torch.empty(B, P, C, device=cam_px.device, dtype=cam_px.dtype)
    for b in range(B):
        refined_px[b] = M[b].index_select(0, sp[b])  # (P,C)
    refined = refined_px.permute(0, 2, 1).view(B, C, H, W)

    # 对非阳性类回退到原 CAM（可选）
    if pos_mask is not None:
        refined = torch.where(
            pos_mask.unsqueeze(-1).unsqueeze(-1), refined, cls_attention
        )
    return refined

# ===========================================
# Tools: A 平滑、B 保边、C CRF（PIL 版本，无 OpenCV）
# ===========================================
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image  # <- 用 PIL 替代 OpenCV

# ---------- 通用 resize 工具（PIL） ----------
def _resize_img_uint8(img_uint8_hwc: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    RGB uint8 HWC -> resize -> RGB uint8 HWC
    """
    im = Image.fromarray(img_uint8_hwc, mode="RGB")
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

def _resize_prob_float(prob_hw: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    float32 HxW in [0,1] -> resize (bilinear) -> float32 HxW
    用 PIL 的 'F' 模式避免量化
    """
    # PIL 期望 (W,H)
    im = Image.fromarray(prob_hw.astype(np.float32), mode="F")
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    out = np.asarray(im, dtype=np.float32)
    # 保守裁剪
    out = np.clip(out, 0.0, 1.0)
    # 确保返回的数组是C-contiguous的
    if not out.flags['C_CONTIGUOUS']:
        out = np.ascontiguousarray(out)
    return out

# ---------- A) MCTformer 风格平滑 ----------
def _gaussian_kernel1d(ks: int, sigma: float, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    k = torch.exp(-(ax**2) / (2 * (sigma**2)))
    return k / (k.sum() + 1e-12)

def _gaussian_blur_depthwise(x4d: torch.Tensor, ks: int = 7, sigma: float = 2.0):
    if ks <= 1 or sigma <= 0:
        return x4d
    B, C, H, W = x4d.shape
    k1d = _gaussian_kernel1d(ks, sigma, x4d.device, x4d.dtype)
    kx = k1d.view(1,1,1,ks)
    ky = k1d.view(1,1,ks,1)
    y = F.pad(x4d, (ks//2, ks//2, ks//2, ks//2), mode="reflect")
    y = F.conv2d(y, ky.expand(C,1,ks,1), groups=C)
    y = F.conv2d(y, kx.expand(C,1,1,ks), groups=C)
    return y

def _smooth_like_mctformer_4d(x4d: torch.Tensor, target: int = 14, ks: int = 7, sigma: float = 2.0):
    """(B,C,H,W) -> down(target) -> up(H,W) -> gaussian -> per-class [0,1]"""
    B, C, H, W = x4d.shape
    y = F.interpolate(x4d, size=(target, target), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    y = _gaussian_blur_depthwise(y, ks=ks, sigma=sigma)
    y_min = y.amin(dim=(2,3), keepdim=True)
    y_max = y.amax(dim=(2,3), keepdim=True)
    return (y - y_min) / (y_max - y_min + 1e-6)



# ---------- B) 边缘保形（引导扩散，纯 torch） ----------
@torch.no_grad()
def edge_aware_smooth_4d(cam4d: torch.Tensor,
                         img4d: torch.Tensor,
                         iters: int = 5,
                         lam: float = 0.2,
                         alpha: float = 10.0):
    """
    cam4d: (1,C,H,W) 概率/响应
    img4d: (1,3,H,W) [0,1] 浮点
    """
    B, C, H, W = cam4d.shape
    x = cam4d

    def shift(y, dy, dx):
        return F.pad(y, (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0)),
                     mode="replicate")[:, :, max(-dy,0):H+min(dy,0), max(-dx,0):W+min(dx,0)]

    def w_from_img(dy, dx):
        neigh = shift(img4d, dy, dx)
        col = (img4d - neigh).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
        return torch.exp(-alpha * col).clamp_(0.0, 1.0)

    Wp, Wm, Rp, Rm = w_from_img(1,0), w_from_img(-1,0), w_from_img(0,1), w_from_img(0,-1)

    for _ in range(iters):
        diffs, weights = [], []
        for (dy,dx,w) in [(1,0,Wp),(-1,0,Wm),(0,1,Rp),(0,-1,Rm)]:
            neigh = shift(x, dy, dx)
            wC = w.expand(B, C, H, W)
            diffs.append((neigh - x) * wC)
            weights.append(wC)
        Wsum = torch.stack(weights,0).sum(0).clamp_min(1e-6)
        x = x + lam * torch.stack(diffs,0).sum(0) / Wsum

    x_min = x.amin(dim=(2,3), keepdim=True)
    x_max = x.amax(dim=(2,3), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)

# ---------- C) CRF（pydensecrf，CPU；一次性初始化 pairwise） ----------
def _init_densecrf_2d(rgb_uint8, sxy1=80, srgb1=3, w1=3, sxy2=3, w2=5):
    import pydensecrf.densecrf as dcrf
    H, W = rgb_uint8.shape[:2]
    
    # 确保rgb_uint8是C-contiguous的
    if not rgb_uint8.flags['C_CONTIGUOUS']:
        rgb_uint8 = np.ascontiguousarray(rgb_uint8)
    
    d = dcrf.DenseCRF2D(W, H, 2)
    # 只添加一次 pairwise
    d.addPairwiseBilateral(sxy=sxy1, srgb=srgb1, rgbim=rgb_uint8, compat=w1)
    d.addPairwiseGaussian(sxy=sxy2, compat=w2)
    return d

def _crf_infer_with_unary(d, prob_hw, iters=5, eps=1e-6):
    """只替换 unary，然后 inference；prob_hw: (H,W) float [0,1]"""
    H, W = prob_hw.shape
    # 确保输入数组是C-contiguous的
    if not prob_hw.flags['C_CONTIGUOUS']:
        prob_hw = np.ascontiguousarray(prob_hw)
    
    pf = np.clip(prob_hw, eps, 1 - eps).reshape(-1)
    pb = 1.0 - pf
    # 确保pb也在有效范围内，避免数值不稳定
    pb = np.clip(pb, eps, 1 - eps)
    U = np.vstack([-np.log(pb), -np.log(pf)]).astype(np.float32)  # (2, H*W)
    
    # 确保U也是C-contiguous的
    if not U.flags['C_CONTIGUOUS']:
        U = np.ascontiguousarray(U)
    
    d.setUnaryEnergy(U)
    Q = d.inference(iters)
    return np.array(Q)[1, :].reshape(H, W).astype(np.float32)

def _crf_refine_prob(d, prob_hw, out_W, out_H, crf_W, crf_H, iters=5, temp=1.0, normalize=True, eps=1e-6, only_scale_if_gt1=False):
    """
    Refine a single-class probability map with DenseCRF and return float32 HxW map.
    - d: initialized DenseCRF2D object
    - prob_hw: numpy array HxW in [0,1]
    - out_W,out_H: output size to return
    - crf_W,crf_H: internal DenseCRF working size (can be equal to out size)
    - temp: temperature applied BEFORE building unary; matches legacy behavior
      (if only_scale_if_gt1=True, only apply when temp>1.0; else apply when temp!=1.0)
    - normalize: min-max to [0,1] at the end (preserves prior behavior)
    """
    prob = np.asarray(prob_hw, dtype=np.float32)
    if not prob.flags['C_CONTIGUOUS']:
        prob = np.ascontiguousarray(prob)

    # Decide whether to apply temperature scaling (keep legacy conditions)
    try:
        t_val = float(temp)
    except Exception:
        t_val = 1.0
    do_scale = (t_val > 1.0) if only_scale_if_gt1 else (abs(t_val - 1.0) > 1e-8)
    if do_scale:
        prob = np.clip(prob, eps, 1.0 - eps) ** (1.0 / t_val)
        if not prob.flags['C_CONTIGUOUS']:
            prob = np.ascontiguousarray(prob)

    # Resize to CRF working size if needed
    if (crf_W, crf_H) != (out_W, out_H):
        prob_small = _resize_prob_float(prob, crf_W, crf_H)
    else:
        prob_small = prob

    pr = _crf_infer_with_unary(d, prob_small, iters=iters, eps=eps)

    # Resize back to output size if needed
    if (crf_W, crf_H) != (out_W, out_H):
        pr = _resize_prob_float(pr, out_W, out_H)

    if normalize:
        mn, mx = pr.min(), pr.max()
        pr = (pr - mn) / (mx - mn + eps)

    if not pr.flags['C_CONTIGUOUS']:
        pr = np.ascontiguousarray(pr)
    return pr


# ---- L_sp superpixel alignment loss helper functions ----

def compute_overlap_weights(spix_labels, dino_labels, tau=0.4, min_area=0, eps=1e-8):
    """
    计算SLIC超像素与DINO超像素的重合权重矩阵
    
    Args:
        spix_labels: (H, W) SLIC超像素标签
        dino_labels: (H, W) DINO超像素标签（允许 -1 作为缺失占位）
        tau: 重合阈值
        min_area: DINO区域最小面积
        eps: 数值稳定性常数
    
    Returns:
        W_masked: (T, R) 权重矩阵，已应用阈值过滤；若无有效像素则返回 None
    """
    # 仅使用 dino_labels 中的有效像素（>=0），忽略缺失占位 -1
    valid_mask = dino_labels >= 0
    if valid_mask.sum() == 0:
        return None
    dino_valid = dino_labels.clone()
    dino_valid[~valid_mask] = 0  # 临时填充，不参与统计

    # 连续化 DINO 标签，仅基于有效像素
    r_vals = torch.unique(dino_valid[valid_mask])
    if len(r_vals) == 0:
        return None

    # 创建连续映射
    remap = {int(v.item()): i for i, v in enumerate(r_vals)}
    dino_cont = dino_valid.clone()
    for old_val, new_val in remap.items():
        dino_cont[dino_cont == old_val] = new_val

    T = int(spix_labels.max().item()) + 1
    R = int(dino_cont.max().item()) + 1

    # 展平，仅在有效位置统计重合
    s = spix_labels.view(-1)
    d = dino_cont.view(-1)
    m = valid_mask.view(-1)
    if m.sum() == 0:
        return None
    idx = s[m] * R + d[m]
    if idx.numel() == 0:
        return None

    # 使用bincount计算重合像素数
    overlap = torch.bincount(idx, minlength=T * R).view(T, R).float()

    # 计算面积
    area_t = overlap.sum(dim=1, keepdim=True)  # (T, 1)
    area_r = overlap.sum(dim=0, keepdim=True)  # (1, R)

    # 计算权重
    W = overlap / (area_t.clamp_min(eps))  # (T, R)

    # 应用过滤条件
    mask = (W >= tau) & (area_r >= min_area)
    W_masked = W * mask.float()

    if W_masked.sum() == 0:
        return None
    return W_masked


def compute_region_consistency_loss(p_tok, W_masked, cls_mask, eps=1e-8):
    """
    计算区域一致性损失
    
    Args:
        p_tok: (T, K) token概率
        W_masked: (T, R) 权重矩阵
        cls_mask: (K,) 类别掩码
        eps: 数值稳定性常数
    
    Returns:
        loss: 标量损失值
    """
    if W_masked is None or W_masked.sum() == 0:
        return torch.tensor(0.0, device=p_tok.device, dtype=p_tok.dtype)
    
    # 对齐 T 维：实际 sp 超像素数量可能 < 预设 T，导致与 p_tok.shape[0] 不一致
    T_p = p_tok.shape[0]
    T_w, R = W_masked.shape
    if T_w != T_p:
        if T_w < T_p:
            W_pad = torch.zeros((T_p, R), device=W_masked.device, dtype=W_masked.dtype)
            W_pad[:T_w, :] = W_masked
            W_masked = W_pad
        else:
            W_masked = W_masked[:T_p, :]
        T_w = T_p
    K = p_tok.shape[1]
    
    # 归一化权重（按列归一）
    Z = W_masked.sum(dim=0, keepdim=True).clamp_min(eps)  # (1, R)
    W_norm = W_masked / Z  # (T, R)
    
    # 计算区域均值概率
    p_bar = W_norm.t().matmul(p_tok)  # (R, K)
    
    # 计算残差
    # p_tok: (T, K) -> (T, 1, K)
    # p_bar: (R, K) -> (1, R, K)
    diff = p_tok.unsqueeze(1) - p_bar.unsqueeze(0)  # (T, R, K)
    
    # 应用类别掩码
    cls_mask = cls_mask.view(1, 1, -1)  # (1, 1, K)
    res2 = (diff * diff) * cls_mask  # (T, R, K)
    
    # 权重广播到K维
    w3 = W_masked.unsqueeze(-1)  # (T, R, 1)
    
    # 计算加权损失
    num_pairs = W_masked.sum()
    num_cls = cls_mask.sum()
    
    if num_pairs == 0 or num_cls == 0:
        return torch.tensor(0.0, device=p_tok.device, dtype=p_tok.dtype)
    
    loss = (res2 * w3).sum() / (num_pairs.clamp_min(1.0) * num_cls.clamp_min(1.0))
    
    return loss