# DRCFormer 消融模块实现调查总结

本文对项目中用于消融实验的三个模块进行代码级梳理：

- RCP（超像素 token 映射）
- DCC（双域共识校正）
- SNS（噪声抑制）

并给出它们在训练/推理流程中的接入方式与开关参数，便于后续复查。

---

## 1. RCP：超像素 token 映射（通常常开）

### 1.1 实现位置

- 模型主体：`src/drcformer/models.py`
  - `SuperpixelVisionTransformer.tokenization()`
  - `SuperpixelVisionTransformer.prepare_tokens()`
  - `SuperpixelVisionTransformer.forward_features()`
- 超像素标签生成：`src/drcformer/datasets.py`
  - `get_spix_label()`（基于 `fast_slic`）
  - 各 Dataset 在 `__getitem__` 中构造 `sp`（SLIC 标签图）
- 配置入口：
  - `src/drcformer/main.py`：`--num-sp-tokens`
  - `src/drcformer/run_drcformer.py`：`--num_sp_tokens`

### 1.2 核心机制（代码行为）

1. 先提取像素级特征 `x`（`get_feats` + 位置编码 + 可选投影）；
2. 用 `spix_label` 将像素特征按超像素区域聚合为 token：
   - 支持 `max/min/avg/std/softmax` 等聚合方式（默认常见是 `max+avg`）；
3. token 数量由 `num_sp_tokens` 控制，标签会被 `clamp` 到合法范围，避免越界；
4. 对空簇（没有像素落入的超像素 id）构建 `mask`，并在注意力中屏蔽；
5. 将多类 `cls_token` 与超像素 token 拼接后送入 Transformer。

### 1.3 结论

RCP 是本项目的基础表示方式：把规则网格 patch token 换成超像素 token，并对空 token 做掩码处理，贯穿训练、CAM 生成与推理流程。

---

## 2. DCC：双域共识校正（DINO 区域内 token 校正）

> 代码里并未直接命名为 `DCC`，对应实现主要是训练中的 **L_sp / superpixel alignment loss**（SLIC 与 DINO 区域重合约束）。

### 2.1 实现位置

- 训练逻辑：`src/drcformer/engine.py`
  - `train_one_epoch()` 中 `# --- L_sp: Superpixel alignment loss (region consistency) ---`
  - `compute_overlap_weights()`
  - `compute_region_consistency_loss()`
- DINO 区域读取：`src/drcformer/datasets.py`
  - `VOC12Dataset._load_dino_superpixel()`
  - `COCODataset._load_dino_superpixel()`
  - 训练样本返回 `(img, sp, dino, label)`
- 配置入口：`src/drcformer/main.py`
  - `--sp-align-enable`
  - `--sp-align-loss-weight`
  - `--sp-idx-dir`
  - `--sp-overlap-tau`
  - `--sp-min-area`
  - `--sp-align-apply-to` / `--sp-align-start-epoch` / `--sp-align-warmup-epochs`
- 运行脚本默认开启：`src/drcformer/run_drcformer.py`
  - 训练命令中固定带 `--sp-align-enable`，并设置了 `w_lsp` 等参数

### 2.2 核心机制（代码行为）

1. 数据侧提供两种区域划分：
   - SLIC 超像素（模型 token 的归属）
   - DINO 区域标签（外部区域先验）
2. 在每个 batch 中，先取 token 级类别概率 `p_tok`（来自 `b1_bundle` 的 `x_token_cls -> sigmoid`）；
3. 计算 SLIC 与 DINO 的重合权重矩阵 `W_masked(T,R)`：
   - `tau` 做重合比例阈值过滤，`min_area` 做区域面积过滤；
4. 基于 `W_masked` 计算区域均值概率 `p_bar`，再约束 token 概率到区域均值（残差平方加权）；
5. 得到 `L_sp` 后以 `eff_w` 加入总损失，实现“DINO 区域内共识校正”。

### 2.3 结论

DCC 在代码中体现为 `L_sp`：利用 DINO 区域先验，让落在同一 DINO 区域的超像素 token 预测更一致，从而做跨域（SLIC token 域 vs DINO 区域域）的共识校正。

---

## 3. SNS：噪声抑制（正类熵 + 负类边际）

> 代码里对应 `B1 token-level structural supervision` 分支。

### 3.1 实现位置

- 模型返回 token 分类 logits：`src/drcformer/models.py`
  - `forward(..., return_b1=True)` 时附加返回 `(x_token_cls, mask_tok)`
- 损失计算：`src/drcformer/engine.py`
  - `train_one_epoch()` 中 `# --- B1: Token-level 结构性监督 ---`
  - `evaluate()` 中同步统计 `entropy_pos_mean` 与 `pmax_neg_mean`
- 配置入口：`src/drcformer/main.py`
  - `--b1-enable`
  - `--w-ent`（正类熵项权重）
  - `--w-neg`（负类边际项权重）
  - `--b1-margin`、`--b1-prob-eps`、`--b1-warmup-epochs`
- 运行脚本默认开启：`src/drcformer/run_drcformer.py`
  - 固定带 `--b1-enable`，并将 `w_ent/w_neg` 透传为外部可调

### 3.2 核心机制（代码行为）

1. 对 token logits 做 sigmoid 得到 `p(B,T,K)`，并用 `mask_tok` 过滤无效 token；
2. 正类熵项 `L_ent`：
   - 先算 token 熵，再仅在图像正类上聚合平均；
   - 通过 `w_ent` 控制强度（抑制正类内部不稳定/高噪响应）；
3. 负类边际项 `L_neg`：
   - 取每个负类在 token 维的最大响应 `p_hat`；
   - 对超过 margin 的部分做 hinge 惩罚；
   - 通过 `w_neg` 控制强度（压制负类伪激活）；
4. `L_B1 = w_ent * L_ent + w_neg * L_neg (+ w_var*L_var, 默认0)`，并带 warmup 后加到总损失。

### 3.3 结论

SNS 的实现就是 B1 分支中的两项核心噪声约束：
- 正类熵约束（稳正类）
- 负类边际抑制（压负类噪声）

与需求描述基本一致。

---

## 4. 三模块在流程中的关系（简表）

- **RCP**：输入建模层（像素 -> 超像素 token），基础能力，通常常开；
- **SNS(B1)**：token 概率层的噪声正则，作用在训练损失；
- **DCC(L_sp)**：引入 DINO 区域先验的区域一致性正则，也作用在训练损失。

可理解为：
- RCP 定义了 token 表达；
- SNS 和 DCC 在该 token 表达之上施加不同约束（分类噪声抑制 vs 区域共识校正）。

---

## 5. 消融开关建议（基于当前代码）

在现有参数体系下可对应为：

- **Baseline + RCP**：
  - 关闭 SNS：`--b1-enable` 不开，或令 `--w-ent 0 --w-neg 0`
  - 关闭 DCC：`--sp-align-enable` 不开，或 `--sp-align-loss-weight 0`
- **Baseline + RCP + DCC**：
  - 开 DCC：`--sp-align-enable --sp-align-loss-weight > 0`
  - 关 SNS：`--w-ent 0 --w-neg 0`（并可不启 `--b1-enable`）
- **Baseline + RCP + SNS**：
  - 开 SNS：`--b1-enable --w-ent > 0 --w-neg > 0`
  - 关 DCC：`--sp-align-loss-weight 0` 或关闭 `--sp-align-enable`
- **完整体（RCP + DCC + SNS）**：
  - 同时开启上述两组。

注：`run_drcformer.py` 当前训练命令默认同时开启了 `--b1-enable` 与 `--sp-align-enable`，做消融时需在脚本参数上显式覆盖。
