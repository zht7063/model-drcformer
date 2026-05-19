# DRCFormer 环境复现（prepare_env）

本目录用于 **1:1 复现** 当前开发机 `py310` conda 环境，已取代项目根目录的 `requirements.txt`。

## 前提条件

- **操作系统**: Linux x86_64（推荐 Ubuntu 18.04+）
- **Python**: 3.10.20（由 `conda-env.yml` 固定）
- **GPU**: NVIDIA GPU，驱动支持 CUDA 12.1+
- **Conda**: Miniconda 或 Anaconda
- **磁盘**: 建议预留 ≥ 15GB（含 PyTorch/CUDA 依赖）

导出时的环境快照见 [`metadata.env`](metadata.env)。

## 快速安装

```bash
cd /home/ubuntu/projects/drcformer/prepare_env
bash install.sh
conda activate py310
```

指定环境名（例如测试安装）：

```bash
ENV_NAME=py310-test bash install.sh
conda activate py310-test
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `install.sh` | 一键安装入口（推荐） |
| `verify.sh` | 安装后校验（import + `main.py --help`） |
| `conda-env.yml` | conda 基座：python、pydensecrf、pytorch-cuda |
| `pip.lock` | py310 全部 pip 包精确版本（可移植） |
| `pip.exclude` | 安装 pip.lock 时跳过的包（已在前面步骤安装） |
| `metadata.env` | 导出时机器信息（只读参考） |
| `conda-env.full.yml` | conda 全量快照（同机型兜底，见下文） |

## 安装流程（install.sh 内部）

1. `conda env create/update -f conda-env.yml` — Python 3.10.20、pydensecrf（conda-forge）、pytorch-cuda 12.1
2. `pip install torch/torchvision/torchaudio` — PyTorch cu121 官方 wheel 源
3. `pip install torch-scatter` — PyG wheel 源
4. `pip install fast-slic` — 先装 Cython，再 `--no-build-isolation`
5. `pip install -r`（过滤后的 pip.lock）— 其余依赖
6. 运行 `verify.sh`

## 手动分步安装（排错用）

```bash
conda env create -n py310 -f conda-env.yml
conda activate py310

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install torch-scatter==2.1.2+pt25cu121 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

pip install Cython==3.2.4
pip install fast-slic==0.4.0 --no-build-isolation

# 过滤后安装其余包（需自行排除 pip.exclude 中的包）
grep -vFf pip.exclude pip.lock > /tmp/pip-filtered.lock
pip install -r /tmp/pip-filtered.lock
```

## 校验

```bash
conda activate py310
bash prepare_env/verify.sh
```

或对比关键包版本：

```bash
pip list --format=freeze | grep -E 'torch|numpy|timm|einops|opencv|pydensecrf|fast-slic'
```

## 常见问题

### torch-scatter 找不到版本

PyPI 默认源没有 `+pt25cu121` 后缀包，必须使用 PyG wheel：

```bash
pip install torch-scatter==2.1.2+pt25cu121 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

### pydensecrf 编译失败

本方案通过 **conda-forge** 安装预编译包（见 `conda-env.yml`），不要单独 `pip install pydensecrf`。

### fast-slic 安装失败

```bash
pip install Cython==3.2.4
pip install fast-slic==0.4.0 --no-build-isolation
```

### 网络慢 / 超时

如有本地代理，可先开启代理再执行 `install.sh`。详见 `docs/环境配置补充说明.md`。

### conda-env.full.yml 失败

`conda-env.full.yml` 为全量 conda 快照（~280 依赖），仅建议在 **相同 Ubuntu + x86_64** 上尝试：

```bash
conda env create -f conda-env.full.yml
```

失败时请使用主路径：`conda-env.yml` + `install.sh`。

## 可选依赖

| 包 | 说明 |
|----|------|
| `pycocotools` | py310 中未安装；仅 `tools/convert_coco_to_segmentation_masks.py` 需要：`pip install pycocotools` |

## 与旧 requirements.txt 的关系

根目录 `requirements.txt` 已弃用，仅保留指向本目录的说明。旧文件中的版本（如 numpy 1.26.4、huggingface-hub 0.35.3）与当前 py310 **不一致**；请以 `pip.lock` 为准。

## 更新锁定文件

在已验证可运行的 `py310` 环境中重新导出：

```bash
conda activate py310
pip list --format=freeze > prepare_env/pip.lock
conda env export -n py310 --no-builds > prepare_env/conda-env.full.yml
# 按需更新 metadata.env
```
