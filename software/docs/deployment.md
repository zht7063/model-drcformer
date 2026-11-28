# 部署

后端运行在项目已有的 conda `py310` 环境（`prepare_env/`）之上，只需额外安装 Web 依赖：

```bash
/home/ubuntu/miniconda3/envs/py310/bin/pip install -r software/backend/requirements.txt
```

## 配置

```bash
cp software/.env.example software/.env   # software/.env 已被 Git 忽略
```

- `SEG_API_KEY`：至少 16 个 ASCII 字符。
- `SEG_MODELS`：JSON 列表，每项 `{"id", "checkpoint", "num_sp_tokens"}`，第一项为默认模型。checkpoint 为 `main.py` 训练输出的 `checkpoint.pth`（仅含 `model` 权重，以 `weights_only=True` 加载）；VOC（20 类）或 COCO（80 类）由权重自动识别。`num_sp_tokens` 必须与训练时 `--num-sp-tokens` 一致。所有模型启动时一次性加载并常驻显存。
- `SEG_DEVICE`：`cuda:0`（默认单卡 4090），也支持 `cpu`。
- `SEG_MAX_INFERENCE_SIDE`：推理最长边，默认 512。
- `SEG_GT_DIR`：可选，VOC 2012 语义分割掩码目录（用于推理时按上传文件名自动命中 `<stem>.png` 并启用超像素-GT 增强/抑制，默认未配置即关闭）。可直接指向掩码目录（如 `.../voc2012/SegmentationClass` 或 `.../voc2012/SegmentationClassAug`），也可指向 VOC 根目录（存在 `SegmentationClass` / `SegmentationClassAug` 子目录时按 `SegmentationClass`、`SegmentationClassAug` 顺序查找）。
- 其余上传、像素、队列、TTL 限制同 `.env.example`。

检查配置（不加载模型、不探测 GPU）：

```bash
bash software/backend/run.sh --check-config
```

## 启动

服务必须在 tmux 中运行，从仓库根目录启动：

```bash
tmux new -s drcformer-seg
CUDA_VISIBLE_DEVICES=0 bash software/backend/run.sh
```

`run.sh` 读取 `software/.env`（可用 `SERVER_ENV_FILE` 覆盖），并用 `SERVER_PYTHON`（默认 `/home/ubuntu/miniconda3/envs/py310/bin/python`）执行 `python -m software.backend`。Uvicorn 单 worker，默认监听 `127.0.0.1:8000`；GPU 推理在单独线程中串行执行。

## 冒烟测试

```bash
python -m software.backend.smoke_engine --checkpoint <checkpoint.pth> --image <img.jpg> [--crf]
```
