## 环境安装（py310）

推荐使用 `prepare_env/` 一键复现当前开发环境：

```bash
cd prepare_env
bash install.sh
conda activate py310
```

详见 [`prepare_env/README.md`](prepare_env/README.md)。

## 运行前注意事项（py310）

### 训练入口与工作目录

- `src/drcformer/main.py` 使用了 `from datasets import ...`、`from engine import ...` 这类相对导入，**需要在 `src/drcformer/` 目录下运行**，否则会出现导入失败或导入到错误模块的情况。

示例：

```bash
conda activate py310
cd /home/ubuntu/projects/model-drcformer/src/drcformer
python main.py --help
```

### VOC2012Aug 数据与列表文件

- VOC 根目录（含 `JPEGImages/`, `Annotations/`, `SegmentationClassAug/` 等）在：
  - `/home/ubuntu/projects/datasets/voc2012aug`
- 图像列表与分类标签（需包含 `train_aug_id.txt` / `val_id.txt` / `cls_labels.npy`）在：
  - `/home/ubuntu/projects/datasets/voc2012aug/voc12`

训练时建议显式指定：

```bash
python main.py \
  --data-path /home/ubuntu/projects/datasets/voc2012aug \
  --img-list /home/ubuntu/projects/datasets/voc2012aug/voc12
```

### DINO 超像素（sp_idx）路径

`datasets.py` 会在训练阶段根据 `--sp_idx_dir` 去读取 `"{img_id}.npy"`。如果 `sp_idx_dir` 配错，会出现日志：
`[VOC12Dataset] DINO superpixel missing ...`，这通常是**路径不匹配**而不是数据真的缺失。

- DINO 超像素文件目录为：
  - `/home/ubuntu/projects/outputs/drcformer/sp_idx/dino/npy`

训练时建议显式指定：

```bash
python main.py \
  --sp_idx_dir /home/ubuntu/projects/outputs/drcformer/sp_idx/dino/npy
```

### 生成 DINO 区域划分（已验证）

本机可用的 DINO 权重和 VOC2012 数据分别位于：

- 权重：`/home/ubuntu/projects/weights/dino22416b.npy`
- 数据：`/home/ubuntu/projects/datasets/voc2012`
- 图像列表：`/home/ubuntu/projects/datasets/voc2012/voc12`

项目根目录已有由 `uv` 创建的 `.venv`。先做两张图的 GPU 验证：

```bash
cd /home/ubuntu/projects/model-drcformer
.venv/bin/python src/dino-sp-generator/generate_sp.py \
  --data-path /home/ubuntu/projects/datasets/voc2012 \
  --img-list /home/ubuntu/projects/datasets/voc2012/voc12 \
  --model-path /home/ubuntu/projects/weights/dino22416b.npy \
  --out-dir /tmp/drcformer-dino-sp-smoke \
  --tau 0.3 --device cuda --max-count 2 --batch-size 2 --num-workers 2 \
  --crf-only --no-png --no-resume
```

正式生成时移除 `--max-count 2`，并把 `--out-dir` 设为训练要读取的位置：

```bash
.venv/bin/python src/dino-sp-generator/generate_sp.py \
  --data-path /home/ubuntu/projects/datasets/voc2012 \
  --img-list /home/ubuntu/projects/datasets/voc2012/voc12 \
  --model-path /home/ubuntu/projects/weights/dino22416b.npy \
  --out-dir /home/ubuntu/projects/outputs/drcformer/sp_idx/dino \
  --tau 0.3 --device cuda --batch-size 8 --num-workers 4 \
  --crf-only --no-png
```

脚本会把供训练读取的区域标签写入 `--out-dir/npy/{image_id}.npy`；`--crf-only --no-png` 只保留该索引数组，避免额外生成 PNG 可视化图。
