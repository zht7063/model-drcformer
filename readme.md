## 运行前注意事项（py310）

### 训练入口与工作目录

- `src/drcformer/main.py` 使用了 `from datasets import ...`、`from engine import ...` 这类相对导入，**需要在 `src/drcformer/` 目录下运行**，否则会出现导入失败或导入到错误模块的情况。

示例：

```bash
conda activate py310
cd /home/ubuntu/projects/drcformer/src/drcformer
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
