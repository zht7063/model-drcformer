这是一个叫做 DRCFormer 的深度学习项目。

项目相关环境配置包括：

- 项目运行环境

conda 环境，默认使用 `py310`，通过 `conda activate py310` 激活；

- 显卡使用约定

当前机器默认使用单卡 `NVIDIA RTX 4090`（逻辑设备 `cuda:0`）。
训练时固定使用该卡，建议使用如下方式：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/bin/conda run -n py310 \
python run_drcformer.py --all --cuda 0
```

- VOC 2012 数据集路径

数据集路径为：`~/projects/voc2012`，其下包括需要的子目录：

```
(base) zhangpeng@ubuntu:/data/user/tyb/datasets$ find -maxdepth 2
.
./voc2012
./voc2012/SegmentationObject
./voc2012/JPEGImages
./voc2012/Annotations
./voc2012/voc12
./voc2012/SegmentationClassAug
./voc2012/SegmentationClass
./voc2012/ImageSets
./coco2014
./coco2014/val
./coco2014/train
./train2014.zip
./val2014.zip
```

- 所有预训练权重路径都保存在 `~/projects/pre-weights` 目录下

- 所有输出内容保存目录为 `~/projects/outputs/drcformer` （包括训练和推理结果）

> 注意：为项目构建代码修改计划的时候，使用简体中文编写计划内容。

---

## 最近一次启动训练的参数踩坑记录（2026-05-06）

以下问题都发生在启动 `wo/SNS` 实验时，后续可直接规避：

1. `dataset_path` 默认值与实际数据目录不一致  
   - 现象：报错 `训练集索引不存在: /home/ubuntu/projects/datasets/voc12/train_aug_id.txt`  
   - 原因：`run_drcformer.py` 默认 `--dataset_path` 指向 `~/projects/datasets`，但当前可用数据在 `~/projects/datasets/voc2012aug`。  
   - 处理：启动时显式传参  
     `--dataset_path /home/ubuntu/projects/datasets/voc2012aug`

2. 参数名写法错误：`--dataset-path`（错误）  
   - 现象：报错 `unrecognized arguments: --dataset-path ...`  
   - 原因：该脚本参数名使用下划线，不是中划线。  
   - 正确写法：`--dataset_path /home/ubuntu/projects/datasets/voc2012aug`

3. Shell 变量名冲突（zsh）  
   - 现象：命令尾部出现 `zsh: read-only variable: status`  
   - 原因：`status` 在 zsh 中是只读特殊变量。  
   - 处理：改用其他变量名，如 `exit_code`。
