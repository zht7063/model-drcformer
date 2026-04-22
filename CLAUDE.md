这是一个叫做 DRCFormer 的深度学习项目。

项目相关环境配置包括：

- 项目运行环境

conda 环境，通过 `conda avtivate py310` 激活环境；

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
