# 项目目标

当前项目是一个叫做 DRCFormer 的弱监督语义分割方法实验代码仓库，包含有执行一个多阶段的若监督语义分割方法的完整流程。

在 src/ 目录中，每个子目录的主要内容与职责如下：

1. dino-sp-generator：基于深度模型（DINO）设计的像素区域划分器，用于将图像划分为多个像素区域；

2. drcformer：模型主程序，基于 MCTFormerPlus，设计了多个模块，包括：

    采用了超像素 patch 替换原有 patch；

    设计了熵压缩和边界抑制模块，用于降低模型输出的不确定性和标签噪声；
    
    设计了区域性标签校正模块，通过令同一个深度特征划分的像素区域中的超像素 patch 进行群体投票和均值矫正，用于实现区域性标签校正；

3. deeplabv3plus：基于 DeepLabV3Plus 设计的语义分割模型，用于实现语义分割；


# 项目参数配置

- 项目 conda 环境：`py310`，直接通过终端命令`conda activate py310`激活，不需要写入到项目程序中。

- 项目根目录：`/home/zhangpeng/tyb`，其余路径均采用相对路径表示：

    - 数据集路径：`voc2012/`，包含 VOC2012 数据集；
    - 数据图像索引列表（txt格式）：`voc2012/voc12/`，以 txt 格式保存了不同子集（如训练集、验证集等）的图像索引列表；
    - 预训练权重路径：`pre-weights`，所有训练过程要用到的模型权重都保存在该路径下；
    - 输出路径：`outputs`，所有训练过程要用到的输出都保存在该路径下；

# 项目运行流程

1. 由预训练好的 dino-sp-generator 生成超像素区域索引文件，保存在 `outputs/dino_sp/npy/` 路径下；

2. 由 drcformer 模型主程序，结合超像素区域索引文件，生成多尺度 CAM 图，保存在 `outputs/drcformer(或者 spit)/<实验id>/` 路径下；

3. 通过 drcformer 模型主程序，结合多尺度 CAM 图，生成伪标签，保存到 `outputs/psa/attn-patchrefine-npy-crf_xxx/` 路径下，训练 psa 模型，并用训练得到的最佳权重推理产生伪标签，保存在`outputs/psa/pgt-psa-rw`中；

4. 利用 psa 推理得到的伪标签，训练 deeplabv3plus 模型，得到最终的语义分割结果，保存在 `outputs/deeplabv3plus/<实验id>/` 路径下；
