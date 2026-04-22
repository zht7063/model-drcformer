#! /bin/bash

# DeepLabV3Plus ResNet101 训练脚本
# 用于在 VOC2012 Aug 数据集上训练语义分割模型

# 屏蔽 cuda:0，只使用 cuda:1
# export CUDA_VISIBLE_DEVICES=1

# 配置项目基础路径
BASE="/home/zhangpeng/tyb"

# 配置数据集相关路径
dataset_dir="${BASE}"
voc_data_root="${BASE}/voc2012"
train_image_dir="${voc_data_root}/JPEGImages"
val_image_dir="${voc_data_root}/JPEGImages"
train_mask_dir="${voc_data_root}/SegmentationClassAug"
val_mask_dir="${voc_data_root}/SegmentationClass"

# 配置 ID 列表路径
id_list_dir="${BASE}/voc2012/voc12"
train_id_list="${id_list_dir}/train_aug_id.txt"
val_id_list="${id_list_dir}/val_id.txt"

# 配置类别标签路径
cls_labels_path="${id_list_dir}/cls_labels.npy"

# 配置预训练权重路径
pretrained_weight_dir="${BASE}/pre-weights/best_deeplabv3plus_resnet101_voc_os16.pth"

# 配置模型输出路径
checkpoints_dir="${BASE}/outputs/seg/checkpoints"

# 检查数据集路径是否存在
if [ ! -d "${voc_data_root}" ]; then
    echo "错误: VOC2012 数据集路径不存在: ${voc_data_root}"
    echo "请确保 VOC2012 数据集已正确解压到指定目录"
    exit 1
fi

# 检查训练集 ID 列表文件是否存在
if [ ! -f "${train_id_list}" ]; then
    echo "错误: 训练集 ID 列表文件不存在: ${train_id_list}"
    echo "请确保训练集 ID 列表文件存在于指定路径"
    exit 1
fi

# 检查验证集 ID 列表文件是否存在
if [ ! -f "${val_id_list}" ]; then
    echo "错误: 验证集 ID 列表文件不存在: ${val_id_list}"
    echo "请确保验证集 ID 列表文件存在于指定路径"
    exit 1
fi

# 检查类别标签文件是否存在
if [ ! -f "${cls_labels_path}" ]; then
    echo "错误: 类别标签文件不存在: ${cls_labels_path}"
    echo "请确保类别标签文件存在于指定路径"
    exit 1
fi

# 检查预训练权重文件是否存在
if [ ! -f "${pretrained_weight_dir}" ]; then
    echo "错误: 预训练权重文件不存在: ${pretrained_weight_dir}"
    echo "请确保预训练权重文件存在于指定路径"
    exit 1
fi

# 创建checkpoints输出目录
mkdir -p "${checkpoints_dir}"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建checkpoints目录: ${checkpoints_dir}"
    exit 1
fi

echo "开始训练 DeepLabV3Plus ResNet101 模型..."
echo "数据集基础路径: ${dataset_dir}"
echo "VOC2012 数据路径: ${voc_data_root}"
echo "训练集图像路径: ${train_image_dir}"
echo "验证集图像路径: ${val_image_dir}"
echo "训练集标签路径: ${train_mask_dir}"
echo "验证集标签路径: ${val_mask_dir}"
echo "训练集 ID 列表: ${train_id_list}"
echo "验证集 ID 列表: ${val_id_list}"
echo "类别标签文件: ${cls_labels_path}"
echo "预训练权重: ${pretrained_weight_dir}"
echo "模型输出目录: ${checkpoints_dir}"
echo "使用 GPU: ${CUDA_VISIBLE_DEVICES}"

# 训练脚本
python main.py \
    --model deeplabv3plus_resnet101 \
    --dataset voc \
    --year 2012_aug \
    --data_root ${dataset_dir} \
    --voc_data_root ${voc_data_root} \
    --train_image_dir ${train_image_dir} \
    --val_image_dir ${val_image_dir} \
    --train_mask_dir ${train_mask_dir} \
    --val_mask_dir ${val_mask_dir} \
    --train_id_list ${train_id_list} \
    --val_id_list ${val_id_list} \
    --cls_labels_path ${cls_labels_path} \
    --ckpt ${pretrained_weight_dir} \
    --checkpoints_dir ${checkpoints_dir} \
    --crop_val \
    --lr 0.01 \
    --crop_size 513 \
    --batch_size 16 \
    --output_stride 16 \
    --total_itrs 30000 \
    --gpu_id 1

