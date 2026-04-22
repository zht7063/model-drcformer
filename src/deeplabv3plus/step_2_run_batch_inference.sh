#!/bin/bash

# DeepLabV3Plus 批量推理脚本
# 用于对训练集或验证集进行批量推理

# =============================================================================
# 可调整参数配置区域
# =============================================================================

# 推理目标设置 (可选: train, val)
INFERENCE_TARGET="val"  # 默认对验证集进行推理，可修改为 "train" 对训练集推理

# GPU 设置
GPU_ID="3"  # 使用的 GPU ID

# 模型配置
MODEL_NAME="deeplabv3plus_resnet101"  # 模型名称
OUTPUT_STRIDE=16  # 输出步长
CROP_SIZE=513  # 裁剪尺寸
BATCH_SIZE=4  # 批处理大小

# 是否对验证图像进行裁剪
# 注意：在弱监督语义分割流程中，DeepLabV3Plus作为最后阶段应该对完整图像进行推理
# 以生成与原图尺寸一致的完整伪标签，因此禁用裁剪功能
CROP_VAL=false  # 设置为 true 启用裁剪，false 禁用（推荐false以获得完整图像的伪标签）

# =============================================================================
# 路径配置区域 (仿照训练脚本结构)
# =============================================================================

# 配置项目基础路径
BASE="/data/user/tyb"

# 配置数据集相关路径
dataset_dir="${BASE}/datasets"
voc_data_root="${BASE}/datasets/voc2012"
image_dir="${voc_data_root}/JPEGImages"

# 配置 ID 列表路径
id_list_dir="${voc_data_root}/voc12"
train_id_list="${id_list_dir}/train_aug_id.txt"
val_id_list="${id_list_dir}/val_id.txt"

# 配置预训练权重路径（使用预训练权重）
model_checkpoint="${BASE}/pre-weights/best_deeplabv3plus_resnet101_voc_os16.pth"

# 配置输出路径
output_base_dir="${BASE}/outputs/drcformer/seg"
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${output_base_dir}/${INFERENCE_TARGET}_${timestamp}"

# =============================================================================
# 路径和文件检查
# =============================================================================

echo "=== DeepLabV3Plus 批量推理脚本 ==="
echo "推理目标: ${INFERENCE_TARGET}"
echo "时间戳: ${timestamp}"
echo ""

# 检查数据集路径是否存在
if [ ! -d "${voc_data_root}" ]; then
    echo "错误: VOC2012 数据集路径不存在: ${voc_data_root}"
    echo "请确保 VOC2012 数据集已正确解压到指定目录"
    exit 1
fi

# 检查图像目录是否存在
if [ ! -d "${image_dir}" ]; then
    echo "错误: 图像目录不存在: ${image_dir}"
    exit 1
fi

# 根据推理目标选择对应的 ID 列表文件
if [ "${INFERENCE_TARGET}" = "train" ]; then
    target_id_list="${train_id_list}"
    echo "使用训练集 ID 列表: ${target_id_list}"
elif [ "${INFERENCE_TARGET}" = "val" ]; then
    target_id_list="${val_id_list}"
    echo "使用验证集 ID 列表: ${target_id_list}"
else
    echo "错误: 无效的推理目标 '${INFERENCE_TARGET}'"
    echo "请设置 INFERENCE_TARGET 为 'train' 或 'val'"
    exit 1
fi

# 检查 ID 列表文件是否存在
if [ ! -f "${target_id_list}" ]; then
    echo "错误: ID 列表文件不存在: ${target_id_list}"
    echo "请确保 ID 列表文件存在于指定路径"
    exit 1
fi

# 检查模型权重文件是否存在
if [ ! -f "${model_checkpoint}" ]; then
    echo "错误: 模型权重文件不存在: ${model_checkpoint}"
    echo "请确保模型权重文件存在于指定路径"
    exit 1
fi

# 创建输出目录
mkdir -p "${output_dir}"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建输出目录: ${output_dir}"
    exit 1
fi

# =============================================================================
# 配置信息输出
# =============================================================================

echo "=== 配置信息 ==="
echo "项目基础路径: ${BASE}"
echo "数据集基础路径: ${dataset_dir}"
echo "VOC2012 数据路径: ${voc_data_root}"
echo "图像目录: ${image_dir}"
echo "目标 ID 列表: ${target_id_list}"
echo "模型权重: ${model_checkpoint}"
echo "输出目录: ${output_dir}"
echo "使用 GPU: ${GPU_ID}"
echo "模型: ${MODEL_NAME}"
echo "输出步长: ${OUTPUT_STRIDE}"
echo "裁剪尺寸: ${CROP_SIZE}"
echo "批处理大小: ${BATCH_SIZE}"
echo "是否裁剪: ${CROP_VAL}"
echo ""

# 统计要推理的图像数量
image_count=$(wc -l < "${target_id_list}")
echo "待推理图像数量: ${image_count}"
echo ""

# =============================================================================
# 执行批量推理
# =============================================================================

echo "=== 开始批量推理 ==="

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 构建推理命令参数
inference_args=(
    --input "${image_dir}"
    --dataset voc
    --model "${MODEL_NAME}"
    --output_stride ${OUTPUT_STRIDE}
    --crop_size ${CROP_SIZE}
    --val_batch_size ${BATCH_SIZE}
    --ckpt "${model_checkpoint}"
    --save_val_results_to "${output_dir}"
    --gpu_id "${GPU_ID}"
)

# 如果启用裁剪，添加 --crop_val 参数
if [ "${CROP_VAL}" = true ]; then
    inference_args+=(--crop_val)
fi

# 切换到 deeplabv3plus 目录执行推理脚本
cd "${BASE}/drcformer/src/deeplabv3plus" || {
    echo "错误: 无法切换到 deeplabv3plus 目录"
    exit 1
}

# 执行推理命令
echo "执行命令: python predict.py ${inference_args[*]}"
echo ""

python predict.py "${inference_args[@]}"

# 检查推理是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 推理完成 ==="
    echo "推理结果已保存到: ${output_dir}"
    
    # 统计生成的结果文件数量
    result_count=$(find "${output_dir}" -name "*.png" | wc -l)
    echo "生成的结果文件数量: ${result_count}"
    
    # 创建推理日志
    log_file="${output_dir}/inference_log.txt"
    {
        echo "=== DeepLabV3Plus 批量推理日志 ==="
        echo "推理时间: $(date)"
        echo "推理目标: ${INFERENCE_TARGET}"
        echo "模型: ${MODEL_NAME}"
        echo "模型权重: ${model_checkpoint}"
        echo "输入图像目录: ${image_dir}"
        echo "ID 列表文件: ${target_id_list}"
        echo "输出目录: ${output_dir}"
        echo "GPU ID: ${GPU_ID}"
        echo "输出步长: ${OUTPUT_STRIDE}"
        echo "裁剪尺寸: ${CROP_SIZE}"
        echo "批处理大小: ${BATCH_SIZE}"
        echo "是否裁剪: ${CROP_VAL}"
        echo "待推理图像数量: ${image_count}"
        echo "生成结果文件数量: ${result_count}"
        echo "推理状态: 成功"
    } > "${log_file}"
    
    echo "推理日志已保存到: ${log_file}"
else
    echo ""
    echo "=== 推理失败 ==="
    echo "请检查错误信息并重试"
    exit 1
fi