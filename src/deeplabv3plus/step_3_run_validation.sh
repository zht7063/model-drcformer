#!/bin/bash

# DeepLabV3Plus 模型验证脚本
# 用于对训练完成的模型进行全面性能评估

# # 基本验证（使用默认最佳模型）
# ./run_validation.sh
# # 指定特定模型权重进行验证
# ./run_validation.sh --model_checkpoint checkpoints/latest_deeplabv3plus_resnet101_voc_os16.pth
# # 验证并保存结果图像
# ./run_validation.sh --save_results
# # 查看帮助信息
# ./run_validation.sh --help

# 屏蔽 cuda:0，只使用 cuda:1
export CUDA_VISIBLE_DEVICES=1

# 配置项目基础路径
BASE="/data/user/tyb"

# 配置数据集相关路径
dataset_dir="${BASE}/datasets"
voc_data_root="${BASE}/datasets/voc2012"
val_image_dir="${voc_data_root}/JPEGImages"
val_mask_dir="${voc_data_root}/SegmentationClass"

# 配置 ID 列表路径
id_list_dir="${BASE}/datasets/voc2012/voc12"
val_id_list="${id_list_dir}/val_id.txt"

# 配置类别标签路径
cls_labels_path="${id_list_dir}/cls_labels.npy"

# 配置模型权重路径（默认使用最佳模型）
# model_checkpoint="${BASE}/deeplabv3plus/checkpoints/best_deeplabv3plus_resnet101_voc_os16.pth"
model_checkpoint="${BASE}/outputs/seg/checkpoints_pseudolabel/best_deeplabv3plus_resnet101_voc_os16.pth"

# 配置结果输出路径
results_dir="${BASE}/outputs/seg/validation_results"
timestamp=$(date +"%Y%m%d_%H%M%S")
current_results_dir="${results_dir}/${timestamp}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_checkpoint)
            model_checkpoint="$2"
            shift 2
            ;;
        --results_dir)
            current_results_dir="$2"
            shift 2
            ;;
        --save_results)
            save_results=true
            shift
            ;;
        --help)
            echo "DeepLabV3Plus 模型验证脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model_checkpoint PATH    指定模型权重文件路径"
            echo "  --results_dir PATH         指定结果输出目录"
            echo "  --save_results            保存验证结果图像"
            echo "  --help                    显示此帮助信息"
            echo ""
            echo "默认配置:"
            echo "  模型权重: ${model_checkpoint}"
            echo "  结果目录: ${current_results_dir}"
            echo "  验证数据: ${voc_data_root}"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必要的路径和文件
echo "=== DeepLabV3Plus 模型验证 ==="
echo "验证时间: $(date)"
echo ""

# 检查模型权重文件
if [ ! -f "${model_checkpoint}" ]; then
    echo "错误: 模型权重文件不存在: ${model_checkpoint}"
    echo "请确保模型已训练完成或指定正确的权重文件路径"
    exit 1
fi

# 检查验证数据路径
if [ ! -d "${voc_data_root}" ]; then
    echo "错误: VOC2012 验证数据路径不存在: ${voc_data_root}"
    exit 1
fi

# 检查验证集 ID 列表文件
if [ ! -f "${val_id_list}" ]; then
    echo "错误: 验证集 ID 列表文件不存在: ${val_id_list}"
    exit 1
fi

# 检查类别标签文件
if [ ! -f "${cls_labels_path}" ]; then
    echo "错误: 类别标签文件不存在: ${cls_labels_path}"
    exit 1
fi

# 创建结果输出目录
mkdir -p "${current_results_dir}"

echo "验证配置:"
echo "  模型权重: ${model_checkpoint}"
echo "  验证数据: ${voc_data_root}"
echo "  验证图像: ${val_image_dir}"
echo "  验证标签: ${val_mask_dir}"
echo "  验证列表: ${val_id_list}"
echo "  类别标签: ${cls_labels_path}"
echo "  结果目录: ${current_results_dir}"
echo "  使用 GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

# 构建验证命令参数
validation_args=(
    --model deeplabv3plus_resnet101
    --dataset voc
    --year 2012_aug
    --data_root "${dataset_dir}"
    --voc_data_root "${voc_data_root}"
    --train_image_dir "${voc_data_root}/JPEGImages"
    --train_mask_dir "${voc_data_root}/SegmentationClassAug"
    --train_id_list "${id_list_dir}/train_aug_id.txt"
    --val_image_dir "${val_image_dir}"
    --val_mask_dir "${val_mask_dir}"
    --val_id_list "${val_id_list}"
    --cls_labels_path "${cls_labels_path}"
    --ckpt "${model_checkpoint}"
    --test_only
    --crop_val
    --crop_size 513
    --batch_size 16
    --output_stride 16
    --gpu_id 0
)

# 如果指定保存结果，添加相应参数
if [ "${save_results}" = true ]; then
    validation_args+=(--save_val_results)
    echo "将保存验证结果图像到 results/ 目录"
fi

echo "开始模型验证..."
echo "命令: python main.py ${validation_args[*]}"
echo ""

# 执行验证
python main.py "${validation_args[@]}" 2>&1 | tee "${current_results_dir}/validation_log.txt"

# 检查验证是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=== 验证完成 ==="
    echo "验证日志已保存到: ${current_results_dir}/validation_log.txt"
    
    # 如果保存了结果图像，移动到结果目录
    if [ "${save_results}" = true ] && [ -d "results" ]; then
        mv results "${current_results_dir}/validation_images"
        echo "验证结果图像已保存到: ${current_results_dir}/validation_images/"
    fi
    
    echo "验证结果目录: ${current_results_dir}"
    echo ""
    echo "可以查看以下文件:"
    echo "  - 验证日志: ${current_results_dir}/validation_log.txt"
    if [ "${save_results}" = true ]; then
        echo "  - 结果图像: ${current_results_dir}/validation_images/"
    fi
else
    echo ""
    echo "=== 验证失败 ==="
    echo "请检查验证日志: ${current_results_dir}/validation_log.txt"
    exit 1
fi