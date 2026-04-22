#!/usr/bin/env python3
"""
从 COCO 标注 JSON 文件生成分割掩码（PNG格式）

用法:
    python convert_coco_to_segmentation_masks.py \
        --annotation_file /path/to/annotations/instances_val2014.json \
        --output_dir /path/to/output/segmentation_masks \
        --img_list /path/to/val_id.txt  # 可选，如果提供则只处理列表中的图像
"""

import os
import json
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as coco_mask


# COCO 80 个前景类别（不包括背景）
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_coco_annotations(annotation_file):
    """加载 COCO 标注文件"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data


def build_image_annotations_map(annotations):
    """构建图像ID到标注列表的映射"""
    image_annotations = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    return image_annotations


def build_image_info_map(annotations):
    """构建图像ID到图像信息的映射"""
    image_info_map = {}
    for img in annotations['images']:
        image_info_map[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    return image_info_map


def build_category_id_to_index_map(annotations):
    """构建类别ID到索引的映射（COCO类别ID从1开始，映射到1-80，0为背景）"""
    # COCO类别ID不连续（1-90，但只有80个类别），需要映射到连续的1-80
    # 按类别ID排序，然后映射到1-80
    sorted_categories = sorted(annotations['categories'], key=lambda x: x['id'])
    category_id_to_idx = {}
    for idx, cat in enumerate(sorted_categories, start=1):
        # 将COCO的类别ID映射到连续的1-80索引
        category_id_to_idx[cat['id']] = idx
    return category_id_to_idx


def convert_to_segmentation_mask(image_annotations, image_info, category_id_to_idx):
    """
    将COCO标注转换为分割掩码
    
    Args:
        image_annotations: 该图像的所有标注
        image_info: 图像信息（宽度、高度）
        category_id_to_idx: 类别ID到索引的映射
    
    Returns:
        segmentation_mask: 分割掩码数组，值为类别索引（0=背景，1-80=前景类别）
    """
    height = image_info['height']
    width = image_info['width']
    
    # 初始化掩码（0表示背景）
    segmentation_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 按面积从大到小排序，确保大物体覆盖小物体
    sorted_annotations = sorted(image_annotations, key=lambda x: x.get('area', 0), reverse=True)
    
    for ann in sorted_annotations:
        category_id = ann['category_id']
        category_idx = category_id_to_idx.get(category_id)
        
        if category_idx is None:
            continue
        
        # 解码分割掩码
        if isinstance(ann['segmentation'], list):
            # RLE格式或polygon格式
            if isinstance(ann['segmentation'][0], list):
                # Polygon格式：转换为RLE
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                rle = coco_mask.merge(rle)
            else:
                # RLE格式
                rle = ann['segmentation']
            
            # 解码RLE为二进制掩码
            mask = coco_mask.decode(rle)
            
            # 如果mask是3D的（多个通道），取第一个通道
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # 将掩码区域设置为对应的类别索引
            segmentation_mask[mask > 0] = category_idx
    
    return segmentation_mask


def convert_coco_to_masks(annotation_file, output_dir, img_list_file=None):
    """
    从COCO标注文件生成分割掩码
    
    Args:
        annotation_file: COCO 标注 JSON 文件路径
        output_dir: 输出目录
        img_list_file: 图像列表文件（可选），如果提供则只处理列表中的图像
    """
    print(f"Loading COCO annotations from: {annotation_file}")
    annotations = load_coco_annotations(annotation_file)
    
    print("Building annotation mappings...")
    image_annotations_map = build_image_annotations_map(annotations)
    image_info_map = build_image_info_map(annotations)
    category_id_to_idx = build_category_id_to_index_map(annotations)
    
    # 如果提供了图像列表，只处理列表中的图像
    if img_list_file is not None and os.path.exists(img_list_file):
        print(f"Loading image list from: {img_list_file}")
        with open(img_list_file, 'r') as f:
            img_name_list = [line.strip() for line in f.readlines()]
    else:
        # 使用所有图像
        img_name_list = [os.path.splitext(img['file_name'])[0] for img in annotations['images']]
        print(f"Processing all {len(img_name_list)} images from annotations")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建图像名称到图像ID的映射
    name_to_id = {}
    for img_id, img_info in image_info_map.items():
        name = os.path.splitext(img_info['file_name'])[0]
        name_to_id[name] = img_id
    
    # 处理每张图像
    processed_count = 0
    skipped_count = 0
    
    for img_name in tqdm(img_name_list, desc="Converting masks"):
        # 查找对应的图像ID
        image_id = name_to_id.get(img_name)
        
        if image_id is None:
            print(f"Warning: Image {img_name} not found in annotations, skipping...")
            skipped_count += 1
            continue
        
        # 获取图像信息和标注
        image_info = image_info_map[image_id]
        image_annotations = image_annotations_map.get(image_id, [])
        
        if len(image_annotations) == 0:
            # 如果没有标注，创建全背景掩码
            segmentation_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        else:
            # 转换为分割掩码
            segmentation_mask = convert_to_segmentation_mask(
                image_annotations, image_info, category_id_to_idx
            )
        
        # 保存为PNG文件
        output_file = os.path.join(output_dir, f'{img_name}.png')
        mask_image = Image.fromarray(segmentation_mask, mode='L')
        mask_image.save(output_file)
        
        processed_count += 1
    
    print(f"\n✓ Successfully converted {processed_count} masks")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} images (not found in annotations)")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to segmentation masks (PNG format)"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to COCO annotation JSON file (e.g., instances_val2014.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for segmentation masks"
    )
    parser.add_argument(
        "--img_list",
        type=str,
        default=None,
        help="Optional: Path to image list file (e.g., val_id.txt). If provided, only process images in the list"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    try:
        import pycocotools
    except ImportError:
        print("Error: pycocotools is required. Please install it with: pip install pycocotools")
        return
    
    convert_coco_to_masks(args.annotation_file, args.output_dir, args.img_list)


if __name__ == "__main__":
    main()
