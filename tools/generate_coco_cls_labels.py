#!/usr/bin/env python3
"""
从 COCO 标注 JSON 文件生成类别标签文件 (cls_labels.npy)

用法:
    python generate_coco_cls_labels.py \
        --annotation_file /path/to/annotations/instances_train2014.json \
        --output_file /path/to/output/cls_labels.npy \
        --img_list /path/to/train_id.txt  # 可选，如果提供则只处理列表中的图像
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


# COCO 80 个类别（不包括背景）
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


def build_image_category_map(annotations):
    """构建图像ID到类别列表的映射"""
    # 创建类别ID到索引的映射（COCO类别ID从1开始，我们需要映射到0-79）
    category_id_to_idx = {}
    for cat in annotations['categories']:
        # COCO类别ID从1开始，映射到0-79
        category_id_to_idx[cat['id']] = cat['id'] - 1
    
    # 构建图像ID到类别集合的映射
    image_categories = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        category_idx = category_id_to_idx.get(category_id)
        
        if category_idx is not None:
            if image_id not in image_categories:
                image_categories[image_id] = set()
            image_categories[image_id].add(category_idx)
    
    return image_categories


def build_image_name_map(annotations):
    """构建图像ID到图像名称的映射"""
    image_name_map = {}
    for img in annotations['images']:
        image_name_map[img['id']] = os.path.splitext(img['file_name'])[0]  # 去掉扩展名
    return image_name_map


def generate_cls_labels(annotation_file, output_file, img_list_file=None):
    """
    生成类别标签文件
    
    Args:
        annotation_file: COCO 标注 JSON 文件路径
        output_file: 输出 cls_labels.npy 文件路径
        img_list_file: 图像列表文件（可选），如果提供则只处理列表中的图像
    """
    print(f"Loading COCO annotations from: {annotation_file}")
    annotations = load_coco_annotations(annotation_file)
    
    print("Building image-category mapping...")
    image_categories = build_image_category_map(annotations)
    image_name_map = build_image_name_map(annotations)
    
    # 如果提供了图像列表，只处理列表中的图像
    if img_list_file is not None and os.path.exists(img_list_file):
        print(f"Loading image list from: {img_list_file}")
        with open(img_list_file, 'r') as f:
            img_name_list = [line.strip() for line in f.readlines()]
    else:
        # 使用所有图像
        img_name_list = list(image_name_map.values())
        print(f"Processing all {len(img_name_list)} images from annotations")
    
    # 构建类别标签字典
    cls_labels_dict = {}
    num_classes = 80
    
    for img_name in tqdm(img_name_list, desc="Generating labels"):
        # 查找对应的图像ID
        image_id = None
        for img_id, name in image_name_map.items():
            if name == img_name:
                image_id = img_id
                break
        
        if image_id is None:
            # 如果找不到对应的图像ID，使用零标签
            cls_labels_dict[img_name] = np.zeros(num_classes, dtype=np.float32)
            continue
        
        # 创建类别标签向量（80维，每个类别对应一个位置）
        label_vector = np.zeros(num_classes, dtype=np.float32)
        if image_id in image_categories:
            for cat_idx in image_categories[image_id]:
                if 0 <= cat_idx < num_classes:
                    label_vector[cat_idx] = 1.0
        
        # 使用图像名称（不含扩展名）作为键
        cls_labels_dict[img_name] = label_vector
    
    # 保存为 npy 文件
    print(f"Saving labels to: {output_file}")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    np.save(output_file, cls_labels_dict)
    print(f"✓ Successfully generated {len(cls_labels_dict)} image labels")
    print(f"  Output file: {output_file}")
    
    # 统计信息
    num_images_with_labels = sum(1 for label in cls_labels_dict.values() if label.sum() > 0)
    print(f"  Images with labels: {num_images_with_labels} / {len(cls_labels_dict)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO class labels file (cls_labels.npy) from COCO annotations"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to COCO annotation JSON file (e.g., instances_train2014.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output cls_labels.npy file"
    )
    parser.add_argument(
        "--img_list",
        type=str,
        default=None,
        help="Optional: Path to image list file (e.g., train_id.txt). If provided, only process images in the list"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    generate_cls_labels(args.annotation_file, args.output_file, args.img_list)


if __name__ == "__main__":
    main()
