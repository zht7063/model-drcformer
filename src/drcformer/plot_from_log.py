#!/usr/bin/env python3
"""
从日志文件重新绘制训练指标统计图

使用方法:
    python plot_from_log.py --log-file outputs/drcformer/exp-xxx/metrics_history.jsonl
    python plot_from_log.py --log-file metrics_history.jsonl --output custom_plot.png
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def load_metrics_from_log(log_file):
    """从 JSON Lines 日志文件加载指标数据
    
    Args:
        log_file: JSON Lines 格式的日志文件路径
        
    Returns:
        epochs: epoch 列表
        entropy_pos: 正类熵均值列表
        pmax_neg: 负类最大响应列表
        map_values: mAP 值列表（如果有）
    """
    epochs = []
    entropy_pos = []
    pmax_neg = []
    map_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                entry = json.loads(line)
                epochs.append(entry['epoch'])
                entropy_pos.append(entry.get('entropy_pos_mean', 0.0))
                pmax_neg.append(entry.get('pmax_neg_mean', 0.0))
                if 'mAP' in entry:
                    map_values.append(entry['mAP'])
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
                continue
    
    return epochs, entropy_pos, pmax_neg, map_values


def plot_metrics(epochs, entropy_pos, pmax_neg, output_path, show_map=False, map_values=None):
    """绘制指标统计图
    
    Args:
        epochs: epoch 列表
        entropy_pos: 正类熵均值列表
        pmax_neg: 负类最大响应列表
        output_path: 输出图像路径
        show_map: 是否在单独的子图中显示 mAP
        map_values: mAP 值列表
    """
    if show_map and map_values and len(map_values) == len(epochs):
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 第一个子图：熵和负类响应
        ax1.plot(epochs, entropy_pos, 'b-o', label=r'$\bar{H}$ (Positive Avg Entropy)', 
                linewidth=2, markersize=6)
        ax1.plot(epochs, pmax_neg, 'r-s', label=r'$\bar{P}_{max}^{-}$ (Negative Max Response)', 
                linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Validation Metrics: Positive Entropy Mean and Negative Max Response', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 第二个子图：mAP
        ax2.plot(epochs, map_values, 'g-^', label='mAP', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('mAP', fontsize=12)
        ax2.set_title('Mean Average Precision (mAP)', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    else:
        # 单个图：只显示熵和负类响应
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, entropy_pos, 'b-o', label=r'$\bar{H}$ (Positive Avg Entropy)', 
                linewidth=2, markersize=6)
        plt.plot(epochs, pmax_neg, 'r-s', label=r'$\bar{P}_{max}^{-}$ (Negative Max Response)', 
                linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Validation Metrics: Positive Entropy Mean and Negative Max Response', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"统计图已保存到: {output_path}")


def print_summary(epochs, entropy_pos, pmax_neg, map_values):
    """打印统计摘要"""
    print("\n" + "="*60)
    print("训练指标摘要")
    print("="*60)
    print(f"总 Epoch 数: {len(epochs)}")
    if epochs:
        print(f"Epoch 范围: {min(epochs)} - {max(epochs)}")
    
    if entropy_pos:
        print(f"\n正类熵均值:")
        print(f"  最小值: {min(entropy_pos):.4f} (Epoch {epochs[entropy_pos.index(min(entropy_pos))]})")
        print(f"  最大值: {max(entropy_pos):.4f} (Epoch {epochs[entropy_pos.index(max(entropy_pos))]})")
        print(f"  最终值: {entropy_pos[-1]:.4f}")
    
    if pmax_neg:
        print(f"\n负类最大响应:")
        print(f"  最小值: {min(pmax_neg):.4f} (Epoch {epochs[pmax_neg.index(min(pmax_neg))]})")
        print(f"  最大值: {max(pmax_neg):.4f} (Epoch {epochs[pmax_neg.index(max(pmax_neg))]})")
        print(f"  最终值: {pmax_neg[-1]:.4f}")
    
    if map_values:
        print(f"\nmAP:")
        print(f"  最小值: {min(map_values):.4f} (Epoch {epochs[map_values.index(min(map_values))]})")
        print(f"  最大值: {max(map_values):.4f} (Epoch {epochs[map_values.index(max(map_values))]})")
        print(f"  最终值: {map_values[-1]:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='从日志文件重新绘制训练指标统计图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python plot_from_log.py --log-file outputs/drcformer/exp-1230-b/metrics_history.jsonl
  
  # 指定输出路径
  python plot_from_log.py --log-file metrics_history.jsonl --output custom_plot.png
  
  # 显示 mAP 子图
  python plot_from_log.py --log-file metrics_history.jsonl --show-map
        """
    )
    parser.add_argument(
        '--log-file', 
        type=str, 
        required=True, 
        help='指标日志文件路径 (metrics_history.jsonl)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None, 
        help='输出图像路径（默认保存到日志文件同目录，文件名：val_metrics_plot_replay.png）'
    )
    parser.add_argument(
        '--show-map',
        action='store_true',
        help='在单独的子图中显示 mAP 曲线'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='不打印统计摘要'
    )
    
    args = parser.parse_args()
    
    # 检查日志文件是否存在
    log_file_path = Path(args.log_file)
    if not log_file_path.exists():
        print(f"错误: 日志文件不存在: {args.log_file}")
        exit(1)
    
    # 加载数据
    print(f"正在从日志文件加载数据: {args.log_file}")
    epochs, entropy_pos, pmax_neg, map_values = load_metrics_from_log(args.log_file)
    
    if not epochs:
        print("错误: 日志文件中没有找到有效数据")
        exit(1)
    
    print(f"成功加载 {len(epochs)} 个 epoch 的数据")
    
    # 打印摘要
    if not args.no_summary:
        print_summary(epochs, entropy_pos, pmax_neg, map_values)
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = str(log_file_path.parent / 'val_metrics_plot_replay.png')
    
    # 绘制并保存图表
    plot_metrics(epochs, entropy_pos, pmax_neg, output_path, 
                show_map=args.show_map, map_values=map_values)
    
    print(f"\n完成！")

