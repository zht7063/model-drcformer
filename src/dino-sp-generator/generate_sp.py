"""
VOC12 超像素生成脚本

功能：
1. 使用 DINO 模型为 VOC12 数据集生成超像素分割
2. 支持 CRF 后处理
3. 输出 PNG 可视化文件和 NPY 索引数组

快速使用：
---------
方式1 - 直接调用 voc12_inference 函数（推荐）:
    conda activate py310
    python -c "from generate_sp import voc12_inference; voc12_inference()"

方式2 - 命令行参数:
    python generate_sp.py --data-path /path/to/VOC2012 \\
                          --img-list /path/to/voc12 \\
                          --model-path /path/to/dino.npy \\
                          --out-dir /path/to/output \\
                          --tau 0.3 --use-crf --crf-only

配置说明：
---------
voc12_inference() 函数默认配置：
- 数据集: /home/zhangpeng/tyb/voc2012
- 模型权重: /home/zhangpeng/tyb/pre-weights/dino22416b.npy
- 输出目录: /home/zhangpeng/tyb/dataset/dino_sp
- GPU: cuda:1 (通过 CUDA_VISIBLE_DEVICES=1)
- tau: 0.3
- CRF: 启用
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# 确保当前目录在 sys.path 最前面，优先导入本目录下的 datasets.py
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from datasets import VOC12Dataset, COCODataset
from dino_sp_generator import SuperPixelGenerator
# Optional progress bar
try:
    from tqdm import tqdm  # progress bar
except Exception:
    tqdm = None


def should_skip_image(img_name, npy_dir, png_dir, save_npy, save_png, resume):
    """
    检查图像是否应该被跳过（断点续传）
    
    Args:
        img_name: 图像名称（不含扩展名）
        npy_dir: NPY 输出目录
        png_dir: PNG 输出目录
        save_npy: 是否保存 NPY 文件
        save_png: 是否保存 PNG 文件
        resume: 是否启用断点续传
    
    Returns:
        bool: 如果应该跳过返回 True，否则返回 False
    """
    if not resume:
        return False
    
    # 检查所有需要的输出文件是否存在
    if save_npy and not (npy_dir / f"{img_name}.npy").exists():
        return False
    if save_png and not (png_dir / f"{img_name}.png").exists():
        return False
    
    # 所有需要的文件都存在，可以跳过
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        "Generate superpixel index maps (PNG + NPY) for VOC12/COCO datasets"
    )
    parser.add_argument("--dataset", type=str, default="voc2012", choices=["voc2012", "coco2014"],
                        help="Dataset type: voc2012 or coco2014 (default: voc2012)")
    parser.add_argument("--data-path", default="/home/zht/VOC2012", type=str,
                        help="Dataset root path (VOC2012: contains JPEGImages/, COCO2014: contains train2014/ and val2014/)")
    parser.add_argument("--img-list", default="voc12", type=str,
                        help="For VOC2012: Directory that contains train_aug_id.txt/val_id.txt. For COCO2014: ignored (scan directory)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="For COCO2014: dataset split (train or val, default: train). For VOC2012: ignored")
    parser.add_argument("--model-path", type=str,
                        default="assets/saved_models/sp_generator/dino22416b.npy",
                        help="Path to DINO numpy weights used by SuperPixelGenerator")
    parser.add_argument("--out-dir", type=str, default="assets/sp_idx",
                        help="Base output directory; will create subdirs 'png' and 'npy'")
    parser.add_argument("--tau", type=float, default=0.4,
                        help="Threshold tau used by SuperPixelGenerator")
    parser.add_argument("--use-crf", action="store_true", default=True,
                        help="Enable CRF post-processing inside SuperPixelGenerator (default: on)")
    parser.add_argument("--crf-only", action="store_true", default=False,
                        help="Only save CRF post-processed results, skip original outputs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run generation on: 'cuda' or 'cpu'")
    parser.add_argument("--max-count", type=int, default=None,
                        help="Process at most N images (for quick test runs)")
    # New toggles for outputs
    parser.add_argument("--no-png", dest="save_png", action="store_false",
                        help="Do not write PNG masks to disk")
    parser.add_argument("--no-npy", dest="save_npy", action="store_false",
                        help="Do not write NPY index arrays to disk")
    parser.set_defaults(save_png=True, save_npy=True)
    return parser.parse_args()


def run_generation(args):
    """核心生成逻辑，可被 main() 和 voc12_inference() 复用"""
    # Prepare output directories
    base_out = Path(args.out_dir)
    png_dir = base_out / "png"
    npy_dir = base_out / "npy"
    # Create only when needed, but harmless to create anyway
    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset based on dataset type
    dataset_type = getattr(args, 'dataset', 'voc2012')
    if dataset_type == 'voc2012':
        dataset = VOC12Dataset(
            img_name_list_path=args.img_list,
            voc12_root=args.data_path,
            train=True,
            gen_attn=False,
            transform=None,
        )
        logger.info(f"Loaded {len(dataset)} VOC2012 images from: {args.img_list}")
    elif dataset_type == 'coco2014':
        split = getattr(args, 'split', 'train')
        dataset = COCODataset(
            coco_root=args.data_path,
            split=split,
            transform=None,
        )
        logger.info(f"Loaded {len(dataset)} COCO2014 {split} images from: {args.data_path}")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Initialize generator
    generator = SuperPixelGenerator(model_path=args.model_path, device=args.device)

    total = len(dataset)
    n_ok, n_err = 0, 0
    n_skip = 0  # 跳过的图像数量
    
    # 获取 resume 参数（默认启用）
    resume = getattr(args, 'resume', True)
    
    if resume:
        logger.info("断点续传已启用：将自动跳过已完成的图像")

    # Build iterable with optional max-count and tqdm progress bar
    if args.max_count is not None:
        total_selected = min(total, args.max_count)
        indices = list(range(total_selected))
    else:
        total_selected = total
        indices = list(range(total))

    iterable = (
        tqdm(indices, total=total_selected, desc="Generating SP", unit="img")
        if tqdm is not None else indices
    )

    # Decide where to send PNG outputs generated by SuperPixelGenerator
    out_dir_for_gen = (str(png_dir) + "/") if args.save_png else None
    tmp_ctx = None
    if not args.save_png:
        try:
            import tempfile
            tmp_ctx = tempfile.TemporaryDirectory()
            out_dir_for_gen = tmp_ctx.name.rstrip("/") + "/"
        except Exception as e:
            logger.warning(f"Failed to create TemporaryDirectory for PNG suppression, fallback to {png_dir}: {e}")
            png_dir.mkdir(parents=True, exist_ok=True)
            out_dir_for_gen = str(png_dir) + "/"

    for idx, dataset_idx in enumerate(iterable, start=1):
        # Get image path from dataset
        sample = dataset[dataset_idx]
        img_file = sample['img_path']
        img_name = sample['img_name']
        
        if not os.path.exists(img_file):
            logger.warning(f"[{idx}/{total_selected}] Image not found: {img_file}")
            n_err += 1
            continue
        
        # 检查是否应该跳过（断点续传）
        if should_skip_image(img_name, npy_dir, png_dir, args.save_npy, args.save_png, resume):
            n_skip += 1
            if (tqdm is None) and (idx % 50 == 0 or idx == total_selected):
                logger.info(f"Progress: {idx}/{total_selected} | OK={n_ok}, ERR={n_err}, SKIP={n_skip}")
            continue

        try:
            # Generate SP; PNGs will go to out_dir_for_gen (tmp when --no-png)
            npimg, count = generator.generate_superpixel(
                img_file=img_file,
                tau=args.tau,
                out_dir=out_dir_for_gen,
                use_crf=args.use_crf,
                crf_only=args.crf_only,
                if_tip=False,
            )

            # Save raw index map as .npy only when enabled
            if args.save_npy:
                out_npy = npy_dir / f"{img_name}.npy"
                # npimg is a numpy array of integer labels
                # Delay import to avoid global dependency if not used elsewhere
                import numpy as np  # local import
                np.save(out_npy, np.asarray(npimg, dtype=np.int32))

            n_ok += 1
            if (tqdm is None) and (idx % 50 == 0 or idx == total_selected):
                logger.info(f"Progress: {idx}/{total_selected} | OK={n_ok}, ERR={n_err}, SKIP={n_skip}")
        except Exception as e:
            logger.error(f"[{idx}/{total_selected}] Failed on {img_name}: {e}")
            n_err += 1

    # Cleanup temp dir if we suppressed PNG outputs
    if tmp_ctx is not None:
        try:
            tmp_ctx.cleanup()
        except Exception:
            pass

    logger.info(f"Done. Total={total}, Saved={n_ok}, Failed={n_err}, Skipped={n_skip}. PNG -> {png_dir}, NPY -> {npy_dir}")


def voc12_inference():
    """
    VOC12 数据集推理函数，复刻 run_dino_sp_generator.sh 的逻辑
    屏蔽 cuda:0，只使用 cuda:1
    
    使用方式：
    ```python
    from generate_sp import voc12_inference
    voc12_inference()
    ```
    
    或者在命令行中：
    ```bash
    conda activate py310
    python -c "from generate_sp import voc12_inference; voc12_inference()"
    ```
    """
    # 设置 CUDA 设备（屏蔽 cuda:0，只使用 cuda:1）
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # 创建参数对象（使用 argparse.Namespace）
    class Args:
        def __init__(self):
            BASE = Path("/home/zhangpeng/tyb")
            
            # 输入参数（对应 shell 脚本中的路径）
            # 注意：所有路径转换为字符串以确保兼容性
            self.data_path = str(BASE / "voc2012")
            # img_list 可以是目录或文件路径，这里使用目录路径
            self.img_list = str(BASE / "voc2012" / "voc12")
            
            # 输出目录
            self.out_dir = str(BASE / "outputs" / "dino_sp")
            
            # 模型路径
            self.model_path = str(BASE / "pre-weights" / "dino22416b.npy")
            
            # 生成参数
            self.tau = 0.3  # 默认采用 0.3
            self.device = "cuda"
            self.use_crf = True
            self.crf_only = True
            
            # 输出选项
            self.save_png = True
            self.save_npy = True
            
            # 可选：限制处理数量（对应 shell 脚本中注释的 --max-count 100）
            self.max_count = None  # 设置为 None 表示处理所有图像
    
    args = Args()
    
    logger.info("=" * 60)
    logger.info("开始 VOC12 数据集超像素生成")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"数据集路径: {args.data_path}")
    logger.info(f"图像列表: {args.img_list}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"输出目录: {args.out_dir}")
    logger.info(f"Tau 参数: {args.tau}")
    logger.info(f"使用 CRF: {args.use_crf}, 仅 CRF: {args.crf_only}")
    logger.info("=" * 60)
    
    # 调用核心生成逻辑
    run_generation(args)


def main():
    """命令行模式：使用命令行参数"""
    args = parse_args()
    run_generation(args)


if __name__ == "__main__":
    import sys
    
    # 如果有命令行参数（除了脚本名），使用命令行模式
    if len(sys.argv) > 1:
        main()
    else:
        # 默认调用 voc12_inference() 进行 VOC12 数据集推理
        logger.info("未提供命令行参数，使用默认配置运行 voc12_inference()")
        logger.info("如需使用命令行参数，请参考: python generate_sp.py --help")
        logger.info("")
        voc12_inference()
