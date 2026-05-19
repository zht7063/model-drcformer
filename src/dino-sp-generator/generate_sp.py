"""
VOC12 超像素生成脚本

功能：
1. 使用 DINO 模型为 VOC12 数据集生成超像素分割
2. 支持 GPU 批量推理与多进程后处理（默认 batch_size=8）
3. 支持 CRF 后处理
4. 输出 PNG 可视化文件和 NPY 索引数组

快速使用：
---------
方式1 - 直接调用 voc12_inference 函数:
    conda activate py310
    python -c "from generate_sp import voc12_inference; voc12_inference()"

方式2 - 命令行参数（批量推理，推荐）:
    python generate_sp.py \\
        --data-path ~/projects/voc2012 \\
        --img-list ~/projects/voc2012/voc12 \\
        --model-path ~/projects/pre-weights/dino22416b.npy \\
        --out-dir ~/projects/outputs/dino_sp \\
        --batch-size 8 --num-workers 4 \\
        --tau 0.3 --use-crf --crf-only --no-png

方式3 - 串行推理（batch_size=1）:
    python generate_sp.py --batch-size 1 ...
"""

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# 确保当前目录在 sys.path 最前面，优先导入本目录下的 datasets.py
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from datasets import VOC12Dataset, COCODataset
from dino_sp_generator import SuperPixelGenerator

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def should_skip_image(img_name, npy_dir, png_dir, save_npy, save_png, resume):
    """检查图像是否应该被跳过（断点续传）"""
    if not resume:
        return False

    if save_npy and not (npy_dir / f"{img_name}.npy").exists():
        return False
    if save_png and not (png_dir / f"{img_name}.png").exists():
        return False

    return True


def get_voc_palette(num_cls):
    """生成 VOC 调色板（批量路径 PNG 保存用）"""
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return palette


def process_features_single(args_tuple):
    """单张图像的后处理（聚类 + CRF），用于多进程并行处理"""
    (
        features_np, h, w, img0, h0, w0, img_name, tau, use_crf,
        out_dir_png, out_dir_npy, save_png, save_npy, crf_only,
    ) = args_tuple

    try:
        from skimage import transform
        from PIL import Image
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels

        f = features_np / np.sqrt(np.sum(features_np**2, axis=1, keepdims=True))

        apq = f @ f.T
        bpq = apq > 0

        pset = np.ones([h * w, 1], dtype=bool)
        selset = np.zeros([0, h * w], dtype=np.int32)
        count = 0

        while np.sum(pset) != 0:
            bpq_m = bpq * pset * pset.T
            dp = np.sum(bpq_m, 0)
            dp[dp == 0] = h * w + 1
            pstar = np.argmin(dp)
            S = apq[pstar]
            set1 = (S > tau) & (pset[:, 0] == 1)
            selset = np.concatenate([selset, set1.reshape(1, h * w)], axis=0)
            pset[set1 == True] = 0
            count += 1

        npimg = np.argmax(selset, axis=0).reshape(h, w)

        npimg_resized = transform.resize(
            npimg, (h0, w0),
            order=0, preserve_range=True, anti_aliasing=False,
        ).astype(np.int32)

        if use_crf:
            d = dcrf.DenseCRF2D(w0, h0, np.max(npimg_resized) + 1)
            unary = unary_from_labels(
                npimg_resized, np.max(npimg_resized) + 1,
                gt_prob=0.7, zero_unsure=False,
            )
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=10, compat=10)
            d.addPairwiseBilateral(
                sxy=20, srgb=5,
                rgbim=np.ascontiguousarray(np.copy(img0)), compat=10,
            )
            q = d.inference(20)
            npimg_resized = np.argmax(
                np.array(q).reshape((np.max(npimg_resized) + 1, h0, w0)), axis=0,
            )

        if save_png:
            palette = get_voc_palette(np.max(npimg_resized) + 2)
            result_img = Image.fromarray((npimg_resized + 1).astype(np.uint8), mode='P')
            result_img.putpalette(palette)
            result_img.save(os.path.join(out_dir_png, f"{img_name}.png"))

        if save_npy:
            np.save(
                os.path.join(out_dir_npy, f"{img_name}.npy"),
                npimg_resized.astype(np.int32),
            )

        return True, img_name, count

    except Exception as e:
        return False, img_name, str(e)


def parse_args():
    parser = argparse.ArgumentParser(
        "Generate superpixel index maps (PNG + NPY) for VOC12/COCO datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="voc2012", choices=["voc2012", "coco2014"],
        help="Dataset type: voc2012 or coco2014 (default: voc2012)",
    )
    parser.add_argument(
        "--data-path", default="/home/zht/VOC2012", type=str,
        help="Dataset root path (VOC2012: contains JPEGImages/)",
    )
    parser.add_argument(
        "--img-list", default="voc12", type=str,
        help="VOC2012: dir with train_aug_id.txt/val_id.txt, or path to a .txt file",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="COCO2014 split (default: train). Ignored for VOC2012.",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="assets/saved_models/sp_generator/dino22416b.npy",
        help="Path to DINO numpy weights",
    )
    parser.add_argument(
        "--out-dir", type=str, default="assets/sp_idx",
        help="Base output directory; creates subdirs 'png' and 'npy'",
    )
    parser.add_argument(
        "--tau", type=float, default=0.4,
        help="Threshold tau for superpixel clustering",
    )
    parser.add_argument(
        "--use-crf", action="store_true", default=True,
        help="Enable CRF post-processing (default: on)",
    )
    parser.add_argument(
        "--crf-only", action="store_true", default=False,
        help="Only save CRF post-processed results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--max-count", type=int, default=None,
        help="Process at most N images (for quick test runs)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="GPU batch size; use 1 for serial inference (default: 8)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="CPU workers for post-processing (default: half of cpu_count)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Disable resume: reprocess all images",
    )
    parser.add_argument(
        "--no-png", dest="save_png", action="store_false",
        help="Do not write PNG masks",
    )
    parser.add_argument(
        "--no-npy", dest="save_npy", action="store_false",
        help="Do not write NPY index arrays",
    )
    parser.set_defaults(save_png=True, save_npy=True, resume=True)
    return parser.parse_args()


def run_generation_serial(args):
    """串行生成：逐张调用 SuperPixelGenerator.generate_superpixel"""
    base_out = Path(args.out_dir)
    png_dir = base_out / "png"
    npy_dir = base_out / "npy"
    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)

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

    generator = SuperPixelGenerator(model_path=args.model_path, device=args.device)

    total = len(dataset)
    n_ok, n_err = 0, 0
    n_skip = 0

    resume = getattr(args, 'resume', True)
    if resume:
        logger.info("断点续传已启用：将自动跳过已完成的图像")

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

    out_dir_for_gen = (str(png_dir) + "/") if args.save_png else None
    tmp_ctx = None
    if not args.save_png:
        try:
            import tempfile
            tmp_ctx = tempfile.TemporaryDirectory()
            out_dir_for_gen = tmp_ctx.name.rstrip("/") + "/"
        except Exception as e:
            logger.warning(
                f"Failed to create TemporaryDirectory for PNG suppression, "
                f"fallback to {png_dir}: {e}"
            )
            png_dir.mkdir(parents=True, exist_ok=True)
            out_dir_for_gen = str(png_dir) + "/"

    for idx, dataset_idx in enumerate(iterable, start=1):
        sample = dataset[dataset_idx]
        img_file = sample['img_path']
        img_name = sample['img_name']

        if not os.path.exists(img_file):
            logger.warning(f"[{idx}/{total_selected}] Image not found: {img_file}")
            n_err += 1
            continue

        if should_skip_image(img_name, npy_dir, png_dir, args.save_npy, args.save_png, resume):
            n_skip += 1
            if (tqdm is None) and (idx % 50 == 0 or idx == total_selected):
                logger.info(f"Progress: {idx}/{total_selected} | OK={n_ok}, ERR={n_err}, SKIP={n_skip}")
            continue

        try:
            npimg, count = generator.generate_superpixel(
                img_file=img_file,
                tau=args.tau,
                out_dir=out_dir_for_gen,
                use_crf=args.use_crf,
                crf_only=args.crf_only,
                if_tip=False,
            )

            if args.save_npy:
                out_npy = npy_dir / f"{img_name}.npy"
                np.save(out_npy, np.asarray(npimg, dtype=np.int32))

            n_ok += 1
            if (tqdm is None) and (idx % 50 == 0 or idx == total_selected):
                logger.info(f"Progress: {idx}/{total_selected} | OK={n_ok}, ERR={n_err}, SKIP={n_skip}")
        except Exception as e:
            logger.error(f"[{idx}/{total_selected}] Failed on {img_name}: {e}")
            n_err += 1

    if tmp_ctx is not None:
        try:
            tmp_ctx.cleanup()
        except Exception:
            pass

    logger.info(
        f"Done. Total={total}, Saved={n_ok}, Failed={n_err}, Skipped={n_skip}. "
        f"PNG -> {png_dir}, NPY -> {npy_dir}"
    )


def run_generation_batch(args, batch_size=8, num_workers=4):
    """批量生成：GPU 批量前向 + 多进程聚类/CRF 后处理"""
    from PIL import Image
    from skimage import transform as sk_transform

    base_out = Path(args.out_dir)
    png_dir = base_out / "png"
    npy_dir = base_out / "npy"

    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Batch size: {batch_size}, CPU workers: {num_workers}")

    resume = getattr(args, 'resume', True)
    if resume:
        logger.info("断点续传已启用：将自动跳过已完成的图像")

    generator = SuperPixelGenerator(model_path=args.model_path, device=args.device)
    generator.model.eval()

    total = len(dataset)
    if args.max_count is not None:
        total_selected = min(total, args.max_count)
        indices = list(range(total_selected))
    else:
        total_selected = total
        indices = list(range(total))

    if resume:
        remaining_indices = []
        n_skip = 0
        for dataset_idx in indices:
            sample = dataset[dataset_idx]
            img_name = sample['img_name']
            if should_skip_image(img_name, npy_dir, png_dir, args.save_npy, args.save_png, resume):
                n_skip += 1
            else:
                remaining_indices.append(dataset_idx)

        if n_skip > 0:
            logger.info(f"跳过 {n_skip} 个已完成的图像，剩余 {len(remaining_indices)} 个图像需要处理")
        indices = remaining_indices
        total_selected = len(indices)
    else:
        n_skip = 0

    n_ok, n_err = 0, 0

    if tqdm is not None:
        pbar = tqdm(total=total_selected, desc="Processing", unit="img")

    total_batches = (total_selected + batch_size - 1) // batch_size if total_selected > 0 else 0
    logger.info(f"开始批量处理: 共 {total_batches} 个批次，每批次 {batch_size} 张图像")

    try:
        for batch_idx, batch_start in enumerate(range(0, total_selected, batch_size), 1):
            batch_end = min(batch_start + batch_size, total_selected)
            batch_indices = indices[batch_start:batch_end]

            try:
                batch_data = []
                batch_load_errors = 0

                for dataset_idx in batch_indices:
                    try:
                        sample = dataset[dataset_idx]
                        img_file = sample['img_path']
                        img_name = sample['img_name']

                        if not os.path.exists(img_file):
                            logger.warning(f"Image not found: {img_file}")
                            n_err += 1
                            batch_load_errors += 1
                            continue

                        try:
                            img = Image.open(img_file).convert('RGB')
                            img_array = np.array(img)
                            h0, w0 = img_array.shape[:2]
                            P = generator.model.P

                            h = int((h0 - 1) // P + 1)
                            w = int((w0 - 1) // P + 1)

                            resized_img = sk_transform.resize(
                                img_array, (h * P, w * P),
                                order=3, preserve_range=True, anti_aliasing=True,
                            ) / 255.0

                            img_tensor = torch.from_numpy(resized_img).float().permute(2, 0, 1)

                            batch_data.append({
                                'tensor': img_tensor,
                                'name': img_name,
                                'h': h, 'w': w,
                                'img0': img_array,
                                'h0': h0, 'w0': w0,
                                'shape': (h, w),
                            })
                        except Exception as e:
                            logger.error(f"Failed to load {img_name}: {e}", exc_info=True)
                            n_err += 1
                            batch_load_errors += 1
                    except Exception as e:
                        logger.error(f"Failed to get sample for index {dataset_idx}: {e}", exc_info=True)
                        n_err += 1
                        batch_load_errors += 1

                if len(batch_data) == 0:
                    logger.warning(f"批次 {batch_idx} 没有有效图像，跳过")
                    if tqdm is not None:
                        pbar.update(len(batch_indices))
                    continue

                if batch_load_errors > 0:
                    logger.warning(f"批次 {batch_idx} 中有 {batch_load_errors} 张图像加载失败")

                shape_groups = defaultdict(list)
                for idx, data in enumerate(batch_data):
                    shape_groups[data['shape']].append(idx)

                features_list = [None] * len(batch_data)
                inference_errors = 0

                for shape, shape_indices in shape_groups.items():
                    try:
                        sub_batch_tensors = torch.stack(
                            [batch_data[i]['tensor'] for i in shape_indices],
                        ).to(args.device)

                        with torch.no_grad():
                            sub_features_batch = generator.model(sub_batch_tensors)

                        for batch_idx_in_shape, features in enumerate(sub_features_batch):
                            features_list[shape_indices[batch_idx_in_shape]] = features.cpu().numpy()

                    except Exception as e:
                        logger.error(
                            f"Batch inference failed for shape {shape} in batch {batch_idx}: {e}",
                            exc_info=True,
                        )
                        for idx in shape_indices:
                            n_err += 1
                            inference_errors += 1
                            features_list[idx] = None

                if inference_errors > 0:
                    logger.warning(f"批次 {batch_idx} 中有 {inference_errors} 个推理失败")

                process_args = []
                for i in range(len(batch_data)):
                    if features_list[i] is not None:
                        process_args.append((
                            features_list[i],
                            batch_data[i]['h'], batch_data[i]['w'],
                            batch_data[i]['img0'],
                            batch_data[i]['h0'], batch_data[i]['w0'],
                            batch_data[i]['name'],
                            args.tau, args.use_crf,
                            str(png_dir), str(npy_dir),
                            args.save_png, args.save_npy,
                            args.crf_only,
                        ))

                if len(process_args) == 0:
                    logger.warning(f"批次 {batch_idx} 没有可处理的样本")
                    if tqdm is not None:
                        pbar.update(len(batch_data))
                    continue

                post_process_errors = 0
                try:
                    with Pool(processes=num_workers) as pool:
                        async_result = pool.map_async(process_features_single, process_args)
                        timeout = len(process_args) * 300
                        try:
                            results = async_result.get(timeout=timeout)
                        except Exception as e:
                            logger.error(f"多进程处理超时或异常 (batch {batch_idx}): {e}", exc_info=True)
                            pool.terminate()
                            pool.join()
                            results = []
                            post_process_errors = len(process_args)
                except Exception as e:
                    logger.error(f"多进程池创建或执行失败 (batch {batch_idx}): {e}", exc_info=True)
                    results = []
                    post_process_errors = len(process_args)

                for success, name, info in results:
                    if success:
                        n_ok += 1
                    else:
                        n_err += 1
                        post_process_errors += 1
                        logger.error(f"Post-processing failed for {name}: {info}")

                if post_process_errors > 0:
                    logger.warning(f"批次 {batch_idx} 中有 {post_process_errors} 个后处理失败")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if tqdm is not None:
                    pbar.update(len(batch_data))

                if batch_idx % 10 == 0:
                    logger.info(
                        f"进度: {batch_idx}/{total_batches} 批次完成, "
                        f"已保存 {n_ok} 张, 失败 {n_err} 张"
                    )

            except Exception as e:
                logger.error(f"批次 {batch_idx} 处理失败: {e}", exc_info=True)
                n_err += len(batch_indices)
                if tqdm is not None:
                    pbar.update(len(batch_indices))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    except KeyboardInterrupt:
        logger.warning("收到中断信号，正在保存进度...")
        raise
    except Exception as e:
        logger.error(f"批量处理过程中发生未预期的异常: {e}", exc_info=True)
        import traceback
        logger.error(f"异常堆栈:\n{traceback.format_exc()}")
        raise
    finally:
        if tqdm is not None:
            pbar.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Done. Total={total_selected}, Saved={n_ok}, Failed={n_err}, Skipped={n_skip}")
    logger.info(f"PNG -> {png_dir}, NPY -> {npy_dir}")


def run_generation(args):
    """统一入口：batch_size > 1 走批量路径，否则走串行路径"""
    batch_size = getattr(args, "batch_size", 8)
    num_workers = getattr(args, "num_workers", None)
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    if batch_size <= 1:
        logger.info("使用串行推理 (batch_size <= 1)")
        return run_generation_serial(args)

    logger.info(f"使用批量 GPU 推理 (batch_size={batch_size}, num_workers={num_workers})")
    return run_generation_batch(args, batch_size=batch_size, num_workers=num_workers)


def voc12_inference():
    """
    VOC12 数据集推理（默认批量模式）

    使用方式：
        from generate_sp import voc12_inference
        voc12_inference()
    """
    projects = Path.home() / "projects"

    class Args:
        def __init__(self):
            self.data_path = str(projects / "datasets" / "voc2012aug")
            self.img_list = str(projects / "datasets" / "voc2012aug" / "voc12")
            self.out_dir = str(projects / "datasets" / "dino_sp")
            self.model_path = str(projects / "pre-weights" / "dino22416b.npy")
            self.tau = 0.3
            self.device = "cuda"
            self.use_crf = True
            self.crf_only = True
            self.save_png = False
            self.save_npy = True
            self.max_count = None
            self.batch_size = 8
            self.num_workers = 4
            self.resume = True

    args = Args()

    logger.info("=" * 60)
    logger.info("开始 VOC12 数据集超像素生成")
    logger.info(f"数据集路径: {args.data_path}")
    logger.info(f"图像列表: {args.img_list}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"输出目录: {args.out_dir}")
    logger.info(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    logger.info(f"Tau: {args.tau}, CRF: {args.use_crf}, crf_only: {args.crf_only}")
    logger.info("=" * 60)

    run_generation(args)


def main():
    """命令行模式：使用命令行参数"""
    args = parse_args()
    run_generation(args)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        logger.info("未提供命令行参数，使用默认配置运行 voc12_inference()")
        logger.info("如需使用命令行参数，请参考: python generate_sp.py --help")
        logger.info("")
        voc12_inference()
