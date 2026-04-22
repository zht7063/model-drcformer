"""
优化版本：支持批量推理的超像素生成

主要改进：
1. 批量 GPU 推理 - 提升 GPU 利用率
2. 多进程后处理 - 并行执行聚类和 CRF
3. 显著提升处理速度（预计 3-5倍加速）

使用方法：
    from generate_sp_batch import voc12_inference_batch
    voc12_inference_batch(batch_size=8, num_workers=4)
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

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


def process_features_single(args_tuple):
    """
    单张图像的后处理（聚类 + CRF）
    用于多进程并行处理
    """
    features_np, h, w, img0, h0, w0, img_name, tau, use_crf, out_dir_png, out_dir_npy, save_png, save_npy, crf_only = args_tuple
    
    try:
        from skimage import transform
        from PIL import Image
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels
        
        # 归一化特征
        f = features_np / np.sqrt(np.sum(features_np**2, axis=1, keepdims=True))
        
        # 计算相似度矩阵
        apq = f @ f.T
        bpq = apq > 0
        
        # 超像素聚类算法
        pset = np.ones([h*w, 1], dtype=bool)
        selset = np.zeros([0, h*w], dtype=np.int32)
        count = 0
        
        while np.sum(pset) != 0:
            bpq_m = bpq * pset * pset.T
            dp = np.sum(bpq_m, 0)
            dp[dp == 0] = h * w + 1
            pstar = np.argmin(dp)
            S = apq[pstar]
            set1 = (S > tau) & (pset[:, 0] == 1)
            selset = np.concatenate([selset, set1.reshape(1, h*w)], axis=0)
            pset[set1 == True] = 0
            count += 1
        
        # 生成超像素标签
        npimg = np.argmax(selset, axis=0).reshape(h, w)
        
        # 调整到原始大小
        npimg_resized = transform.resize(
            npimg, (h0, w0),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.int32)
        
        # CRF 后处理
        if use_crf:
            d = dcrf.DenseCRF2D(w0, h0, np.max(npimg_resized)+1)
            unary = unary_from_labels(npimg_resized, np.max(npimg_resized)+1, gt_prob=0.7, zero_unsure=False)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=10, compat=10)
            d.addPairwiseBilateral(sxy=20, srgb=5, rgbim=np.ascontiguousarray(np.copy(img0)), compat=10)
            q = d.inference(20)  # CRF迭代次数保持20次，确保输出质量
            npimg_resized = np.argmax(np.array(q).reshape((np.max(npimg_resized)+1, h0, w0)), axis=0)
        
        # 保存 PNG
        if save_png:
            palette = get_voc_palette(np.max(npimg_resized) + 2)
            result_img = Image.fromarray((npimg_resized + 1).astype(np.uint8), mode='P')
            result_img.putpalette(palette)
            result_img.save(os.path.join(out_dir_png, f"{img_name}.png"))
        
        # 保存 NPY
        if save_npy:
            np.save(os.path.join(out_dir_npy, f"{img_name}.npy"), npimg_resized.astype(np.int32))
        
        return True, img_name, count
        
    except Exception as e:
        return False, img_name, str(e)


def get_voc_palette(num_cls):
    """生成 VOC 调色板"""
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j*3+0] = 0
        palette[j*3+1] = 0
        palette[j*3+2] = 0
        i = 0
        while lab > 0:
            palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return palette


def run_generation_batch(args, batch_size=8, num_workers=4):
    """
    批量生成超像素的核心逻辑
    
    Args:
        args: 参数对象
        batch_size: GPU 批量推理的大小（建议 4-16）
        num_workers: CPU 多进程数量（建议 4-8）
    """
    from PIL import Image
    from skimage import transform as sk_transform
    
    # 准备输出目录
    base_out = Path(args.out_dir)
    png_dir = base_out / "png"
    npy_dir = base_out / "npy"
    
    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建数据集
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
    
    # 获取 resume 参数（默认启用）
    resume = getattr(args, 'resume', True)
    
    if resume:
        logger.info("断点续传已启用：将自动跳过已完成的图像")
    
    # 初始化生成器
    generator = SuperPixelGenerator(model_path=args.model_path, device=args.device)
    generator.model.eval()
    
    # 选择要处理的图像
    total = len(dataset)
    if args.max_count is not None:
        total_selected = min(total, args.max_count)
        indices = list(range(total_selected))
    else:
        total_selected = total
        indices = list(range(total))
    
    # 过滤掉已完成的图像（断点续传）
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
    
    # 创建进度条
    if tqdm is not None:
        pbar = tqdm(total=total_selected, desc="Processing", unit="img")
    
    # 计算总批次数
    total_batches = (total_selected + batch_size - 1) // batch_size
    logger.info(f"开始批量处理: 共 {total_batches} 个批次，每批次 {batch_size} 张图像")
    
    # 批量处理（按相同尺寸分组）
    # 外层异常处理：确保即使出现未预期的异常，也能记录并继续
    try:
        for batch_idx, batch_start in enumerate(range(0, total_selected, batch_size), 1):
            batch_end = min(batch_start + batch_size, total_selected)
            batch_indices = indices[batch_start:batch_end]
            
            # logger.debug(f"处理批次 {batch_idx}/{total_batches}: 图像索引 {batch_start}-{batch_end-1}")
            
            # 每个batch的独立异常处理，确保单个batch失败不影响后续处理
            try:
                # === 步骤1: 批量加载图像并按尺寸分组 ===
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
                            # 加载和预处理图像
                            img = Image.open(img_file).convert('RGB')
                            img_array = np.array(img)
                            h0, w0 = img_array.shape[:2]
                            P = generator.model.P
                            
                            h = int((h0 - 1) // P + 1)
                            w = int((w0 - 1) // P + 1)
                            
                            resized_img = sk_transform.resize(
                                img_array, (h * P, w * P),
                                order=3, preserve_range=True, anti_aliasing=True
                            ) / 255.0
                            
                            img_tensor = torch.from_numpy(resized_img).float().permute(2, 0, 1)
                            
                            batch_data.append({
                                'tensor': img_tensor,
                                'name': img_name,
                                'h': h, 'w': w,
                                'img0': img_array,
                                'h0': h0, 'w0': w0,
                                'shape': (h, w)  # 用于分组
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
                
                # === 步骤2: 按尺寸分组并分别推理 ===
                # 将相同尺寸的图像分为一组
                from collections import defaultdict
                shape_groups = defaultdict(list)
                for idx, data in enumerate(batch_data):
                    shape_groups[data['shape']].append(idx)
                
                features_list = [None] * len(batch_data)
                inference_errors = 0
                
                for shape, shape_indices in shape_groups.items():
                    try:
                        # 只stack相同尺寸的tensor
                        sub_batch_tensors = torch.stack([batch_data[i]['tensor'] for i in shape_indices]).to(args.device)
                        
                        with torch.no_grad():
                            sub_features_batch = generator.model(sub_batch_tensors)  # [B, H*W, C]
                        
                        # 将特征放回对应位置
                        for batch_idx_in_shape, features in enumerate(sub_features_batch):
                            features_list[shape_indices[batch_idx_in_shape]] = features.cpu().numpy()
                            
                    except Exception as e:
                        logger.error(f"Batch inference failed for shape {shape} in batch {batch_idx}: {e}", exc_info=True)
                        for idx in shape_indices:
                            n_err += 1
                            inference_errors += 1
                            features_list[idx] = None
                
                if inference_errors > 0:
                    logger.warning(f"批次 {batch_idx} 中有 {inference_errors} 个推理失败")
                
                # === 步骤3: CPU 多进程后处理 ===
                # 过滤掉推理失败的样本
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
                            args.crf_only
                        ))
                
                if len(process_args) == 0:
                    logger.warning(f"批次 {batch_idx} 没有可处理的样本")
                    if tqdm is not None:
                        pbar.update(len(batch_data))
                    continue
                
                # 使用多进程处理，添加异常处理
                post_process_errors = 0
                try:
                    with Pool(processes=num_workers) as pool:
                        # 使用 map_async 可以更好地处理异常
                        async_result = pool.map_async(process_features_single, process_args)
                        # 设置超时（每个图像最多5分钟）
                        timeout = len(process_args) * 300
                        try:
                            results = async_result.get(timeout=timeout)
                        except Exception as e:
                            logger.error(f"多进程处理超时或异常 (batch {batch_idx}): {e}", exc_info=True)
                            # 尝试终止池并获取部分结果
                            pool.terminate()
                            pool.join()
                            results = []
                            post_process_errors = len(process_args)
                except Exception as e:
                    logger.error(f"多进程池创建或执行失败 (batch {batch_idx}): {e}", exc_info=True)
                    results = []
                    post_process_errors = len(process_args)
                
                # 统计结果
                for success, name, info in results:
                    if success:
                        n_ok += 1
                    else:
                        n_err += 1
                        post_process_errors += 1
                        logger.error(f"Post-processing failed for {name}: {info}")
                
                if post_process_errors > 0:
                    logger.warning(f"批次 {batch_idx} 中有 {post_process_errors} 个后处理失败")
                
                # 清理GPU缓存，防止内存累积
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 更新进度条
                if tqdm is not None:
                    pbar.update(len(batch_data))
                
                # 每处理10个batch输出一次进度
                if batch_idx % 10 == 0:
                    logger.info(f"进度: {batch_idx}/{total_batches} 批次完成, 已保存 {n_ok} 张, 失败 {n_err} 张")
                    
            except Exception as e:
                # 单个batch的异常处理，记录但继续处理下一个batch
                logger.error(f"批次 {batch_idx} 处理失败: {e}", exc_info=True)
                n_err += len(batch_indices)
                if tqdm is not None:
                    pbar.update(len(batch_indices))
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 继续处理下一个batch
                continue
                
    except KeyboardInterrupt:
        logger.warning("收到中断信号，正在保存进度...")
        raise
    except Exception as e:
        # 最外层异常处理，记录所有未预期的异常
        logger.error(f"批量处理过程中发生未预期的异常: {e}", exc_info=True)
        import traceback
        logger.error(f"异常堆栈:\n{traceback.format_exc()}")
        raise
    finally:
        # 确保进度条被关闭
        if tqdm is not None:
            pbar.close()
        
        # 最终清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"Done. Total={total_selected}, Saved={n_ok}, Failed={n_err}, Skipped={n_skip}")
    logger.info(f"PNG -> {png_dir}, NPY -> {npy_dir}")


def voc12_inference_batch(batch_size=8, num_workers=None):
    """
    VOC12 数据集批量推理函数（优化版本）
    
    Args:
        batch_size: GPU 批量大小（建议 4-16，取决于显存）
        num_workers: CPU 进程数（默认为 CPU 核心数的一半）
    """
    # 设置 CUDA 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # 自动设置 worker 数量
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)
    
    class Args:
        def __init__(self):
            BASE = Path("/home/zhangpeng/tyb")
            self.data_path = str(BASE / "voc2012")
            self.img_list = str(BASE / "voc2012" / "voc12")
            self.out_dir = str(BASE / "outputs" / "dino_sp")
            self.model_path = str(BASE / "pre-weights" / "dino22416b.npy")
            self.tau = 0.3
            self.device = "cuda"
            self.use_crf = True
            self.crf_only = True
            self.save_png = False
            self.save_npy = True
            self.max_count = None
    
    args = Args()
    
    logger.info("=" * 60)
    logger.info("VOC12 超像素生成 - 批量推理优化版本")
    logger.info(f"Batch size: {batch_size}, CPU workers: {num_workers}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"数据集路径: {args.data_path}")
    logger.info(f"输出目录: {args.out_dir}")
    logger.info("=" * 60)
    
    run_generation_batch(args, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    import sys
    
    # 解析简单的命令行参数
    batch_size = 8
    num_workers = 4  # 推荐4-8，过多会导致竞争
    
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except:
            pass
    
    if len(sys.argv) > 2:
        try:
            num_workers = int(sys.argv[2])
        except:
            pass
    
    logger.info(f"使用参数: batch_size={batch_size}, num_workers={num_workers}")
    voc12_inference_batch(batch_size=batch_size, num_workers=num_workers)

