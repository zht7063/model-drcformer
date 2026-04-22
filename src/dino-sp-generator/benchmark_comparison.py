"""
性能对比测试脚本
对比原始版本和批量优化版本的处理速度
"""

import time
import sys
from pathlib import Path
from loguru import logger

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_original(num_images=20):
    """测试原始版本"""
    from generate_sp import run_generation
    from pathlib import Path
    
    class Args:
        def __init__(self):
            BASE = Path("/home/zhangpeng/tyb")
            self.data_path = str(BASE / "voc2012")
            self.img_list = str(BASE / "voc2012" / "voc12")
            self.out_dir = str(BASE / "outputs" / "dino_sp_benchmark_original")
            self.model_path = str(BASE / "pre-weights" / "dino22416b.npy")
            self.tau = 0.3
            self.device = "cuda"
            self.use_crf = True
            self.crf_only = True
            self.save_png = True
            self.save_npy = True
            self.max_count = num_images
    
    args = Args()
    
    logger.info("=" * 70)
    logger.info(f"测试原始版本 - 处理 {num_images} 张图像")
    logger.info("=" * 70)
    
    start_time = time.time()
    run_generation(args)
    elapsed = time.time() - start_time
    
    speed = elapsed / num_images
    logger.info(f"原始版本完成！总时间: {elapsed:.2f}秒, 平均速度: {speed:.3f}秒/张")
    
    return elapsed, speed


def benchmark_batch(num_images=20, batch_size=8, num_workers=4):
    """测试批量优化版本"""
    from generate_sp_batch import run_generation_batch
    from pathlib import Path
    
    class Args:
        def __init__(self):
            BASE = Path("/home/zhangpeng/tyb")
            self.data_path = str(BASE / "voc2012")
            self.img_list = str(BASE / "voc2012" / "voc12")
            self.out_dir = str(BASE / "outputs" / "dino_sp_benchmark_batch")
            self.model_path = str(BASE / "pre-weights" / "dino22416b.npy")
            self.tau = 0.3
            self.device = "cuda"
            self.use_crf = True
            self.crf_only = True
            self.save_png = True
            self.save_npy = True
            self.max_count = num_images
    
    args = Args()
    
    logger.info("=" * 70)
    logger.info(f"测试批量优化版本 - 处理 {num_images} 张图像")
    logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
    logger.info("=" * 70)
    
    start_time = time.time()
    run_generation_batch(args, batch_size=batch_size, num_workers=num_workers)
    elapsed = time.time() - start_time
    
    speed = elapsed / num_images
    logger.info(f"批量版本完成！总时间: {elapsed:.2f}秒, 平均速度: {speed:.3f}秒/张")
    
    return elapsed, speed


def main():
    """运行完整的性能对比测试"""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    num_images = 20  # 测试图像数量
    
    logger.info("🚀 开始性能对比测试")
    logger.info(f"测试图像数量: {num_images}")
    logger.info("")
    
    # 测试原始版本
    try:
        time_orig, speed_orig = benchmark_original(num_images)
    except Exception as e:
        logger.error(f"原始版本测试失败: {e}")
        time_orig, speed_orig = None, None
    
    logger.info("")
    logger.info("-" * 70)
    logger.info("")
    
    # 测试批量版本（不同的 batch size）
    batch_results = []
    for batch_size in [4, 8, 16]:
        try:
            logger.info(f"测试 batch_size={batch_size}")
            time_batch, speed_batch = benchmark_batch(num_images, batch_size=batch_size, num_workers=4)
            batch_results.append((batch_size, time_batch, speed_batch))
        except Exception as e:
            logger.error(f"批量版本 (batch={batch_size}) 测试失败: {e}")
        
        logger.info("")
    
    # 打印对比结果
    logger.info("=" * 70)
    logger.info("📊 性能对比结果")
    logger.info("=" * 70)
    
    if time_orig:
        logger.info(f"原始版本:        {speed_orig:.3f} 秒/张  (总时间: {time_orig:.2f}秒)")
    
    for batch_size, time_batch, speed_batch in batch_results:
        speedup = speed_orig / speed_batch if time_orig else 0
        logger.info(f"批量版本(B={batch_size:2d}): {speed_batch:.3f} 秒/张  (总时间: {time_batch:.2f}秒) - 🚀 {speedup:.2f}x 加速")
    
    logger.info("=" * 70)
    
    # 估算完整数据集时间
    if batch_results:
        best_batch_size, best_time, best_speed = min(batch_results, key=lambda x: x[2])
        total_images = 10582
        
        logger.info("")
        logger.info("⏱️  预估完整数据集 (10,582张图像) 处理时间:")
        if time_orig:
            est_orig = speed_orig * total_images / 3600
            logger.info(f"  原始版本:             {est_orig:.2f} 小时")
        
        est_batch = best_speed * total_images / 3600
        logger.info(f"  批量优化版本(B={best_batch_size}): {est_batch:.2f} 小时")
        
        if time_orig:
            time_saved = (speed_orig - best_speed) * total_images / 3600
            logger.info(f"  💡 节省时间: {time_saved:.2f} 小时")


if __name__ == "__main__":
    main()

