#!/usr/bin/env python3
"""
DRCFormer 模型在 COCO 数据集上的训练、CAM生成和评估运行脚本

使用方法：
    # ===== 单GPU训练 =====
    # 运行完整流程（默认单GPU，训练集）
    python run_coco.py --all --split train
    
    # 指定GPU设备
    python run_coco.py --all --split train --device cuda:1
    
    # 单独运行某个步骤
    python run_coco.py --step train --split train
    python run_coco.py --step cam --split train
    python run_coco.py --step eval --split train
    
    # ===== 多GPU分布式训练（DDP）=====
    # 使用4张GPU进行训练
    python run_coco.py --all --split train --num_gpus 4
    
    # 使用2张GPU，自定义batch size
    python run_coco.py --all --split train --num_gpus 2 --batch_size 128
    
    # 仅运行训练步骤（多GPU）
    python run_coco.py --step train --split train --num_gpus 4
    
    # ===== 自定义参数 =====
    # 自定义实验ID、epoch和损失权重
    python run_coco.py --all --split train --exp_id my_exp --epochs 100 --w_lsp 1000
    
    # 多GPU训练 + 自定义参数
    python run_coco.py --all --split train --num_gpus 4 --batch_size 32 --epochs 60

    # ===== 指定实验 id 运行第二、三阶段 =====
    # 生成 CAM（默认只保存 npy 文件，不保存 PNG 图片）
    python run_coco.py --step cam --split train --exp_id exp_011416
    
    # 生成 CAM 并保存 PNG 可视化图片
    python run_coco.py --step cam --split train --exp_id exp_011416 --save-png
    
    # 评估 CAM
    python run_coco.py --step eval --split train --exp_id exp_011416
    
    # ===== 批量推理优化 =====
    # 使用批量推理加速 CAM 生成
    python run_coco.py --step cam --split train --exp_id exp_011416 --enable-batch-inference --cam-batch-size 16

注意事项：
    - 单GPU模式（num_gpus=1）：使用 --device 参数指定设备
    - 多GPU模式（num_gpus>1）：自动使用 torchrun 启动DDP分布式训练
    - CAM生成和评估步骤始终使用单GPU运行
    - 多GPU训练时，有效的全局batch size = batch_size × num_gpus
    - COCO数据集有80个前景类别（+1背景类=81类）
    - 需要先生成类别标签文件：tools/generate_coco_cls_labels.py
    - 需要先生成GT分割标签：tools/convert_coco_to_segmentation_masks.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def _get_base_dir() -> Path:
    """根据 AGENTS.md 配置返回项目根目录: ~/projects"""
    return Path.home() / "projects"


class COCORunner:
    """COCO 数据集 DRCFormer 运行器"""
    
    def __init__(self, args):
        # 项目路径配置（见 AGENTS.md）
        self.base = _get_base_dir()
        self.project_dir = self.base / "drcformer" / "src" / "drcformer"
        
        # 实验配置
        self.split = args.split  # "train" or "val"
        self.exp_id = args.exp_id
        self.device = args.device
        self.num_gpus = args.num_gpus
        
        # 数据集路径
        self.dataset_path = self.base / "coco2014"
        self.img_dir = self.dataset_path / f"{self.split}2014"
        
        # 图像列表文件（可选）
        self.img_list_file = self.dataset_path / f"{self.split}_id.txt"
        
        # 超像素和预训练权重
        self.sp_idx_dir = self.base / "outputs" / "dino_sp" / "dino_sp_coco" / "npy"
        self.pretrained_weight = self.base / "pre-weights" / "suit_base_224.pth"
        
        # 输出路径
        self.output_dir = self.base / "outputs" / "drcformer" / f"coco_{self.split}_{self.exp_id}"
        self.tensorboard_dir = self.base / "outputs" / "drcformer" / "tensorboard" / f"coco_{self.split}_{self.exp_id}"
        
        # 训练超参数
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.seed = args.seed
        
        # 损失权重
        self.w_ent = args.w_ent
        self.w_neg = args.w_neg
        self.w_lsp = args.w_lsp
        
        # CAM 生成选项
        self.save_png = args.save_png
        
        # 批量推理选项
        self.enable_batch_inference = args.enable_batch_inference
        self.cam_batch_size = args.cam_batch_size
        self.cam_progress_interval = args.cam_progress_interval
        
        # B1 分支开关
        self.b1_enable = args.b1_enable
        self.b1_margin = args.b1_margin
    
    def check_files(self):
        """检查必要文件是否存在"""
        checks = [
            (self.pretrained_weight, "预训练权重"),
            (self.dataset_path, "COCO2014数据集根目录"),
            (self.img_dir, f"COCO2014 {self.split} 图像目录"),
        ]
        
        all_ok = True
        for path, name in checks:
            if not path.exists():
                print(f"✗ {name}不存在: {path}")
                all_ok = False
            else:
                print(f"✓ {name}")
        
        # 检查类别标签文件
        cls_labels_file = self.dataset_path / "cls_labels.npy"
        if not cls_labels_file.exists():
            print(f"⚠ 警告: 类别标签文件不存在: {cls_labels_file}")
            print(f"   请先运行: python tools/generate_coco_cls_labels.py")
            print(f"   训练将继续，但所有图像的类别标签将为零向量")
        else:
            print(f"✓ 类别标签文件")
        
        # 检查图像列表文件（可选）
        if self.img_list_file.exists():
            print(f"✓ 图像列表文件: {self.img_list_file}")
        else:
            print(f"ℹ 图像列表文件不存在: {self.img_list_file}")
            print(f"   将自动从 {self.img_dir} 扫描所有图像")
        
        # 检查超像素索引（可选，仅训练时需要）
        if self.sp_idx_dir.exists():
            print(f"✓ 超像素索引目录")
        else:
            print(f"⚠ 警告: 超像素索引目录不存在: {self.sp_idx_dir}")
            print(f"   如果启用了超像素对齐损失，请先生成超像素索引")
            print(f"   运行: python main.py --mode sp-generator --sp-dataset coco2014 --sp-split {self.split}")
        
        return all_ok
    
    def _build_train_args(self):
        """构建训练参数"""
        sys.path.insert(0, str(self.project_dir))
        from main import get_args_parser
        
        parser = get_args_parser()
        train_args = parser.parse_args([])
        
        # 设置 COCO 数据集参数
        train_args.data_path = str(self.dataset_path)
        train_args.data_set = "COCO"
        train_args.img_list = str(self.dataset_path)  # 数据集类会自动查找列表文件或扫描目录
        train_args.coco_split = self.split
        train_args.finetune = str(self.pretrained_weight)
        train_args.output_dir = str(self.output_dir)
        train_args.num_workers = self.num_workers
        train_args.batch_size = self.batch_size
        train_args.seed = self.seed
        train_args.epochs = self.epochs
        train_args.device = self.device
        
        # 模型和优化器参数
        train_args.model = "drcformer"
        train_args.cct_temp = 0.8
        train_args.mct_loss_weight = 1.0
        train_args.attn_loss_weight = 2.0
        train_args.patch_loss_weight = 1.0
        
        # 损失权重交换
        train_args.loss_swap_enable = True
        train_args.loss_swap_epoch = 20
        train_args.loss_swap_warm_epochs = 6
        train_args.new_mct_loss_weight = 1.0
        train_args.new_attn_loss_weight = 1.8
        train_args.new_patch_loss_weight = 1.0
        
        # B1 结构监督
        train_args.b1_enable = self.b1_enable
        train_args.b1_warmup_epochs = 10
        train_args.w_ent = self.w_ent
        train_args.w_neg = self.w_neg
        train_args.b1_margin = self.b1_margin
        train_args.b1_prob_eps = 5e-4
        
        # 超像素对齐
        train_args.sp_align_enable = True
        train_args.sp_idx_dir = str(self.sp_idx_dir)
        train_args.sp_align_apply_to = "all"
        train_args.sp_align_loss_weight = self.w_lsp
        train_args.sp_align_start_epoch = 10
        train_args.sp_align_warmup_epochs = 5
        train_args.sp_overlap_tau = 0.4
        train_args.sp_min_area = 48
        
        # 其他参数
        train_args.clip_grad = 1.0
        train_args.tensorboard = True
        train_args.tensorboard_dir = str(self.tensorboard_dir)
        
        return train_args
    
    def train(self):
        """训练模型"""
        if self.num_gpus > 1:
            # 多GPU模式：使用 subprocess 调用 torchrun
            train_args_list = [
                "--data-path", str(self.dataset_path),
                "--data-set", "COCO",
                "--img-list", str(self.dataset_path),
                "--coco-split", self.split,
                "--finetune", str(self.pretrained_weight),
                "--output_dir", str(self.output_dir),
                "--num_workers", str(self.num_workers),
                "--batch-size", str(self.batch_size),
                "--seed", str(self.seed),
                "--epochs", str(self.epochs),
                "--cct-temp", "0.8",
                "--mct-loss-weight", "1.0",
                "--attn-loss-weight", "2.0",
                "--patch-loss-weight", "1.0",
                "--loss-swap-enable",
                "--loss-swap-epoch", "20",
                "--loss-swap-warm-epochs", "6",
                "--new-mct-loss-weight", "1.0",
                "--new-attn-loss-weight", "1.8",
                "--new-patch-loss-weight", "1.0",
            ]
            
            if self.b1_enable:
                train_args_list.extend([
                    "--b1-enable",
                    "--b1-warmup-epochs", "10",
                    "--w-ent", str(self.w_ent),
                    "--w-neg", str(self.w_neg),
                    "--b1-margin", str(self.b1_margin),
                    "--b1-prob-eps", "5e-4",
                ])
            
            train_args_list.extend([
                "--sp-align-enable",
                "--sp-idx-dir", str(self.sp_idx_dir),
                "--sp-align-apply-to", "all",
                "--sp-align-loss-weight", str(self.w_lsp),
                "--sp-align-start-epoch", "10",
                "--sp-align-warmup-epochs", "5",
                "--sp-overlap-tau", "0.4",
                "--sp-min-area", "48",
                "--clip-grad", "1.0",
                "--tensorboard",
                "--tensorboard-dir", str(self.tensorboard_dir),
            ])
            
            cmd = [
                "torchrun",
                f"--nproc_per_node={self.num_gpus}",
                "--master_port=29500",
                str(self.project_dir / "main.py"),
            ] + train_args_list
            
            print(f"\n{'='*60}\n训练 DRCFormer 模型 (COCO {self.split}, 实验ID: {self.exp_id}, 多GPU: {self.num_gpus})\n{'='*60}")
            try:
                subprocess.run(cmd, check=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ 训练失败 (返回码: {e.returncode})")
                return False
        else:
            # 单GPU模式：直接调用函数
            train_args = self._build_train_args()
            
            sys.path.insert(0, str(self.project_dir))
            from main import main as drcformer_main
            
            print(f"\n{'='*60}\n训练 DRCFormer 模型 (COCO {self.split}, 实验ID: {self.exp_id}, 单GPU: {self.device})\n{'='*60}")
            train_args.distributed = False
            try:
                drcformer_main(train_args)
                return True
            except Exception as e:
                print(f"✗ 训练失败: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def generate_cam(self):
        """生成 CAM"""
        checkpoint = self.output_dir / "checkpoint.pth"
        if not checkpoint.exists():
            print(f"✗ checkpoint不存在: {checkpoint}")
            return False
        
        cam_png_dir = self.output_dir / "cam" / "png"
        cam_npy_dir = self.output_dir / "cam" / "npy"
        cam_png_dir.mkdir(parents=True, exist_ok=True)
        cam_npy_dir.mkdir(parents=True, exist_ok=True)
        
        sys.path.insert(0, str(self.project_dir))
        from main import get_args_parser, main as drcformer_main
        
        parser = get_args_parser()
        cam_args = parser.parse_args([])
        
        # 设置 COCO 多尺度参数
        cam_args.data_set = "COCOMS"
        cam_args.img_list = str(self.dataset_path)
        cam_args.data_path = str(self.dataset_path)
        cam_args.coco_split = self.split
        
        # 批量推理选项
        if self.enable_batch_inference:
            cam_args.gen_attention_maps_batch = True
            cam_args.gen_attention_maps = False
            cam_args.cam_batch_size = self.cam_batch_size
            cam_args.cam_progress_interval = self.cam_progress_interval
        else:
            cam_args.gen_attention_maps = True
            cam_args.gen_attention_maps_batch = False
        
        cam_args.attention_dir = str(cam_png_dir)
        cam_args.cam_npy_dir = str(cam_npy_dir)
        cam_args.resume = str(checkpoint)
        cam_args.device = self.device
        cam_args.save_cam_png = self.save_png
        cam_args.distributed = False
        
        print(f"\n{'='*60}\n生成 CAM (COCO {self.split})\n{'='*60}")
        try:
            drcformer_main(cam_args)
            return True
        except Exception as e:
            print(f"✗ CAM生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate(self):
        """评估 CAM"""
        cam_npy_dir = self.output_dir / "cam" / "npy"
        if not cam_npy_dir.exists() or not any(cam_npy_dir.iterdir()):
            print(f"✗ CAM结果不存在: {cam_npy_dir}")
            return False
        
        # COCO GT 分割标签目录
        gt_dir = self.dataset_path / f"segmentation_masks_{self.split}"
        if not gt_dir.exists():
            print(f"✗ GT分割标签目录不存在: {gt_dir}")
            print(f"   请先运行: python tools/convert_coco_to_segmentation_masks.py")
            return False
        
        # 图像列表文件
        if self.img_list_file.exists():
            list_file = self.img_list_file
        else:
            # 如果没有列表文件，从 CAM 输出目录生成临时列表
            import glob
            cam_files = glob.glob(str(cam_npy_dir / "*.npy"))
            if not cam_files:
                print(f"✗ CAM目录为空: {cam_npy_dir}")
                return False
            
            # 创建临时列表文件
            list_file = self.output_dir / f"temp_{self.split}_list.txt"
            with open(list_file, 'w') as f:
                for cam_file in sorted(cam_files):
                    img_name = Path(cam_file).stem
                    f.write(f"{img_name}\n")
            print(f"ℹ 创建临时图像列表: {list_file}")
        
        logfile = self.output_dir / f"evallog_{self.split}.txt"
        
        # 使用 subprocess 调用 evaluation.py
        cmd = [
            "python", str(self.project_dir / "evaluation.py"),
            "--list", str(list_file),
            "--gt_dir", str(gt_dir),
            "--logfile", str(logfile),
            "--type", "npy",
            "--num_classes", "81",  # COCO: 80 前景类 + 1 背景类
            "--curve", "True",
            "--predict_dir", str(cam_npy_dir),
            "--comment", f"COCO_{self.split}",
        ]
        
        print(f"\n{'='*60}\n评估 CAM (COCO {self.split})\n{'='*60}")
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 评估失败 (返回码: {e.returncode})")
            return False
    
    def run(self, step=None):
        """运行指定步骤或完整流程"""
        print("检查环境...")
        if not self.check_files():
            print("\n⚠ 警告: 部分必要文件缺失，但将继续运行")
        
        steps = {
            "train": self.train,
            "cam": self.generate_cam,
            "eval": self.evaluate,
        }
        
        if step:
            # 运行单个步骤
            if step not in steps:
                print(f"✗ 未知步骤: {step}")
                return False
            
            return steps[step]()
        else:
            # 运行完整流程
            for name, func in steps.items():
                if not func():
                    print(f"✗ {name} 步骤失败")
                    return False
            
            print(f"\n{'='*60}")
            print("所有步骤完成！")
            print(f"输出目录: {self.output_dir}")
            print(f"TensorBoard: tensorboard --logdir={self.tensorboard_dir}")
            print(f"{'='*60}")
            return True


def main():
    # 生成默认实验ID：exp_MMDDHH（月日时）
    default_exp_id = f"exp_{datetime.now().strftime('%m%d%H')}"
    
    parser = argparse.ArgumentParser(description="DRCFormer COCO 数据集运行脚本")
    
    # 运行模式
    parser.add_argument("--all", action="store_true", help="运行完整流程（训练+CAM+评估）")
    parser.add_argument("--step", choices=["train", "cam", "eval"], help="运行单个步骤")
    
    # 基本配置
    parser.add_argument("--split", choices=["train", "val"], default="train", help="数据集分割 (默认: train)")
    parser.add_argument("--exp_id", default=default_exp_id, help=f"实验ID (默认: {default_exp_id}，格式为exp_MMDDHH)")
    parser.add_argument("--device", default="cuda:0", help="单GPU模式下使用的设备 (默认: cuda:0)")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的GPU数量，>1时启用DDP分布式训练 (默认: 1)")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认: 64)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载进程数 (默认: 16)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (默认: 0)")
    
    # 损失权重
    parser.add_argument("--w_ent", type=float, default=0.01, help="熵损失权重 (默认: 0.01)")
    parser.add_argument("--w_neg", type=float, default=0.01, help="负类损失权重 (默认: 0.01)")
    parser.add_argument("--w_lsp", type=float, default=750, help="超像素对齐损失权重 (默认: 750)")
    
    # CAM 生成选项
    parser.add_argument("--save-png", dest="save_png", action="store_true", help="生成 CAM 时保存 PNG 可视化图片 (默认: 不保存)")
    parser.add_argument("--no-b1", action="store_true", help="禁用 B1 结构监督分支")
    parser.add_argument("--b1-margin", dest="b1_margin", type=float, default=0.15, help="B1分支负类边际阈值 (默认: 0.15)")
    
    # 批量推理选项
    parser.add_argument("--enable-batch-inference", dest="enable_batch_inference", action="store_true",
                       help="启用批量推理功能（CAM生成时使用，默认: 禁用）")
    parser.add_argument("--cam-batch-size", dest="cam_batch_size", type=int, default=8,
                       help="CAM生成的批量大小 (默认: 8)")
    parser.add_argument("--cam-progress-interval", dest="cam_progress_interval", type=int, default=50,
                       help="CAM生成进度日志间隔，每N张图像报告一次 (默认: 50)")
    
    args = parser.parse_args()
    
    # 检查运行模式
    if not args.all and not args.step:
        parser.print_help()
        print("\n✗ 请指定 --all 或 --step 参数")
        sys.exit(1)
    
    # 创建运行器并执行
    runner = COCORunner(args)
    step = None if args.all else args.step
    success = runner.run(step)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
