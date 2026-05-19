#!/usr/bin/env python3
"""
DRCformer 模型训练、CAM生成和评估的 Python 运行脚本

使用方法：
    # ===== 单GPU训练 =====
    # 运行完整流程（默认单GPU）
    python run_drcformer.py --all
    
    # 指定GPU设备
    python run_drcformer.py --all --device cuda:1
    
    # 单独运行某个步骤
    python run_drcformer.py --step train
    python run_drcformer.py --step cam
    python run_drcformer.py --step eval
    
    # ===== 多GPU分布式训练（DDP）=====
    # 使用4张GPU进行训练
    python run_drcformer.py --all --num_gpus 4
    
    # 使用2张GPU，自定义batch size
    python run_drcformer.py --all --num_gpus 2 --batch_size 128
    
    # 仅运行训练步骤（多GPU）
    python run_drcformer.py --step train --num_gpus 4
    
    # ===== 自定义参数 =====
    # 自定义实验ID、epoch和损失权重
    python run_drcformer.py --all --exp_id my_exp --epochs 100 --w_lsp 1000
    
    # 多GPU训练 + 自定义参数
    python run_drcformer.py --all --num_gpus 4 --batch_size 32 --epochs 60
    
    # ===== 指定实验 id 运行第二、三阶段 =====
    # 生成 CAM（默认同时保存 npy 和 png 文件）
    python run_drcformer.py --step cam --exp_id exp_112716
    
    # 生成 CAM 并保存 PNG 可视化图片（已默认启用，此参数保留用于兼容）
    python run_drcformer.py --step cam --exp_id exp_112716 --save_png
    
    # 评估 CAM
    python run_drcformer.py --step eval --exp_id exp_112716

注意事项：
    - 单GPU模式（num_gpus=1）：使用 --device 参数指定设备
    - 多GPU模式（num_gpus>1）：自动使用 torchrun 启动DDP分布式训练
    - CAM生成和评估步骤始终使用单GPU运行
    - 多GPU训练时，有效的全局batch size = batch_size × num_gpus
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


class DRCformerRunner:
    """DRCformer 模型运行器"""
    
    def __init__(self, args):
        # 项目路径配置（见 AGENTS.md: 数据集 ~/projects/voc2012, 预训练 ~/projects/pre-weights, 输出 ~/projects/outputs/drcformer）
        self.base = _get_base_dir()
        self.project_dir = self.base / "drcformer" / "src" / "drcformer"
        
        # 实验配置
        self.exp_id = args.exp_id
        self.device = args.device
        self.num_gpus = args.num_gpus
        
        # 自动处理 CUDA 设备编号
        if args.cuda:
            if "," in args.cuda:
                # 多 GPU 模式
                os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
                self.num_gpus = len(args.cuda.split(","))
                self.device = "cuda"  # DDP 模式下 device 会被 main.py 内部处理
            else:
                # 单 GPU 模式
                self.device = f"cuda:{args.cuda}"
                self.num_gpus = 1
        
        # 数据集路径
        dataset_path = Path(args.dataset_path).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = self.base / dataset_path
        self.dataset_path = dataset_path
        self.img_list_dir = self.dataset_path / "voc12"
        self.train_id_list = self.img_list_dir / "train_aug_id.txt"
        self.val_id_list = self.img_list_dir / "val_id.txt"
        
        # 超像素和预训练权重
        self.sp_idx_dir = self.base / "outputs" / "drcformer" / "sp_idx" / "dino" / "npy"
        pretrained_weight = Path(args.pretrained_weight).expanduser()
        if not pretrained_weight.is_absolute():
            pretrained_weight = self.base / pretrained_weight
        self.pretrained_weight = pretrained_weight
        
        # 输出路径
        self.output_dir = self.base / "outputs" / "drcformer" / self.exp_id
        self.tensorboard_dir = self.base / "outputs" / "drcformer" / "tensorboard" / self.exp_id
        
        # 训练超参数
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.seed = args.seed
        self.num_sp_tokens = getattr(args, 'num_sp_tokens', 196)
        
        # 损失权重
        self.w_ent = args.w_ent
        self.w_neg = args.w_neg
        self.w_lsp = args.w_lsp
        
        # CAM 生成选项
        self.save_png = getattr(args, 'save_png', False)
        self.cam_batch = getattr(args, 'cam_batch', False)
        self.cam_batch_size = getattr(args, 'cam_batch_size', 8)
        self.draw_sp_boundaries = getattr(args, 'draw_sp_boundaries', False)
        
        # CAM 超像素增强参数
        self.cam_enable_amp = getattr(args, 'cam_enable_amp', None)
        self.cam_enable_suppr = getattr(args, 'cam_enable_suppr', None)
        self.cam_sp_gamma = getattr(args, 'cam_sp_gamma', None)
        self.cam_sp_theta = getattr(args, 'cam_sp_theta', None)
        self.cam_amp_alpha = getattr(args, 'cam_amp_alpha', None)
        self.cam_amp_beta = getattr(args, 'cam_amp_beta', None)
    
    def check_files(self):
        """检查必要文件是否存在"""
        checks = [
            (self.pretrained_weight, "预训练权重"),
            (self.dataset_path, "VOC2012数据集"),
            (self.train_id_list, "训练集索引"),
            (self.sp_idx_dir, "超像素索引"),
        ]
        
        for path, name in checks:
            if not path.exists():
                print(f"✗ {name}不存在: {path}")
                return False
            print(f"✓ {name}")
        return True
    
    def run_cmd(self, cmd, desc=""):
        """运行命令"""
        if desc:
            print(f"\n{'='*60}\n{desc}\n{'='*60}")
        try:
            subprocess.run(cmd, cwd=str(self.project_dir), check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 执行失败 (返回码: {e.returncode})")
            return False
    
    def train(self):
        """训练模型"""
        # 构建训练参数（main.py的参数）
        train_args = [
            "--data-path", str(self.dataset_path),
            "--data-set", "VOC12",
            "--img-list", str(self.img_list_dir),
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
            "--b1-enable",
            "--b1-warmup-epochs", "10",
            "--w-ent", str(self.w_ent),
            "--w-neg", str(self.w_neg),
            "--b1-margin", "0.15",
            "--b1-prob-eps", "5e-4",
            "--sp-align-enable",
            "--sp-idx-dir", str(self.sp_idx_dir),
            "--sp-align-apply-to", "all",
            "--sp-align-loss-weight", str(self.w_lsp),
            "--sp-align-start-epoch", "10",
            "--sp-align-warmup-epochs", "5",
            "--sp-overlap-tau", "0.4",
            "--sp-min-area", "48",
            "--num-sp-tokens", str(self.num_sp_tokens),
            "--clip-grad", "1.0",
            "--tensorboard",
            "--tensorboard-dir", str(self.tensorboard_dir),
        ]
        
        # 根据 GPU 数量选择启动方式
        if self.num_gpus > 1:
            # 多GPU模式：使用 torchrun
            cmd = [
                "torchrun",
                f"--nproc_per_node={self.num_gpus}",
                "--master_port=29500",
                "main.py",
            ] + train_args
            desc = f"训练 DRCformer 模型 (实验ID: {self.exp_id}, 多GPU: {self.num_gpus})"
        else:
            # 单GPU模式：使用普通 python
            cmd = ["python", "main.py"] + train_args + ["--device", self.device]
            desc = f"训练 DRCformer 模型 (实验ID: {self.exp_id}, 单GPU: {self.device})"
        
        return self.run_cmd(cmd, desc)
    
    def generate_cam(self):
        """生成 CAM"""
        checkpoint = self.output_dir / "checkpoint.pth"
        if not checkpoint.exists():
            print(f"✗ checkpoint不存在: {checkpoint}")
            return False
        
        cam_png_dir = self.output_dir / "cam" / "png"
        cam_npy_dir = self.output_dir / "cam" / "npy"
        
        cmd = [
            "python", "main.py",
            "--data-set", "VOC12MS",
            "--img-list", str(self.img_list_dir),
            "--data-path", str(self.dataset_path),
            "--attention-dir", str(cam_png_dir),
            "--cam-npy-dir", str(cam_npy_dir),
            "--resume", str(checkpoint),
            "--device", self.device,
            "--num-sp-tokens", str(self.num_sp_tokens),  # 确保 CAM 生成时使用与训练时相同的超像素数量
        ]
        
        # 选择批处理模式或单样本模式
        if self.cam_batch:
            cmd.append("--gen_attention_maps_batch")
            cmd.extend(["--cam_batch_size", str(self.cam_batch_size)])
        else:
            cmd.append("--gen_attention_maps")
        
        # 默认同时保存 npy 和 png 文件
        cmd.append("--save-cam-png")
        
        # 如果启用超像素边界描边
        if self.draw_sp_boundaries:
            cmd.append("--draw-sp-boundaries")
        
        # 传递 CAM 超像素增强参数
        if self.cam_enable_amp is not None:
            if self.cam_enable_amp:
                cmd.append("--cam-enable-amp")
            else:
                cmd.append("--no-cam-enable-amp")
        if self.cam_enable_suppr is not None:
            if self.cam_enable_suppr:
                cmd.append("--cam-enable-suppr")
            else:
                cmd.append("--no-cam-enable-suppr")
        if self.cam_sp_gamma is not None:
            cmd.extend(["--cam-sp-gamma", str(self.cam_sp_gamma)])
        if self.cam_sp_theta is not None:
            cmd.extend(["--cam-sp-theta", str(self.cam_sp_theta)])
        if self.cam_amp_alpha is not None:
            cmd.extend(["--cam-amp-alpha", str(self.cam_amp_alpha)])
        if self.cam_amp_beta is not None:
            cmd.extend(["--cam-amp-beta", str(self.cam_amp_beta)])
        
        mode_desc = "批处理模式" if self.cam_batch else "单样本模式"
        return self.run_cmd(cmd, f"生成 CAM ({mode_desc})")
    
    def evaluate(self):
        """评估 CAM"""
        cam_npy_dir = self.output_dir / "cam" / "npy"
        if not cam_npy_dir.exists() or not any(cam_npy_dir.iterdir()):
            print(f"✗ CAM结果不存在: {cam_npy_dir}")
            return False
        
        # 默认优先使用当前 dataset_path 下的增强标注目录，避免依赖固定路径 ~/projects/voc2012
        gt_candidates = [
            self.dataset_path / "SegmentationClassAug",
            self.dataset_path / "SegmentationClass",
            self.base / "voc2012" / "SegmentationClassAug",
            self.base / "voc2012" / "SegmentationClass",
        ]
        gt_dir = next((p for p in gt_candidates if p.exists()), gt_candidates[0])
        print(f"评估 GT 目录: {gt_dir}")
        logfile = self.output_dir / "evallog_train_aug.txt"
        
        cmd = [
            "python", "evaluation.py",
            "--list", str(self.train_id_list),
            "--gt_dir", str(gt_dir),
            "--logfile", str(logfile),
            "--type", "npy",
            "--curve", "True",
            "--predict_dir", str(cam_npy_dir),
            "--comment", "train_aug",
        ]
        return self.run_cmd(cmd, "评估 CAM")
    
    def run(self, step=None):
        """运行指定步骤或完整流程"""
        print("检查环境...")
        if not self.check_files():
            return False
        
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
    
    parser = argparse.ArgumentParser(description="DRCformer 模型运行脚本")
    
    # 运行模式
    parser.add_argument("--all", action="store_true", help="运行完整流程（训练+CAM+评估）")
    parser.add_argument("--step", choices=["train", "cam", "eval"], help="运行单个步骤")
    
    # 基本配置
    parser.add_argument("--exp_id", default=default_exp_id, help=f"实验ID (默认: {default_exp_id}，格式为exp_MMDDHH)")
    parser.add_argument("--device", default="cuda:0", help="单GPU模式下使用的设备 (默认: cuda:0)")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的GPU数量，>1时启用DDP分布式训练 (默认: 1)")
    parser.add_argument("--cuda", help="指定 CUDA 设备编号 (如: 0 或 0,1)。指定后将自动覆盖 --device 和 --num_gpus")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(Path.home() / "projects" / "datasets"),
        help="VOC 数据集路径（默认: ~/projects/datasets）",
    )
    parser.add_argument(
        "--pretrained_weight",
        type=str,
        default=str(Path.home() / "projects" / "pre-weights" / "suit_base_224.pth"),
        help="预训练权重路径（默认: ~/projects/pre-weights/suit_base_224.pth）",
    )
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认: 64)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载进程数 (默认: 16)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子 (默认: 0)")
    parser.add_argument("--num_sp_tokens", type=int, default=196, help="超像素 token 数量 (默认: 196)")
    
    # 损失权重
    parser.add_argument("--w_ent", type=float, default=0.01, help="熵损失权重 (默认: 0.01)")
    parser.add_argument("--w_neg", type=float, default=0.01, help="负类损失权重 (默认: 0.01)")
    parser.add_argument("--w_lsp", type=float, default=750, help="超像素对齐损失权重 (默认: 750)")
    
    # CAM 生成选项
    parser.add_argument("--save_png", action="store_true", help="生成 CAM 时保存 PNG 可视化图片 (已默认启用，此参数保留用于兼容)")
    parser.add_argument("--cam_batch", action="store_true", help="使用批处理模式生成 CAM（按尺寸分组，提升速度）")
    parser.add_argument("--cam_batch_size", type=int, default=8, help="CAM 批处理大小 (默认: 8)")
    parser.add_argument("--draw_sp_boundaries", action="store_true", help="在 CAM PNG 图片中为超像素标记边界")
    
    # CAM 超像素增强参数
    parser.add_argument("--cam_enable_amp", dest="cam_enable_amp", action="store_true", help="启用 CAM 增强（超像素重叠比例 >= gamma 时）")
    parser.add_argument("--no_cam_enable_amp", dest="cam_enable_amp", action="store_false", help="禁用 CAM 增强")
    parser.add_argument("--cam_enable_suppr", dest="cam_enable_suppr", action="store_true", help="启用 CAM 抑制（超像素重叠比例 <= theta 时）")
    parser.add_argument("--no_cam_enable_suppr", dest="cam_enable_suppr", action="store_false", help="禁用 CAM 抑制")
    # 设置默认值为 None，这样可以通过检查是否为 None 来判断用户是否指定了参数
    parser.set_defaults(cam_enable_amp=None, cam_enable_suppr=None)
    parser.add_argument("--cam_sp_gamma", type=float, default=None, help="重叠上阈值，用于判断是否增强 (默认: 0.45)")
    parser.add_argument("--cam_sp_theta", type=float, default=None, help="重叠下阈值，用于判断是否抑制 (默认: 0.38)")
    parser.add_argument("--cam_amp_alpha", type=float, default=None, help="增强系数，增强时乘以 (1 + alpha) (默认: 0.06)")
    parser.add_argument("--cam_amp_beta", type=float, default=None, help="抑制系数，抑制时乘以 (1 - beta) (默认: 0.08)")
    
    args = parser.parse_args()
    
    # 检查运行模式
    if not args.all and not args.step:
        parser.print_help()
        print("\n✗ 请指定 --all 或 --step 参数")
        sys.exit(1)
    
    # 创建运行器并执行
    runner = DRCformerRunner(args)
    step = None if args.all else args.step
    success = runner.run(step)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
