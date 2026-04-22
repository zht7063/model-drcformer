"""
PSA 全流程控制脚本 - Python 版本

功能：实现 PSA (Propagation via Semantic Affinity) 完整流程
    步骤1: 生成多尺度 CAM (Class Activation Maps)
    步骤2: 训练 PSA 后处理模型 (基于 ResNet38)
    步骤3: PSA 推理生成精细化伪标签
    步骤4: 评估伪标签质量 (mIoU)

快速开始：
    # 注意：需要在项目根目录（~/projects）下运行，见 AGENTS.md
    
    # 1. 执行完整流程（4个步骤），使用相对路径指定权重
    python src/drcformer/run_psa.py --checkpoint outputs/drcformer/exp_112716/best_checkpoint.pth
    
    # 2. 只执行步骤1：生成 CAM
    python src/drcformer/run_psa.py --steps 1 --checkpoint outputs/drcformer/exp_112716/best_checkpoint.pth
    
    # 3. 从步骤2开始，保留已有 CAM
    python src/drcformer/run_psa.py --steps 2 3 4 --no-prepare-dirs
    
    # 4. 保存 CAM PNG 可视化文件（调试时使用）
    python src/drcformer/run_psa.py --save-cam-png --checkpoint outputs/drcformer/exp_112716/best_checkpoint.pth
    
    # 5. 不保存伪标签 PNG（默认保存，使用此选项可节省空间）
    python src/drcformer/run_psa.py --no-save-pseudo-png --checkpoint outputs/drcformer/exp_112716/best_checkpoint.pth
    
    # 6. 使用自定义权重路径（支持绝对路径）
    python src/drcformer/run_psa.py \\
        --checkpoint /absolute/path/to/model.pth \\
        --psa-weights /absolute/path/to/resnet38.pth

核心参数：
    --base-dir          项目根目录（默认: ~/projects，见 AGENTS.md）
    --checkpoint        DRCformer 模型权重（支持相对路径和绝对路径）
                        相对路径相对于 base-dir，默认: outputs/drcformer/exp1013/best_checkpoint.pth
                        别名: --checkpoint-path（兼容旧参数）
    --psa-weights       ResNet38 预训练权重（支持相对路径和绝对路径）
                        相对路径相对于 base-dir，默认: pre-weights/res38_cls.pth
                        别名: --resnet-weight-path（兼容旧参数）
    --device            运行设备（默认: cuda:1，可选: cuda:0, cuda:1, cpu）
    --num-workers       数据加载并行进程数（默认: 16，可根据 CPU 核心数调整）
    --cam-batch-size    CAM 生成批处理大小（默认: 8，可根据 GPU 内存调整到 4-16）
    --psa-batch-size    PSA 训练批处理大小（默认: 16，可根据 GPU 内存调整到 8-32）

流程控制：
    --steps             执行步骤（默认: 1 2 3 4，可选: 1-4 任意组合）
                        示例: --steps 1（只生成 CAM）
                             --steps 2 3（训练 PSA + 推理）
                             --steps 4（只评估）
    --no-prepare-dirs   保留已有输出目录（默认会清空，节省空间）
    --force             强制执行（忽略前置步骤检查，不推荐）

可视化控制：
    --save-cam-png         保存 CAM 的 PNG 可视化（默认只保存 NPY，节省空间）
    --no-save-pseudo-png   不保存伪标签 PNG 文件（默认保存 PNG）

输出结构：{BASE_DIR}/outputs/psa/
    attn-patchrefine/               - CAM PNG 可视化（需 --save-cam-png）
    attn-patchrefine-npy/           - CAM 原始数据（NPY 格式）
    attn-patchrefine-npy-crf_1/     - CRF 精炼 CAM（低温度，用于训练 PSA）
    attn-patchrefine-npy-crf_12/    - CRF 精炼 CAM（高温度，用于训练 PSA）
    pgt-psa-rw/                     - 最终伪标签（PNG，默认保存）
    resnet38_aff.pth                - 训练好的 PSA 模型权重
    resnet38_aff.log                - PSA 训练日志

实用技巧：
    • 默认保存伪标签 PNG 文件，使用 --no-save-pseudo-png 可禁用保存以节省空间
    • 使用 --steps 可灵活重新执行某些步骤
    • --no-prepare-dirs 配合 --steps 可在已有输出基础上重跑
    • 步骤间有依赖：2 依赖 1，3 依赖 1+2，4 依赖 3

注意事项：
    • 必须在项目根目录（默认 ~/projects）下运行此脚本，见 AGENTS.md
    • 运行前需先训练 DRCformer 主模型（步骤1需要）
    • ResNet38 权重需提前下载（步骤2需要）
      下载: https://github.com/jiwoon-ahn/psa/releases/download/v1.0/resnet38_cls.pth
    • 支持相对路径和绝对路径，相对路径相对于 base-dir
    • 所有命令会在 src/drcformer/ 目录下执行
    • 使用 --help 查看完整参数说明
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


class PSARunner:
    """PSA 流程控制类，封装从 CAM 生成到伪标签验证的完整流程"""
    
    def __init__(self, args):
        """
        初始化 PSA Runner
        
        Args:
            args: 命令行参数对象
        """
        self.args = args
        
        # 路径配置（见 AGENTS.md: 数据集 ~/projects/voc2012, 输出 ~/projects/outputs）
        self.base_dir = args.base_dir
        self.dataset_path = os.path.join(self.base_dir, "voc2012")
        self.img_list_dir = os.path.join(self.base_dir, "voc2012/voc12")
        self.psa_output_dir = os.path.join(self.base_dir, "outputs/psa")
        
        # 工作目录：drcformer 目录（所有命令都在此目录下执行）
        self.work_dir = os.path.join(self.base_dir, "drcformer/src/drcformer")
        
        # 权重路径
        self.checkpoint_path = args.checkpoint_path
        self.resnet_weight_path = args.resnet_weight_path
        
        # 输出子目录
        self.attn_dir = os.path.join(self.psa_output_dir, "attn-patchrefine")
        self.attn_npy_dir = os.path.join(self.psa_output_dir, "attn-patchrefine-npy")
        self.attn_crf_base = os.path.join(self.psa_output_dir, "attn-patchrefine-npy-crf")
        self.attn_crf_1 = os.path.join(self.psa_output_dir, "attn-patchrefine-npy-crf_1")
        self.attn_crf_12 = os.path.join(self.psa_output_dir, "attn-patchrefine-npy-crf_12")
        self.pgt_psa_rw = os.path.join(self.psa_output_dir, "pgt-psa-rw")
        
        # 训练和验证列表
        self.train_list = os.path.join(self.base_dir, "voc2012/voc12/train_aug_id.txt")
        self.val_list = os.path.join(self.base_dir, "voc2012/voc12/val_id.txt")
        
        # PSA 模型输出路径
        self.psa_model_path = os.path.join(self.psa_output_dir, "resnet38_aff.pth")
        
        # 性能配置
        self.num_workers = getattr(args, 'num_workers', 16)
        self.cam_batch_size = getattr(args, 'cam_batch_size', 8)
        self.psa_batch_size = getattr(args, 'psa_batch_size', 16)
    
    def check_prerequisites(self):
        """检查必要的预训练权重文件是否存在"""
        print("=" * 60)
        print("检查必要的权重文件...")
        print("=" * 60)
        
        # 检查 DRCformer 模型权重
        if not os.path.isfile(self.checkpoint_path):
            print(f"错误: 找不到 DRCformer 模型权重文件: {self.checkpoint_path}")
            print("请先运行 run_drcformer.sh 训练 DRCformer 模型")
            sys.exit(1)
        else:
            print(f"✓ DRCformer 模型权重: {self.checkpoint_path}")
        
        # 检查 ResNet38 预训练权重
        if not os.path.isfile(self.resnet_weight_path):
            print(f"错误: 找不到 ResNet38 预训练权重文件: {self.resnet_weight_path}")
            print("请下载 ResNet38 预训练权重文件并放置在指定路径")
            print("下载链接: https://github.com/jiwoon-ahn/psa/releases/download/v1.0/resnet38_cls.pth")
            sys.exit(1)
        else:
            print(f"✓ ResNet38 预训练权重: {self.resnet_weight_path}")
        
        print()
    
    def prepare_directories(self, clean=True):
        """
        准备输出目录
        
        Args:
            clean: 是否清空已有的输出目录
        """
        print("=" * 60)
        print("准备输出目录...")
        print("=" * 60)
        
        # 清空旧的输出目录
        if clean and os.path.exists(self.psa_output_dir):
            print(f"清空旧的输出目录: {self.psa_output_dir}")
            shutil.rmtree(self.psa_output_dir)
        elif not clean and os.path.exists(self.psa_output_dir):
            print(f"保留已有的输出目录: {self.psa_output_dir}")
        
        # 创建必要的输出目录
        dirs_to_create = [
            self.psa_output_dir,
            self.attn_dir,
            self.attn_npy_dir,
            self.attn_crf_1,
            self.attn_crf_12,
            self.pgt_psa_rw
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            if clean:
                print(f"✓ 创建目录: {dir_path}")
            else:
                print(f"✓ 确保目录存在: {dir_path}")
        
        print()
    
    def generate_attention_maps(self):
        """步骤1: 调用主模型进行推理，生成不同尺度的 CAM 图"""
        print("=" * 60)
        print("步骤 1/4: 生成 Attention Maps (CAM)")
        print("=" * 60)
        
        cmd = [
            "python", "main.py",
            "--data-set", "VOC12MS",
            "--img-list", self.img_list_dir,
            "--data-path", self.dataset_path,
            "--resume", self.checkpoint_path,
            "--gen_attention_maps_batch",
            "--attention-type", "fused",
            "--attention-dir", self.attn_dir,
            "--cam-npy-dir", self.attn_npy_dir,
            "--out-crf", self.attn_crf_base,
            "--crf-save-temps", "1", "12",
            "--seed-bg-thr", "0.39",
            "--no-cam-crf",
            "--device", self.args.device,
            "--num_workers", str(self.num_workers),
            "--cam_batch_size", str(self.cam_batch_size)
        ]
        
        # 如果指定了保存 CAM PNG，则添加参数
        if getattr(self.args, 'save_cam_png', False):
            cmd.append("--save-cam-png")
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {self.work_dir}")
        print()
        
        result = subprocess.run(cmd, cwd=self.work_dir, check=True)
        
        print()
        print("✓ Attention Maps 生成完成")
        print()
    
    def train_psa(self):
        """步骤2: 训练 PSA 模型"""
        print("=" * 60)
        print("步骤 2/4: 训练 PSA 模型")
        print("=" * 60)
        
        cmd = [
            "python", "psa/train_aff.py",
            "--weights", self.resnet_weight_path,
            "--voc12_root", self.dataset_path,
            "--la_crf_dir", self.attn_crf_1,
            "--ha_crf_dir", self.attn_crf_12,
            "--train_list", self.train_list,
            "--val_list", self.val_list,
            "--session_name", os.path.join(self.psa_output_dir, "resnet38_aff"),
            "--num_workers", str(self.num_workers),
            "--batch_size", str(self.psa_batch_size)
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {self.work_dir}")
        print()
        
        result = subprocess.run(cmd, cwd=self.work_dir, check=True)
        
        print()
        print("✓ PSA 模型训练完成")
        print()
    
    def infer_psa(self):
        """步骤3: 使用 PSA 进行推理，生成伪标签"""
        print("=" * 60)
        print("步骤 3/4: 使用 PSA 进行推理")
        print("=" * 60)
        
        cmd = [
            "python", "psa/infer_aff.py",
            "--weights", self.psa_model_path,
            "--infer_list", self.train_list,
            "--cam_dir", self.attn_npy_dir,
            "--voc12_root", self.dataset_path,
            "--out_rw", self.pgt_psa_rw,
            "--num_workers", str(self.num_workers)
        ]
        
        # 默认保存伪标签 PNG，除非指定了 --no-save-pseudo-png
        if getattr(self.args, 'no_save_pseudo_png', False):
            cmd.append("--no_pseudo_png")
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {self.work_dir}")
        print()
        
        result = subprocess.run(cmd, cwd=self.work_dir, check=True)
        
        print()
        print("✓ PSA 推理完成")
        print()
    
    def evaluate_pseudo_labels(self):
        """步骤4: 验证 PSA 生成的伪标签质量"""
        print("=" * 60)
        print("步骤 4/4: 验证伪标签质量")
        print("=" * 60)
        
        # 检查预测目录中是否存在 PNG 文件
        if os.path.exists(self.pgt_psa_rw):
            png_files = [f for f in os.listdir(self.pgt_psa_rw) if f.endswith('.png')]
            if len(png_files) == 0:
                print(f"\n警告：预测目录中未找到 PNG 文件: {self.pgt_psa_rw}")
                print("可能的原因：PSA 推理步骤未保存 PNG 文件（可能使用了 --no-save-pseudo-png）")
                print("解决方案：重新运行步骤 3（PSA 推理），确保未使用 --no-save-pseudo-png 参数")
                print("例如：python src/drcformer/run_psa.py --steps 3")
                print()
                raise FileNotFoundError(
                    f"未找到预测 PNG 文件。请重新运行 PSA 推理步骤（确保未使用 --no-save-pseudo-png）。"
                )
            else:
                print(f"✓ 找到 {len(png_files)} 个 PNG 预测文件")
        else:
            print(f"\n错误：预测目录不存在: {self.pgt_psa_rw}")
            raise FileNotFoundError(f"预测目录不存在: {self.pgt_psa_rw}")
        
        cmd = [
            "python", "evaluation.py",
            "--list", self.train_list,
            "--gt_dir", os.path.join(self.dataset_path, "SegmentationClass"),
            "--logfile", os.path.join(self.pgt_psa_rw, "evallog.txt"),
            "--type", "png",
            "--predict_dir", self.pgt_psa_rw,
            "--comment", "train_aug",
            "--num_workers", str(self.num_workers * 4)  # 评估是 CPU 密集型，可以使用更多进程
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"工作目录: {self.work_dir}")
        print()
        
        result = subprocess.run(cmd, cwd=self.work_dir, check=True)
        
        print()
        print("✓ 伪标签质量验证完成")
        print()
    
    def cleanup(self):
        """清理临时文件"""
        print("=" * 60)
        print("清理临时文件...")
        print("=" * 60)
        
        # 删除 saved_model 目录（在工作目录下）
        saved_model_dir = os.path.join(self.work_dir, "saved_model")
        if os.path.exists(saved_model_dir):
            print(f"删除目录: {saved_model_dir}")
            shutil.rmtree(saved_model_dir)
        
        print()
    
    def check_step_prerequisites(self, step):
        """
        检查执行某个步骤所需的前置条件
        
        Args:
            step: 步骤编号
        """
        if step == 2:
            # 步骤2需要步骤1的输出
            if not os.path.exists(self.attn_crf_1) or not os.path.exists(self.attn_crf_12):
                print(f"⚠ 警告: 步骤 2 需要步骤 1 的输出（CRF 精炼后的 CAM）")
                print(f"    请确保目录存在: {self.attn_crf_1}")
                print(f"    请确保目录存在: {self.attn_crf_12}")
                if not self.args.force:
                    sys.exit(1)
        elif step == 3:
            # 步骤3需要步骤1和步骤2的输出
            if not os.path.exists(self.psa_model_path):
                print(f"⚠ 警告: 步骤 3 需要步骤 2 的输出（PSA 模型权重）")
                print(f"    请确保文件存在: {self.psa_model_path}")
                if not self.args.force:
                    sys.exit(1)
            if not os.path.exists(self.attn_npy_dir):
                print(f"⚠ 警告: 步骤 3 需要步骤 1 的输出（CAM NPY 文件）")
                print(f"    请确保目录存在: {self.attn_npy_dir}")
                if not self.args.force:
                    sys.exit(1)
        elif step == 4:
            # 步骤4需要步骤3的输出
            if not os.path.exists(self.pgt_psa_rw):
                print(f"⚠ 警告: 步骤 4 需要步骤 3 的输出（伪标签）")
                print(f"    请确保目录存在: {self.pgt_psa_rw}")
                if not self.args.force:
                    sys.exit(1)
    
    def run(self):
        """运行完整的 PSA 流程"""
        print("\n")
        print("=" * 60)
        print("PSA 全流程控制程序")
        print("=" * 60)
        
        # 显示将要执行的步骤
        steps = self.args.steps
        step_names = {
            1: "生成 Attention Maps (CAM)",
            2: "训练 PSA 模型",
            3: "使用 PSA 进行推理",
            4: "评估伪标签质量"
        }
        
        print(f"将执行以下步骤: {', '.join([str(s) for s in steps])}")
        for step in steps:
            print(f"  步骤 {step}: {step_names.get(step, '未知步骤')}")
        print()
        
        try:
            # 检查前置条件
            self.check_prerequisites()
            
            # 准备目录（根据参数决定是否清空）
            if not self.args.no_prepare_dirs:
                self.prepare_directories(clean=True)
            else:
                self.prepare_directories(clean=False)
            
            # 执行选定的步骤
            if 1 in steps:
                self.generate_attention_maps()
            else:
                print(f"⊘ 跳过步骤 1: {step_names[1]}")
                self.check_step_prerequisites(1)
            
            if 2 in steps:
                self.check_step_prerequisites(2)
                self.train_psa()
            else:
                print(f"⊘ 跳过步骤 2: {step_names[2]}")
            
            if 3 in steps:
                self.check_step_prerequisites(3)
                self.infer_psa()
            else:
                print(f"⊘ 跳过步骤 3: {step_names[3]}")
            
            if 4 in steps:
                self.check_step_prerequisites(4)
                self.evaluate_pseudo_labels()
            else:
                print(f"⊘ 跳过步骤 4: {step_names[4]}")
            
            # 清理
            self.cleanup()
            
            print("=" * 60)
            print(f"✓ 选定步骤执行完成！")
            print("=" * 60)
            print(f"伪标签输出目录: {self.pgt_psa_rw}")
            if 4 in steps:
                print(f"评估日志: {os.path.join(self.pgt_psa_rw, 'evallog.txt')}")
            print()
            
        except subprocess.CalledProcessError as e:
            print()
            print("=" * 60)
            print(f"✗ 执行失败: {e}")
            print("=" * 60)
            sys.exit(1)
        except KeyboardInterrupt:
            print()
            print("=" * 60)
            print("✗ 用户中断执行")
            print("=" * 60)
            sys.exit(1)


def get_args_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="PSA 全流程控制程序 - 从 CAM 生成到伪标签验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础路径配置
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path.home() / "projects"),
        help="项目根目录路径（见 AGENTS.md，默认 ~/projects）"
    )
    
    # 权重路径
    parser.add_argument(
        "--checkpoint", "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=None,
        help="DRCformer 模型权重路径（支持相对/绝对路径，相对路径相对于 base-dir，默认: outputs/drcformer/exp1013/best_checkpoint.pth）"
    )
    
    parser.add_argument(
        "--psa-weights", "--resnet-weight-path",
        dest="resnet_weight_path",
        type=str,
        default="pre-weights/res38_cls.pth",
        help="ResNet38 预训练权重路径（支持相对/绝对路径，相对路径相对于 base-dir，默认: pre-weights/res38_cls.pth）"
    )
    
    # 设备配置
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="训练/推理使用的设备"
    )
    
    # 数据加载配置
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="数据加载的并行进程数（默认: 16，可根据 CPU 核心数调整）"
    )
    
    # 性能优化配置
    parser.add_argument(
        "--cam-batch-size",
        type=int,
        default=8,
        help="CAM 生成的批处理大小（默认: 8，范围: 4-16，根据 GPU 内存调整）"
    )
    
    parser.add_argument(
        "--psa-batch-size",
        type=int,
        default=16,
        help="PSA 训练的批处理大小（默认: 16，可根据 GPU 内存调整到 8-32）"
    )
    
    # 步骤控制
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="指定要执行的步骤编号 (1:生成CAM, 2:训练PSA, 3:PSA推理, 4:评估)"
    )
    
    # 目录准备控制
    parser.add_argument(
        "--no-prepare-dirs",
        action="store_true",
        help="跳过目录准备阶段（不清空已有输出）"
    )
    
    # 强制执行（即使缺少前置输出）
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制执行（即使缺少前置步骤的输出，也会尝试执行）"
    )
    
    # PNG 文件保存控制
    parser.add_argument(
        "--save-cam-png",
        action="store_true",
        help="保存 CAM 的 PNG 可视化文件（默认只保存 NPY 格式）"
    )
    
    parser.add_argument(
        "--no-save-pseudo-png",
        dest="no_save_pseudo_png",
        action="store_true",
        help="不保存伪标签的 PNG 文件（默认保存 PNG）"
    )
    
    return parser


def clean_sys_argv():
    """清理 sys.argv 中的不可见字符（主要是 \\xa0 不间断空格）"""
    # 清理并过滤空参数
    cleaned = [sys.argv[0]]  # 保留脚本名
    for arg in sys.argv[1:]:
        # 替换 \xa0 为普通空格，并清理首尾空白
        cleaned_arg = arg.replace('\xa0', ' ').strip()
        if cleaned_arg:  # 只保留非空参数
            cleaned.append(cleaned_arg)
    sys.argv = cleaned


def main():
    """主函数"""
    # 清理 sys.argv 中的不可见字符（在参数解析之前）
    clean_sys_argv()
    
    parser = get_args_parser()
    
    # 使用 parse_known_args 来检测无法识别的参数
    try:
        args, unknown_args = parser.parse_known_args()
        
        # 如果存在无法识别的参数，输出错误信息
        if unknown_args:
            # 检查是否包含不可见字符
            has_invisible = any('\xa0' in str(arg) for arg in unknown_args)
            
            print("=" * 60)
            print("错误: 检测到无法识别的参数")
            print("=" * 60)
            print(f"无法识别的参数: {unknown_args}")
            if has_invisible:
                print("\n提示: 检测到不可见字符 (\\xa0)，可能是复制粘贴导致的。")
                print("建议: 重新手动输入命令，或使用脚本文件。")
            print("\n使用 --help 查看所有可用参数:")
            print("  python src/drcformer/run_psa.py --help")
            print("=" * 60)
            sys.exit(1)
    except SystemExit as e:
        if e.code != 0:
            sys.exit(1)
        raise
    
    # 设置默认权重路径（如果未指定）
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(
            args.base_dir, "outputs/drcformer/exp1013/best_checkpoint.pth"
        )
    # 处理 DRCformer 权重路径（如果是相对路径，则相对于 base_dir）
    elif not os.path.isabs(args.checkpoint_path):
        args.checkpoint_path = os.path.join(
            args.base_dir, args.checkpoint_path
        )
    
    # 处理 PSA 权重路径（如果是相对路径，则相对于 base_dir）
    if not os.path.isabs(args.resnet_weight_path):
        args.resnet_weight_path = os.path.join(
            args.base_dir, args.resnet_weight_path
        )
    
    # 创建 PSA Runner 并执行
    runner = PSARunner(args)
    runner.run()


if __name__ == "__main__":
    main()

