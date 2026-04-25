import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import json

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets import build_dataset
from engine import train_one_epoch, evaluate, generate_attention_maps_ms, generate_attention_maps_ms_batch

import models
import utils
import os
import numpy as np
import random

from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        # default="deit_small_MCTformerPlus",
        default="drcformer",  # 默认创建一个 base 模型。
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--num-sp-tokens",
        type=int,
        default=196,
        help="Number of superpixel tokens (default: 196)"
    )

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # Dataset parameters
    parser.add_argument(
        "--data-path", default="VOCdevkit/VOC2012", type=str, help="dataset path"
    )
    parser.add_argument("--img-list", default="voc12", type=str, help="image list path")
    parser.add_argument(
        "--data-set",
        default="VOC12",
        type=str,
        help="dataset, take voc2012 aug as default.",
    )
    parser.add_argument(
        "--coco-split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="COCO dataset split: train or val (default: train)"
    )

    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "projects" / "outputs" / "drcformer" / "default"),
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument("--gen_attention_maps", default=False, action="store_true")
    parser.add_argument("--gen_attention_maps_batch", default=False, action="store_true",
                        help="Generate attention maps with batch processing (grouped by size)")
    parser.add_argument("--cam_batch_size", type=int, default=8,
                        help="Batch size for CAM generation (default: 8)")
    parser.add_argument("--cam-progress-interval", type=int, default=50,
                        help="Progress log interval for CAM generation, report every N images (default: 50)")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--attention-dir", type=str, default="cam-png")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=12,
        help="extract attention maps from the last layers",
    )

    parser.add_argument("--patch-attn-refine", type=bool, default=True)
    parser.add_argument("--visualize-cls-attn", type=bool, default=True)
    parser.add_argument("--save-cam-png", default=True, action="store_true",
                        help="Save CAM visualization as PNG images (default: True, save both npy and png)")
    parser.add_argument("--draw-sp-boundaries", default=False, action="store_true",
                        help="Draw superpixel boundaries on CAM PNG images")

    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--cam-npy-dir", type=str, default="cam-npy")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.75, 1.25])
    parser.add_argument("--label-file-path", type=str, default=None)
    parser.add_argument("--attention-type", type=str, default="fused")

    # CAM CRF refinement parameters
    parser.add_argument("--crf-iters", dest="crf_iters", type=int, default=5,
                        help="Number of CRF inference iterations for CAM refinement (default: 5)")
    parser.add_argument("--cam-crf", dest="cam_crf", action="store_true",
                        help="Enable CRF refinement for CAMs")
    parser.add_argument("--no-cam-crf", dest="cam_crf", action="store_false",
                        help="Disable CRF refinement for CAMs")
    parser.set_defaults(cam_crf=False)

    # CAM superpixel enhancement parameters
    parser.add_argument("--cam-enable-amp", dest="cam_enable_amp", action="store_true",
                        help="Enable CAM enhancement for superpixels with high overlap ratio (default: True)")
    parser.add_argument("--no-cam-enable-amp", dest="cam_enable_amp", action="store_false",
                        help="Disable CAM enhancement for superpixels with high overlap ratio")
    parser.set_defaults(cam_enable_amp=True)
    
    parser.add_argument("--cam-enable-suppr", dest="cam_enable_suppr", action="store_true",
                        help="Enable CAM suppression for superpixels with low overlap ratio (default: True)")
    parser.add_argument("--no-cam-enable-suppr", dest="cam_enable_suppr", action="store_false",
                        help="Disable CAM suppression for superpixels with low overlap ratio")
    parser.set_defaults(cam_enable_suppr=True)
    
    parser.add_argument("--cam-sp-gamma", dest="cam_sp_gamma", type=float, default=0.45,
                        help="Upper threshold for overlap ratio to enable enhancement (default: 0.45)")
    parser.add_argument("--cam-sp-theta", dest="cam_sp_theta", type=float, default=0.38,
                        help="Lower threshold for overlap ratio to enable suppression (default: 0.38)")
    parser.add_argument("--cam-amp-alpha", dest="cam_amp_alpha", type=float, default=0.06,
                        help="Enhancement coefficient: multiply by (1 + alpha) when ratio >= gamma (default: 0.06)")
    parser.add_argument("--cam-amp-beta", dest="cam_amp_beta", type=float, default=0.08,
                        help="Suppression coefficient: multiply by (1 - beta) when ratio <= theta (default: 0.08)")

    # Export CRF-refined CAM seeds for PSA (LA/HA)
    parser.add_argument("--out-crf", type=str, default=None,
                        help="Base directory to save CRF-refined CAM seeds. Will create subdirs with temperature suffix, e.g., _1, _12")
    parser.add_argument("--crf-save-temps", nargs="+", type=float, default=[1.0, 12.0],
                        help="Temperature values used when exporting CRF seeds. Directories will be suffixed by _{int(temp)}.")
    parser.add_argument("--seed-bg-thr", type=float, default=0.39,
                        help="Background score used as channel 0 for exported CRF seeds (LA/HA)")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--loss-weight", default=1.0, type=float)
    parser.add_argument("--num-cct", default=12, type=int)
    parser.add_argument("--cct-temp", type=float, default=1.0,
                    help="Temperature τ for CCT logits; lower=sharper (e.g., 0.2~1.0).")

    # TensorBoard parameters
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default="logs", help="TensorBoard log directory"
    )

    # 权重参数
    parser.add_argument("--mct-loss-weight", default=1.0, type=float, help="Weight for main MCT loss (default: 1.0)")
    parser.add_argument("--attn-loss-weight", default=1.0, type=float, help="Weight for attention regularization loss (default: 1.0)")
    parser.add_argument("--patch-loss-weight", default=1.0, type=float, help="Weight for patch loss (default: 1.0)")
    # 阶段性调整权重参数
    # 开关：到达指定 epoch 是否切换到一组新权重
    parser.add_argument("--loss-swap-enable", action="store_true", help="Enable loss weight swap at a given epoch")
    parser.add_argument("--loss-swap-epoch", type=int, default=0, help="Epoch (inclusive) to start using new loss weights")
    parser.add_argument("--loss-swap-warm-epochs", type=int, default=0, help="Warm-up epochs to smoothly interpolate base->target loss weights after swap epoch (0 means instant switch)")
    # 新权重（不写就默认沿用老权重）
    parser.add_argument("--new-mct-loss-weight",   type=float, default=None)
    parser.add_argument("--new-attn-loss-weight",  type=float, default=None)
    parser.add_argument("--new-patch-loss-weight", type=float, default=None)

    # ---- B1 structural supervision (token-level) ----
    parser.add_argument("--b1-enable", action="store_true", help="Enable B1 token-level structural supervision (default: disabled)")
    parser.add_argument("--w-ent", type=float, default=0.03, help="Weight for positive-class entropy term in B1 (default: 0.05)")
    parser.add_argument("--w-neg", type=float, default=0.01, help="Weight for negative-class margin term in B1 (default: 0.02)")
    parser.add_argument("--w-var", type=float, default=0.0, help="Weight for variance term in B1 (default: 0.0; disabled initially)")
    parser.add_argument("--b1-prob-eps", type=float, default=1e-4, help="Numerical epsilon for probability clamping in B1 (default: 1e-6)")
    parser.add_argument("--b1-margin", type=float, default=0.15, help="Margin m for negative-class hinge in B1 (default: 0.1)")
    parser.add_argument("--b1-warmup-epochs", type=int, default=3, help="Warmup epochs for scaling B1 loss from 0->1 (default: 1)")
    parser.add_argument("--b1-use-cams", action="store_true", help="Use CAMs as source for B1 instead of token head (default: False, not recommended initially)")

    # ---- L_sp superpixel alignment loss (region consistency) ----
    parser.add_argument("--sp-align-enable", action="store_true", help="Enable L_sp superpixel alignment loss (default: disabled)")
    parser.add_argument("--sp-align-loss-weight", type=float, default=0.0, help="Weight for superpixel alignment loss (default: 0.0)")
    parser.add_argument("--sp-align-apply-to", type=str, default="pos", choices=["pos", "all"], help="Apply loss to positive classes only or all classes (default: pos)")
    parser.add_argument("--sp-align-warmup-epochs", type=int, default=0, help="Number of warmup epochs for sp alignment loss (default: 0)")
    parser.add_argument("--sp-align-start-epoch", type=int, default=0, help="Epoch to start applying sp alignment loss (default: 0)")
    parser.add_argument("--sp-idx-dir", type=str, default="assets/sp_idx/npy", help="Directory containing DINO superpixel index files (default: assets/sp_idx/npy)")
    parser.add_argument("--sp-overlap-tau", type=float, default=0.4, help="Overlap threshold for SLIC->DINO assignment (default: 0.4)")
    parser.add_argument("--sp-min-area", type=int, default=0, help="Minimum area for DINO regions to be considered (default: 0)")

    # ---- 分布式训练参数 ----
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank of current process")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main(args):

    logger.info(f"args: {args}")

    # 路径解析：确保相对路径的输出目录位于 output_dir 之下
    output_dir = Path(args.output_dir)
    if args.attention_dir and not os.path.isabs(args.attention_dir):
        args.attention_dir = str(output_dir / args.attention_dir)
    if args.cam_npy_dir and not os.path.isabs(args.cam_npy_dir):
        args.cam_npy_dir = str(output_dir / args.cam_npy_dir)
    if args.out_crf and not os.path.isabs(args.out_crf):
        args.out_crf = str(output_dir / args.out_crf)
    if args.tensorboard_dir and not os.path.isabs(args.tensorboard_dir):
        args.tensorboard_dir = str(output_dir / args.tensorboard_dir)

    # 初始化分布式训练环境
    utils.init_distributed_mode(args)

    # 根据分布式状态设置设备
    if hasattr(args, 'gpu') and args.gpu is not None:
        # 分布式模式：使用 init_distributed_mode 设置的 GPU
        device = torch.device(f"cuda:{args.gpu}")
    else:
        # 单GPU模式：使用原有的 device 参数
        device = torch.device(args.device)
        # 显式设定当前CUDA设备，避免使用非默认GPU（如 cuda:1）时，部分CUDA扩展/内核仍在当前设备（默认为0）上执行
        if device.type == "cuda":
            # 如果未指定具体索引（如仅为 'cuda'），默认使用 cuda:0，避免 torch.cuda.set_device 报错
            if getattr(device, "index", None) is None:
                device = torch.device("cuda:0")
            torch.cuda.set_device(device)
    
    try:
        if device.type == "cuda":
            logger.info(
                f"当前CUDA设备: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}"
            )
    except Exception:
        pass

    # same_seeds(0)
    seed = args.seed
    # 对训练保守设置；在 CAM 生成时我们会单独打开高性能开关
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # 若仅用于生成注意力图，打开高性能后端（不影响数值正确性）
    if getattr(args, "gen_attention_maps", False):
        try:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 高阶 TF32 性能选项（PyTorch 2.0+）
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            logger.info("已为 CAM 生成开启 cudnn.benchmark / TF32 / 高性能 matmul 精度")
        except Exception as _e:
            logger.warning(f"开启高性能后端失败: {_e}")

    # 三个数据集，分别用于：训练模型、验证模型、生成注意力图
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_train_, args.nb_classes = build_dataset(is_train=False, gen_attn=True, args=args)
    dataset_val, nb_classes_ = build_dataset(
        is_train=False, args=args
    )  # nb_classes 是类别数目

    # 采样器：分布式模式下使用 DistributedSampler，否则使用原有采样器
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # CAM 生成的数据管道：提升吞吐（persistent_workers / 预取）
    data_loader_train_ = torch.utils.data.DataLoader(
        dataset_train_,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    logger.info(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        input_size=args.input_size,
        num_classes=args.nb_classes,
        num_sp_tokens=args.num_sp_tokens,
        # decay_parameter=args.gwrp_decay,
    )

    # 微调
    if args.finetune:
        finetune_load_max_reuse(model, args)

    model.to(device)

    # 使用 DDP 包装模型（分布式模式下）
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True  # 允许模型中有未使用的参数
        )
        logger.info(f"Model wrapped with DistributedDataParallel on GPU {args.gpu}")

    # 统计可训练参数量：累加 require_grads=True 的全部参数
    model_without_ddp = model.module if args.distributed else model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # 线性缩放学习率：当全局 batch变大时，按比例放大学习率，保持每样本“更新幅度”大致不变（经典 ImageNet 大批量训练经验）。
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    # 构建优化器
    optimizer = create_optimizer(args, model)
    # AMP 混合精度的损失缩放器
    loss_scaler = NativeScaler()
    # 学习率调度器：根据 args 生成 LR 曲线（常见：cosine、step、multistep），并配置 warmup（warmup_epochs / warmup_lr）。
    lr_scheduler, _ = create_scheduler(args, optimizer)
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 TensorBoard（仅在主进程）
    writer = None
    if args.tensorboard and utils.is_main_process():
        # tensorboard_dir = output_dir / args.tensorboard_dir
        # 展开用户 home 目录路径（~）
        tensorboard_dir = Path(args.tensorboard_dir).expanduser()
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        logger.info(f"TensorBoard logging enabled. Log directory: {tensorboard_dir}")

    # 评估模式
    if args.eval:
        logger.debug("Enter branch: eval")
        # 调用 evaluate() 在验证集进行一轮推理，收集 mAP 指标。 --> 评估样本数目 + mAP 值
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(
            f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%"
        )
        return

    # 注意力图生成
    if args.gen_attention_maps:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        # 使用 model_without_ddp 确保兼容 DDP 包装的模型
        model_without_ddp.load_state_dict(checkpoint["model"])
        generate_attention_maps_ms(data_loader_train_, model, device, args)
        return
    
    # 批处理模式的注意力图生成
    if args.gen_attention_maps_batch:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        generate_attention_maps_ms_batch(data_loader_train_, model, device, args)
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()  # 记录开始时间

    # 初始化最佳验证指标（用于保存 best checkpoint）
    best_map = float('-inf')
    best_epoch = -1

    # 初始化指标历史记录（用于生成折线图）
    entropy_pos_history = []
    pmax_neg_history = []
    epochs_history = []
    
    # 初始化指标日志文件路径（用于持久化保存）
    metrics_log_file = output_dir / "metrics_history.jsonl"

    def plot_val_metrics(epochs, entropy_pos, pmax_neg, output_dir):
        """Generate and save validation metrics plot"""
        if len(epochs) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot two curves
        plt.plot(epochs, entropy_pos, 'b-o', label=r'$\bar{H}$ (Positive Avg Entropy)', linewidth=2, markersize=6)
        plt.plot(epochs, pmax_neg, 'r-s', label=r'$\bar{P}_{max}^{-}$ (Negative Max Response)', linewidth=2, markersize=6)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Validation Metrics: Positive Entropy Mean and Negative Max Response', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plot_path = Path(output_dir) / 'val_metrics_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Validation metrics plot saved to {plot_path}")

    for epoch in range(args.start_epoch, args.epochs):
        # 为分布式采样器设置 epoch，确保每个 epoch 的数据打乱方式不同
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            args=args,
            writer=writer,
        )

        lr_scheduler.step(epoch)  # 按照 epoch 的学习率进行调度

        test_stats = evaluate(data_loader_val, model, device, args=args, writer=writer, epoch=epoch)  # 每轮结束之后做验证

        # 记录指标历史（用于生成折线图）
        if isinstance(test_stats, dict):
            # 准备要保存的数据
            log_entry = {
                'epoch': epoch,
                'timestamp': datetime.datetime.now().isoformat(),
            }
            
            if 'entropy_pos_mean' in test_stats:
                entropy_pos_history.append(test_stats['entropy_pos_mean'])
                log_entry['entropy_pos_mean'] = float(test_stats['entropy_pos_mean'])
            else:
                entropy_pos_history.append(0.0)
                log_entry['entropy_pos_mean'] = 0.0
            
            if 'pmax_neg_mean' in test_stats:
                pmax_neg_history.append(test_stats['pmax_neg_mean'])
                log_entry['pmax_neg_mean'] = float(test_stats['pmax_neg_mean'])
            else:
                pmax_neg_history.append(0.0)
                log_entry['pmax_neg_mean'] = 0.0
            
            # 额外保存 mAP 等其他指标（可选）
            if 'mAP' in test_stats:
                log_entry['mAP'] = float(test_stats['mAP'])
            
            epochs_history.append(epoch)
            
            # 保存到 JSON Lines 文件（追加模式）
            if args.output_dir and utils.is_main_process():
                with open(metrics_log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                logger.info(f"Metrics logged to {metrics_log_file}")
            
            # 生成并保存折线图（保留原有功能）
            if args.output_dir and len(epochs_history) > 0:
                plot_val_metrics(epochs_history, entropy_pos_history, pmax_neg_history, output_dir)

        # 记录并保存 best checkpoint（按 mAP）
        cur_map = None
        if isinstance(test_stats, dict) and 'mAP' in test_stats:
            try:
                cur_map = float(test_stats['mAP'])
            except Exception:
                cur_map = None
        if cur_map is not None and cur_map > best_map:
            best_map = cur_map
            best_epoch = epoch
            if args.output_dir:
                best_ckpt = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if 'lr_scheduler' in locals() else None,
                    'epoch': epoch,
                    'args': args,
                    'best_mAP': best_map,
                }
                utils.save_on_master(best_ckpt, output_dir / 'best_checkpoint.pth')
                logger.info(f"[Checkpoint] New best mAP={best_map:.4f} at epoch={epoch}. Saved to best_checkpoint.pth")

        # TensorBoard 日志记录
        if writer is not None:
            # 记录训练指标
            for key, value in train_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Train/{key}", value, epoch)

            # 记录验证指标
            for key, value in test_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Validation/{key}", value, epoch)

            # 记录学习率
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning_Rate/lr", current_lr, epoch)

            # 记录模型参数数量（仅第一个epoch）
            if epoch == args.start_epoch:
                writer.add_scalar("Model/n_parameters", n_parameters, 0)
            # 记录当前 best mAP
            if best_map != float('-inf'):
                writer.add_scalar("Validation/best_mAP", best_map, epoch)

        # 保存最后一轮的 checkpoint
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 保存最终的 checkpoint（仅主进程）
    if utils.is_main_process():
        torch.save({"model": model_without_ddp.state_dict()}, output_dir / "checkpoint.pth")

    # 关闭 TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard logging completed.")

    total_time = time.time() - start_time  # 记录总共时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def _strip_prefix(sd):
    """去掉常见前缀（module.）。"""
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _select_state_dict(ckpt, cur):
    """从 checkpoint 中选择最匹配当前模型的 state_dict（最大化可重用参数）。
    兼容以下几种：
      - ckpt['model_ema']（若存在，常更平滑）
      - ckpt['model']
      - ckpt['state_dict']（Lightning）
      - ckpt 本体（纯权重）
    选择标准：与当前模型键名相同且形状一致的键数量最多者。
    """
    cands = []
    if isinstance(ckpt, dict):
        for key in ("model_ema", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                cands.append(_strip_prefix(ckpt[key]))
        if not cands:
            # 可能就是纯权重
            cands.append(_strip_prefix(ckpt))
    else:
        cands.append(ckpt)

    def score(sd):
        s = 0
        for k, v in sd.items():
            if k in cur and isinstance(v, torch.Tensor) and cur[k].shape == v.shape:
                s += 1
        return s

    best = max(cands, key=lambda x: score(x))
    return best


def finetune_load_max_reuse(model, args):
    """加载预训练权重并最大化重用；可直接替换你原来的 finetune 代码块。
    说明：
      - args.finetune: URL 或 本地路径
      - args.nb_classes: 目标类别数（用于 MCTformer 多 CLS 复制）
      - args.model: 字符串，检测是否包含 'MCTformer'
      - args.ignore_pos_embed（可选，bool）: 若 True，跳过旧位置编码加载
    """
    # 1) 加载 checkpoint 到 CPU
    if args.finetune.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(
            args.finetune, map_location="cpu", check_hash=True
        )
    else:
        ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)

    cur = model.state_dict()
    sd = _select_state_dict(ckpt, cur)

    # 2) 删除不兼容的分类头（仅当形状不匹配时删除；若匹配则保留以最大化复用）
    for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if k in sd and (k not in cur or sd[k].shape != cur[k].shape):
            print(
                f"[finetune] Removing key {k} from pretrained checkpoint due to shape mismatch: {sd[k].shape} -> {cur.get(k, None)}"
            )
            sd.pop(k)

    # 3) 位置编码处理（可选关闭）
    if not getattr(args, "ignore_pos_embed", False):
        # 3.1 读取 ckpt 与模型中可能的名称
        pe_ckpt_single = sd.get("pos_embed", None)  # (1, L_old, C)
        pe_ckpt_pat = sd.get("pos_embed_pat", None)  # (1, N_old, C)
        pe_ckpt_cls = sd.get("pos_embed_cls", None)  # (1, E, C) 其中 E=extra tokens 数

        # 目标网格 (H_new, W_new)
        H_new = W_new = None
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
            H_new, W_new = model.patch_embed.grid_size
        elif hasattr(model, "patch_embed") and hasattr(
            model.patch_embed, "num_patches"
        ):
            side = int(getattr(model.patch_embed, "num_patches") ** 0.5)
            if side * side == getattr(model.patch_embed, "num_patches"):
                H_new = W_new = side

        def _interpolate_grid(grid_flat):
            """把 (1, N_old, C) 的网格位置编码插值到 (1, H_new*W_new, C)。"""
            if H_new is None or W_new is None:
                # 无法确定目标网格，直接返回原始
                return grid_flat
            C = grid_flat.shape[-1]
            side_old = int(grid_flat.shape[1] ** 0.5)
            grid = grid_flat.reshape(1, side_old, side_old, C).permute(
                0, 3, 1, 2
            )  # (1,C,H_old,W_old)
            if (side_old, side_old) != (H_new, W_new):
                grid = F.interpolate(
                    grid, size=(H_new, W_new), mode="bicubic", align_corners=False
                )
            return grid.permute(0, 2, 3, 1).reshape(1, H_new * W_new, C)

        if pe_ckpt_single is not None:
            # 3.2 单键 pos_embed：自动拆分 extra 与网格，再按目标网格插值
            pe = pe_ckpt_single  # (1, L_old, C)
            C = pe.shape[-1]
            L = pe.shape[-2]
            # 自动推断 extra token 数，使 L-e 为完全平方数（兼容 cls + dist 等多种情况）
            num_extra = 1
            for e in range(0, 8):
                side = int((L - e) ** 0.5)
                if side * side == (L - e):
                    num_extra = e
                    break
            extra = pe[:, :num_extra, :]
            grid = pe[:, num_extra:, :]
            grid = _interpolate_grid(grid)

            # 如果模型是 MCTformer，多 CLS token：当 ckpt 的 extra 只有 1，而模型需要 K>1，则复制
            need_multi_cls = ("cls_token" in cur and cur["cls_token"].shape[1] > 1) or (
                "MCTformer" in getattr(args, "model", "")
            )
            if need_multi_cls and extra.shape[1] == 1:
                K = (
                    cur["cls_token"].shape[1]
                    if "cls_token" in cur
                    else getattr(args, "nb_classes", 0)
                )
                if K and K > 1:
                    extra = extra.repeat(1, K, 1)
                    if "cls_token" in sd and sd["cls_token"].shape[1] == 1:
                        sd["cls_token"] = sd["cls_token"].repeat(1, K, 1)

            # 写回到“模型所需”的键名
            if "pos_embed_pat" in cur or "pos_embed_cls" in cur:
                sd["pos_embed_pat"] = grid
                sd["pos_embed_cls"] = extra
            else:
                # 模型仍然期望单键：合并
                sd["pos_embed"] = torch.cat([extra, grid], dim=1)

        elif pe_ckpt_pat is not None or pe_ckpt_cls is not None:
            # 3.3 ckpt 已拆分：仅对网格部分插值；cls 部分按需要复制
            if pe_ckpt_pat is not None:
                sd["pos_embed_pat"] = _interpolate_grid(pe_ckpt_pat)
            if pe_ckpt_cls is not None:
                need_multi_cls = (
                    "cls_token" in cur and cur["cls_token"].shape[1] > 1
                ) or ("MCTformer" in getattr(args, "model", ""))
                if need_multi_cls and pe_ckpt_cls.shape[1] == 1:
                    K = (
                        cur["cls_token"].shape[1]
                        if "cls_token" in cur
                        else getattr(args, "nb_classes", 0)
                    )
                    if K and K > 1:
                        sd["pos_embed_cls"] = pe_ckpt_cls.repeat(1, K, 1)
                        if "cls_token" in sd and sd["cls_token"].shape[1] == 1:
                            sd["cls_token"] = sd["cls_token"].repeat(1, K, 1)
        else:
            print("[finetune] checkpoint has no position embedding; skip PE loading.")

    # —— BEGIN: cls_token 扩展补丁 ——
    if "cls_token" in sd and "cls_token" in cur:
        src = sd["cls_token"]  # from checkpoint, e.g. (1, 1, C)
        tgt = cur["cls_token"]  # from model,      e.g. (1, 20, C)
        if src.shape != tgt.shape:
            same_batch = src.shape[0] == tgt.shape[0]
            same_dim = src.shape[-1] == tgt.shape[-1]
            can_repeat = src.shape[1] == 1 and tgt.shape[1] > 1
            if same_batch and same_dim and can_repeat:
                sd["cls_token"] = src.repeat(1, tgt.shape[1], 1)
                print(
                    f"[finetune] expanded cls_token: {tuple(src.shape)} -> {tuple(sd['cls_token'].shape)}"
                )
            else:
                # 不可自动扩展时，避免报错：删除之（模型会用自身初始化的参数）
                print(
                    f"[finetune] drop cls_token due to incompatible shape: {tuple(src.shape)} vs {tuple(tgt.shape)}"
                )
                sd.pop("cls_token")

    # 如果 ckpt 里恰好有 pos_embed_cls，也一并扩展到 K 份（不会影响无该键的情况）
    if "pos_embed_cls" in sd and "cls_token" in cur:
        pe_cls = sd["pos_embed_cls"]
        K = cur["cls_token"].shape[1]
        if (
            pe_cls.shape[1] == 1
            and K > 1
            and pe_cls.shape[-1] == cur["cls_token"].shape[-1]
        ):
            sd["pos_embed_cls"] = pe_cls.repeat(1, K, 1)
            print(
                f"[finetune] expanded pos_embed_cls: {tuple(pe_cls.shape)} -> {tuple(sd['pos_embed_cls'].shape)}"
            )
    # —— END: cls_token 扩展补丁 ——

    else:
        # 显式忽略旧位置编码
        for k in list(sd.keys()):
            if any(p in k for p in ["pos_embed", "pos_embed_pat", "pos_embed_cls"]):
                sd.pop(k)

    # 4) 加载（strict=False），并打印缺失/多余键以便排查
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[finetune] missing keys:", len(missing), "\n ", missing)
    print("[finetune] unexpected keys:", len(unexpected), "\n ", unexpected)

    # 5) 返回以便上层可根据需要做额外处理（可选）
    return missing, unexpected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SuiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"训练设备：{args.device}")
    # --- args test for `engine.py->generate_attn_maps_ms()`
    # print("visualize_cls_attn:", getattr(args, "visualize_cls_attn", None))
    # print("attention_dir:", getattr(args, "attention_dir", None))
    # try:
    #     from utils.visualize import show_cam_on_image

    #     print("show_cam_on_image:", show_cam_on_image is not None)
    # except Exception as e:
    #     print("show_cam_on_image import failed:", e)
    # --- part end. ---

    main(args)
