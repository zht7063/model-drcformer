# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import random
from typing import (
    Any,
    Callable,
    Optional,
    List,
    Sequence,
    Tuple,
)  # 合并了所有typing的导入

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor  # 新增导入
from torch.utils.data import Dataset
from loguru import logger
from pathlib import Path

from PIL import Image
import PIL.Image  # 保留，因为PIL.Image和PIL.Image是不同的导入方式，后者直接引入Image模块

from skimage.transform import rescale
from skimage.segmentation import slic
from fast_slic.avx2 import SlicAvx2
import fast_slic  # 保留，因为fast_slic和fast_slic.avx2是不同的导入方式

from torchvision import (
    datasets,
    transforms as T,
)  # 整合并清理了torchvision.transforms的导入
from torchvision.datasets.folder import ImageFolder, default_loader
import torchvision.transforms.functional as TF
from torchvision.transforms import (
    InterpolationMode,
)  # 保留，确保所有transform相关导入都在一起

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torchvision import transforms


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list


def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    """
    Load class labels from .npy file.
    
    Supports multiple naming conventions:
    - VOC format: "2007_000032" or "2007_000032.jpg"
    - COCO format: "COCO_train2014_000000000009" or "COCO_train2014_000000000009.jpg"
    
    Args:
        img_name_list: List of image names (without extension)
        label_file_path: Path to .npy file containing class labels dictionary
        
    Returns:
        List of class label arrays
    """
    if label_file_path is None:
        label_file_path = "voc12/cls_labels.npy"
    
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(
            f"Class labels file not found: {label_file_path}\n"
            f"For COCO dataset, generate it using: python tools/generate_coco_cls_labels.py"
        )
    
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    
    for img_id in img_name_list:
        # Try multiple key variations to find the label
        label = None
        for key_variation in [img_id, img_id + ".jpg", os.path.basename(img_id), os.path.basename(img_id) + ".jpg"]:
            if key_variation in cls_labels_dict:
                label = cls_labels_dict[key_variation]
                break
        
        if label is None:
            # If still not found, raise an error with helpful message
            raise KeyError(
                f"Image '{img_id}' not found in class labels file {label_file_path}.\n"
                f"Available keys sample: {list(cls_labels_dict.keys())[:5]}\n"
                f"Make sure the class labels file matches your image list."
            )
        
        label_list.append(label)
    
    return label_list


class VOC12Dataset(Dataset):
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        train=True,
        transform=None,
        gen_attn=False,
        pair_geom: bool = False,
        out_size: int = 224,
        crop_scale=(0.5, 1.0),
        hflip_p: float = 0.5,
        sp_idx_dir: str = None,
        num_sp_tokens: int = 196,
    ):
        img_name_list_path = os.path.join(
            img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt'
        )
        self.img_name_list = load_img_name_list(img_name_list_path)
        label_file_path = os.path.join(
            os.path.dirname(img_name_list_path), "cls_labels.npy"
        )
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )
        self.voc12_root = voc12_root
        self.transform = transform  # 注意：这里建议仅放"非几何"增强
        self.pair_geom = pair_geom
        self.out_size = out_size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.sp_idx_dir = sp_idx_dir
        self.num_sp_tokens = num_sp_tokens

        # 添加一个参数用来标记是否是训练模式，以判断是否读取 dino 超像素
        self.train = train

    def _load_dino_superpixel(self, img_name: str):
        """加载DINO超像素标签"""
        if self.sp_idx_dir is None:
            return None
        
        sp_path = os.path.join(self.sp_idx_dir, f"{img_name}.npy")
        if os.path.exists(sp_path):
            try:
                dino_labels = np.load(sp_path).astype(np.int64)
                return dino_labels
            except Exception as e:
                logger.warning(f"Failed to load DINO superpixel for {img_name}: {e}")
                return None
        else:
            logger.warning(f"[VOC12Dataset] DINO superpixel missing for {img_name} at path: {sp_path}")
            return None

    def _paired_geom(self, img_pil: PIL.Image.Image, spix_np: np.ndarray, dino_np: np.ndarray = None):
        """对 (img, spix, dino) 做同参数的随机裁剪+resize 和随机水平翻转。"""
        size = (self.out_size, self.out_size)
        # 1) 随机裁剪参数（1:1 长宽比，常见于 ViT）
        i, j, h, w = T.RandomResizedCrop.get_params(
            img_pil, scale=self.crop_scale, ratio=(1.0, 1.0)
        )
        # 图像：双三次
        img = TF.resized_crop(img_pil, i, j, h, w, size, InterpolationMode.BICUBIC)
        
        # SLIC超像素：先裁剪再最近邻 resize（保持离散 id）
        sp = torch.from_numpy(spix_np)  # (H0,W0) long
        sp = sp[i : i + h, j : j + w]
        sp = F.interpolate(sp[None, None].float(), size=size, mode="nearest")[
            0, 0
        ].long()
        
        # DINO超像素：如果存在，应用相同的变换
        dino = None
        if dino_np is not None:
            dino = torch.from_numpy(dino_np)  # (H0,W0) long
            dino = dino[i : i + h, j : j + w]
            dino = F.interpolate(dino[None, None].float(), size=size, mode="nearest")[
                0, 0
            ].long()
        
        # 2) 随机水平翻转
        if random.random() < self.hflip_p:
            img = TF.hflip(img)
            sp = torch.flip(sp, dims=[1])  # 水平翻转：沿 W 轴
            if dino is not None:
                dino = torch.flip(dino, dims=[1])
        
        return img, sp, dino

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img_pil = PIL.Image.open(
            os.path.join(self.voc12_root, "JPEGImages", name + ".jpg")
        ).convert("RGB")
        # 原图上生成SLIC超像素
        img_np = np.array(img_pil)
        spix_np = get_spix_label(img_np, num_components=self.num_sp_tokens).astype(np.int64)  # (H,W), (196,)
        
        # 加载DINO超像素标签
        if self.train:
            dino_np = self._load_dino_superpixel(name)
        else:
            dino_np = None

        if self.pair_geom:
            # 成对几何增强（裁剪/翻转/resize）
            img_pil, sp, dino = self._paired_geom(
                img_pil, spix_np, dino_np
            )  # img_pil: PIL.Image, sp: (H,W) long, dino: (H,W) long or None
            img = img_pil
            # 缺失则用占位张量（全-1，Long）
            if dino is None:
                dino = torch.full_like(sp, fill_value=-1, dtype=torch.long)
        else:
            # 仅 resize 对齐（当 transform 里没有随机裁剪/翻转时可用）
            img = img_pil
            if self.transform and hasattr(self.transform, "__call__"):
                # 先为了得到输出尺寸，暂时用一个"仅 resize"的分支；
                # 若 transform 自身不含 Resize，可直接跳过这段，改为读取 transform 后的张量尺寸。
                pass
            # 目标尺寸：使用 out_size（与 ViT 输入一致）
            Ht = Wt = self.out_size
            sp = torch.from_numpy(spix_np)
            sp = F.interpolate(sp[None, None].float(), size=(Ht, Wt), mode="nearest")[
                0, 0
            ].long()
            
            # DINO超像素也需要resize对齐；若缺失则占位
            if dino_np is not None:
                dino = torch.from_numpy(dino_np)
                dino = F.interpolate(dino[None, None].float(), size=(Ht, Wt), mode="nearest")[
                    0, 0
                ].long()
            else:
                dino = torch.full((Ht, Wt), fill_value=-1, dtype=torch.long)
            
            # 再把图像也 resize 到同样大小（保持几何一致）
            img = TF.resize(img, (Ht, Wt), interpolation=InterpolationMode.BICUBIC)

        # 后续"非几何"变换（ToTensor/Normalize/ColorJitter 等）
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.pil_to_tensor(img).float() / 255.0

        label = torch.as_tensor(self.label_list[idx], dtype=torch.float32)
        return img, sp, dino, label

    def __len__(self):
        return len(self.img_name_list)


class COCODataset(Dataset):
    """
    COCO dataset for single-scale training, mirroring VOC12Dataset structure.
    
    Key features:
      - Image path: {coco_root}/{split}2014/COCO_{split}2014_{image_id:012d}.jpg
      - Class labels: loaded from {coco_root}/cls_labels.npy (80-dim binary vector per image)
      - Image list: from file or auto-scan directory
      - Superpixels: SLIC + optional DINO (from sp_idx_dir)
      - Supports paired geometric augmentation (crop/resize/hflip)
    """
    def __init__(
        self,
        img_name_list_path: str,
        coco_root: str,
        split: str = "train",
        train: bool = True,
        transform: Optional[Callable] = None,
        gen_attn: bool = False,
        pair_geom: bool = False,
        out_size: int = 224,
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        hflip_p: float = 0.5,
        sp_idx_dir: Optional[str] = None,
        num_sp_tokens: int = 196,
    ):
        """
        Args:
            img_name_list_path: Path to image list file or directory containing it
            coco_root: COCO dataset root directory
            split: Dataset split ('train' or 'val')
            train: Whether in training mode
            transform: Non-geometric transforms (ToTensor/Normalize/etc)
            gen_attn: Whether generating attention maps
            pair_geom: Whether to apply paired geometric augmentation
            out_size: Output image size (default: 224)
            crop_scale: Scale range for RandomResizedCrop (default: (0.5, 1.0))
            hflip_p: Probability of horizontal flip (default: 0.5)
            sp_idx_dir: Directory containing DINO superpixel index files
        """
        self.coco_root = coco_root
        self.split = split
        self.train = train
        self.gen_attn = gen_attn
        self.transform = transform
        self.pair_geom = pair_geom
        self.out_size = out_size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.sp_idx_dir = sp_idx_dir
        self.num_sp_tokens = num_sp_tokens
        
        # Load image list
        if os.path.isfile(img_name_list_path):
            # Direct file path
            list_file = img_name_list_path
        elif os.path.isdir(img_name_list_path):
            # Directory: look for {split}_id.txt
            list_file = os.path.join(img_name_list_path, f"{split}_id.txt")
        else:
            # Path might be a directory that doesn't exist yet, or a pattern
            list_file = os.path.join(img_name_list_path, f"{split}_id.txt")
        
        if os.path.exists(list_file):
            self.img_name_list = load_img_name_list(list_file)
            logger.info(f"[COCODataset] Loaded {len(self.img_name_list)} images from {list_file}")
        else:
            # Auto-scan directory if list file not found
            img_dir = os.path.join(coco_root, f"{split}2014")
            if os.path.exists(img_dir):
                import glob
                img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
                self.img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
                self.img_name_list.sort()
                logger.info(f"[COCODataset] Auto-scanned {len(self.img_name_list)} images from {img_dir}")
            else:
                raise FileNotFoundError(f"[COCODataset] Neither list file {list_file} nor image directory {img_dir} exists")
        
        # Load class labels
        label_file_path = os.path.join(coco_root, "cls_labels.npy")
        if not os.path.exists(label_file_path):
            logger.warning(
                f"[COCODataset] Class labels file not found at {label_file_path}. "
                f"All images will have zero labels. Please generate it using tools/generate_coco_cls_labels.py"
            )
            # Create dummy zero labels (80 classes for COCO)
            self.label_list = [np.zeros(80, dtype=np.float32) for _ in self.img_name_list]
        else:
            self.label_list = load_image_label_list_from_npy(
                self.img_name_list, label_file_path
            )
            logger.info(f"[COCODataset] Loaded class labels from {label_file_path}")

    def _load_dino_superpixel(self, img_name: str):
        """Load DINO superpixel labels"""
        if self.sp_idx_dir is None:
            return None
        
        sp_path = os.path.join(self.sp_idx_dir, f"{img_name}.npy")
        if os.path.exists(sp_path):
            try:
                dino_labels = np.load(sp_path).astype(np.int64)
                return dino_labels
            except Exception as e:
                logger.warning(f"Failed to load DINO superpixel for {img_name}: {e}")
                return None
        else:
            # Don't warn for every missing file to avoid log spam
            return None

    def _paired_geom(
        self, img_pil: PIL.Image.Image, spix_np: np.ndarray, dino_np: Optional[np.ndarray] = None
    ) -> Tuple[PIL.Image.Image, Tensor, Optional[Tensor]]:
        """Apply paired geometric augmentation (crop/resize/hflip) to (img, spix, dino)"""
        size = (self.out_size, self.out_size)
        # 1) Random crop parameters (1:1 aspect ratio)
        i, j, h, w = T.RandomResizedCrop.get_params(
            img_pil, scale=self.crop_scale, ratio=(1.0, 1.0)
        )
        # Image: bicubic interpolation
        img = TF.resized_crop(img_pil, i, j, h, w, size, InterpolationMode.BICUBIC)
        
        # SLIC superpixel: crop then nearest-neighbor resize
        sp = torch.from_numpy(spix_np)  # (H0,W0) long
        sp = sp[i : i + h, j : j + w]
        sp = F.interpolate(sp[None, None].float(), size=size, mode="nearest")[0, 0].long()
        
        # DINO superpixel: apply same transform if exists
        dino = None
        if dino_np is not None:
            dino = torch.from_numpy(dino_np)  # (H0,W0) long
            dino = dino[i : i + h, j : j + w]
            dino = F.interpolate(dino[None, None].float(), size=size, mode="nearest")[0, 0].long()
        
        # 2) Random horizontal flip
        if random.random() < self.hflip_p:
            img = TF.hflip(img)
            sp = torch.flip(sp, dims=[1])  # flip along width
            if dino is not None:
                dino = torch.flip(dino, dims=[1])
        
        return img, sp, dino

    def __getitem__(self, idx: int):
        name = self.img_name_list[idx]
        # COCO image path: {split}2014/COCO_{split}2014_{image_id}.jpg
        img_path = os.path.join(self.coco_root, f"{self.split}2014", f"{name}.jpg")
        img_pil = PIL.Image.open(img_path).convert("RGB")
        
        # Generate SLIC superpixels on original image
        img_np = np.array(img_pil)
        spix_np = get_spix_label(img_np, num_components=self.num_sp_tokens).astype(np.int64)  # (H,W)
        
        # Load DINO superpixel labels (only in training mode)
        dino_np = None
        if self.train and not self.gen_attn:
            dino_np = self._load_dino_superpixel(name)

        if self.pair_geom:
            # Paired geometric augmentation (crop/flip/resize)
            img_pil, sp, dino = self._paired_geom(img_pil, spix_np, dino_np)
            img = img_pil
            # Use placeholder if DINO is missing
            if dino is None:
                dino = torch.full_like(sp, fill_value=-1, dtype=torch.long)
        else:
            # Only resize (no random crop/flip)
            img = img_pil
            Ht = Wt = self.out_size
            sp = torch.from_numpy(spix_np)
            sp = F.interpolate(sp[None, None].float(), size=(Ht, Wt), mode="nearest")[0, 0].long()
            
            # Resize DINO superpixel if exists
            if dino_np is not None:
                dino = torch.from_numpy(dino_np)
                dino = F.interpolate(dino[None, None].float(), size=(Ht, Wt), mode="nearest")[0, 0].long()
            else:
                dino = torch.full((Ht, Wt), fill_value=-1, dtype=torch.long)
            
            # Resize image to match
            img = TF.resize(img, (Ht, Wt), interpolation=InterpolationMode.BICUBIC)

        # Apply non-geometric transforms (ToTensor/Normalize/ColorJitter/etc)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.pil_to_tensor(img).float() / 255.0

        label = torch.as_tensor(self.label_list[idx], dtype=torch.float32)
        return img, sp, dino, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12DatasetMS(Dataset):
    """
    Multi-scale VOC12 dataset that mirrors the geometry handling of VOC12Dataset,
    with *paired* image/superpixel augmentations and optional test-time flip.

    Key behavior (aligned with single-scale VOC12Dataset):
      - Superpixels are generated on the *original* image (H0,W0), then any
        geometry (crop/resize/hflip) is applied to (img, spix) with identical
        params.
      - Non-geometric transforms (ToTensor/Normalize/ColorJitter, etc.) are
        applied to images **only** after geometry is finalized.
      - Superpixels are resized using nearest-neighbor and kept as Long tensors.

    Returns: (msf_img_list, msf_sp_list, label, annotation)
      where each list has length = 2 * len(scales)  (original + horizontal flip),
      and annotation is a LongTensor of the GT segmentation mask aligned to the
      reference resolution of msf_img_list[0] (shape: H x W, integer class indices).
    """

    def __init__(
        self,
        img_name_list_path: str,
        voc12_root: str,
        scales: Sequence[float],
        train: bool = True,
        transform: T.Compose | None = None,
        gen_attn: bool = False,
        unit: int = 1,
        pair_geom: bool = False,
        out_size: int = 224,
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        hflip_p: float = 0.5,
        num_sp_tokens: int = 196,
    ) -> None:
        list_file = os.path.join(
            img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt'
        )
        self.img_name_list = load_img_name_list(list_file)

        label_file_path = os.path.join(os.path.dirname(list_file), "cls_labels.npy")
        self.label_list = load_image_label_list_from_npy(
            self.img_name_list, label_file_path
        )

        self.voc12_root = voc12_root
        self.transform = transform  # 非几何增强；几何在本类里处理
        self.unit = unit
        self.scales = list(scales)
        self.pair_geom = pair_geom
        self.out_size = out_size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.num_sp_tokens = num_sp_tokens

    # ---- 对 (img, sp) 做配对的几何增强：crop/resize/hflip ----
    def _paired_geom(
        self, img_pil: PIL.Image.Image, spix_np: np.ndarray
    ) -> tuple[PIL.Image.Image, Tensor]:
        size = (self.out_size, self.out_size)
        # 与 ViT 一致：固定宽高比 1:1 的 RandomResizedCrop
        i, j, h, w = T.RandomResizedCrop.get_params(
            img_pil, scale=self.crop_scale, ratio=(1.0, 1.0)
        )
        img = TF.resized_crop(img_pil, i, j, h, w, size, InterpolationMode.BICUBIC)

        # sp 超像素图与 img 使用同样的裁剪与缩放（最近邻）
        sp = torch.from_numpy(spix_np)  # (H0,W0), int64
        sp = sp[i : i + h, j : j + w]
        sp = F.interpolate(sp[None, None].float(), size=size, mode="nearest")[
            0, 0
        ].long()

        if random.random() < self.hflip_p:
            img = TF.hflip(img)
            sp = torch.flip(sp, dims=[1])  # 沿宽度维翻转 (H,W) -> dim=1

        return img, sp

    def __len__(self) -> int:
        return len(self.img_name_list)

    def __getitem__(self, idx: int):
        name = self.img_name_list[idx]
        img_pil = Image.open(
            os.path.join(self.voc12_root, "JPEGImages", name + ".jpg")
        ).convert("RGB")

        # 1) 在"原图"上计算一次超像素索引图（直接调用你给的函数）
        img_np = np.array(img_pil)
        spix_np = get_spix_label(img_np, num_components=self.num_sp_tokens).astype(np.int64)  # (H0,W0), 连续整型索引

        # 2) 标签
        label = torch.as_tensor(self.label_list[idx], dtype=torch.float32)

        # 3) 可选：尺寸向 unit 对齐（保持原 MS 逻辑）
        # PIL.size = (W,H)
        rounded_size = (
            int(round(img_pil.size[0] / self.unit) * self.unit),  # W
            int(round(img_pil.size[1] / self.unit) * self.unit),  # H
        )

        ms_img_list: List[Tensor | PIL.Image.Image] = []
        ms_sp_list: List[Tensor] = []

        for s in self.scales:
            # 注意：torch/TV 的 size=(H,W)
            target_W = int(round(rounded_size[0] * s))
            target_H = int(round(rounded_size[1] * s))
            target_size_hw = (target_H, target_W)

            if self.pair_geom:
                # (a) 先对 (img, sp) 做配对的裁剪/缩放/翻转
                s_img, s_sp = self._paired_geom(img_pil, spix_np)
                # (b) 如果 out_size 与目标不一致，再整体 resize 到目标尺度
                if target_size_hw != (self.out_size, self.out_size):
                    s_img = TF.resize(
                        s_img, target_size_hw, interpolation=InterpolationMode.BICUBIC
                    )
                    s_sp = F.interpolate(
                        s_sp[None, None].float(), size=target_size_hw, mode="nearest"
                    )[0, 0].long()
            else:
                # 仅做尺度调整（无随机裁剪与翻转）
                s_img = TF.resize(
                    img_pil, target_size_hw, interpolation=InterpolationMode.BICUBIC
                )
                s_sp = torch.from_numpy(spix_np)
                s_sp = F.interpolate(
                    s_sp[None, None].float(), size=target_size_hw, mode="nearest"
                )[0, 0].long()

            ms_img_list.append(s_img)
            ms_sp_list.append(s_sp)

        # 4) 仅对图像做“非几何”变换；sp 保持为 Long 的索引图
        if self.transform is not None:
            for i in range(len(ms_img_list)):
                if isinstance(ms_img_list[i], PIL.Image.Image):
                    ms_img_list[i] = self.transform(ms_img_list[i])
                # 若 transform 中含有仅 tensor 可用的操作，且 ms_img_list[i] 已是 tensor，可在此扩展
        else:
            # 若未提供 transform，默认转 tensor 到 [0,1]
            for i in range(len(ms_img_list)):
                if isinstance(ms_img_list[i], PIL.Image.Image):
                    ms_img_list[i] = TF.pil_to_tensor(ms_img_list[i]).float() / 255.0

        # 5) 为每个尺度追加一个水平翻转分支（图像与 sp 同步翻转）
        msf_img_list: List[Tensor] = []
        msf_sp_list: List[Tensor] = []
        for img_t, sp_t in zip(ms_img_list, ms_sp_list):
            # 原图/原 sp - 为 sp 添加通道维度以与图像对齐 (H,W) -> (1,H,W)
            msf_img_list.append(img_t)
            msf_sp_list.append(sp_t.unsqueeze(0))  # (H,W) -> (1,H,W)
            # 水平翻转：img (C,H,W) -> dim=-1, sp (1,H,W) -> dim=-1
            msf_img_list.append(torch.flip(img_t, dims=[-1]))
            msf_sp_list.append(torch.flip(sp_t.unsqueeze(0), dims=[-1]))  # 先添加维度再翻转

        # ---- 生成 GT 分割标签 annotation 并对齐到参考分辨率 ----
        # 参考分辨率来自第一个尺度的图像张量
        ref_h, ref_w = msf_img_list[0].shape[-2], msf_img_list[0].shape[-1]
        seg_paths = [
            os.path.join(self.voc12_root, "SegmentationClassAug", f"{name}.png"),
            os.path.join(self.voc12_root, "SegmentationClass", f"{name}.png"),
        ]
        seg_path = None
        for p in seg_paths:
            if os.path.exists(p):
                seg_path = p
                break

        if seg_path is None:
            logger.warning(f"[VOC12DatasetMS] GT mask not found for {name}. Using zeros mask.")
            ann_np = np.zeros((ref_h, ref_w), dtype=np.uint8)
            annotation = torch.from_numpy(ann_np).long()
        else:
            seg_pil = Image.open(seg_path)
            seg_np = np.array(seg_pil)  # palette-encoded integers, shape (H0,W0)
            # 保持原始索引，255 视作忽略区域；用最近邻对齐到参考分辨率
            seg_t = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).float()
            seg_t = F.interpolate(seg_t, size=(ref_h, ref_w), mode="nearest")
            annotation = seg_t[0, 0].long()  # (H,W) Long

        return msf_img_list, msf_sp_list, label, annotation


class COCODatasetMS(Dataset):
    """
    Multi-scale COCO dataset for CAM generation, mirroring VOC12DatasetMS structure.
    
    Key features:
      - Multi-scale inference with test-time flip
      - Image path: {coco_root}/{split}2014/COCO_{split}2014_{image_id}.jpg
      - Returns: (msf_img_list, msf_sp_list, label, name)
      - Each list has length = 2 * len(scales) (original + horizontal flip)
    """
    def __init__(
        self,
        img_name_list_path: str,
        coco_root: str,
        split: str = "train",
        scales: Sequence[float] = (1.0, 0.75, 1.25),
        train: bool = True,
        transform: Optional[T.Compose] = None,
        gen_attn: bool = False,
        unit: int = 1,
        pair_geom: bool = False,
        out_size: int = 224,
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        hflip_p: float = 0.5,
        num_sp_tokens: int = 196,
    ) -> None:
        """
        Args:
            img_name_list_path: Path to image list file or directory
            coco_root: COCO dataset root directory
            split: Dataset split ('train' or 'val')
            scales: List of scale factors for multi-scale inference
            train: Whether in training mode
            transform: Non-geometric transforms
            gen_attn: Whether generating attention maps
            unit: Unit for size rounding (default: 1)
            pair_geom: Whether to apply paired geometric augmentation
            out_size: Output size for paired geom mode
            crop_scale: Scale range for RandomResizedCrop
            hflip_p: Probability of horizontal flip
        """
        self.coco_root = coco_root
        self.split = split
        self.transform = transform
        self.unit = unit
        self.scales = list(scales)
        self.pair_geom = pair_geom
        self.out_size = out_size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.num_sp_tokens = num_sp_tokens
        
        # Load image list
        if os.path.isfile(img_name_list_path):
            list_file = img_name_list_path
        elif os.path.isdir(img_name_list_path):
            list_file = os.path.join(img_name_list_path, f"{split}_id.txt")
        else:
            list_file = os.path.join(img_name_list_path, f"{split}_id.txt")
        
        if os.path.exists(list_file):
            self.img_name_list = load_img_name_list(list_file)
            logger.info(f"[COCODatasetMS] Loaded {len(self.img_name_list)} images from {list_file}")
        else:
            # Auto-scan directory
            img_dir = os.path.join(coco_root, f"{split}2014")
            if os.path.exists(img_dir):
                import glob
                img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
                self.img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
                self.img_name_list.sort()
                logger.info(f"[COCODatasetMS] Auto-scanned {len(self.img_name_list)} images from {img_dir}")
            else:
                raise FileNotFoundError(f"[COCODatasetMS] Neither list file {list_file} nor image directory {img_dir} exists")
        
        # Load class labels
        label_file_path = os.path.join(coco_root, "cls_labels.npy")
        if not os.path.exists(label_file_path):
            logger.warning(
                f"[COCODatasetMS] Class labels file not found at {label_file_path}. "
                f"Using zero labels. Generate it using tools/generate_coco_cls_labels.py"
            )
            self.label_list = [np.zeros(80, dtype=np.float32) for _ in self.img_name_list]
        else:
            self.label_list = load_image_label_list_from_npy(
                self.img_name_list, label_file_path
            )
            logger.info(f"[COCODatasetMS] Loaded class labels from {label_file_path}")

    def _paired_geom(
        self, img_pil: PIL.Image.Image, spix_np: np.ndarray
    ) -> Tuple[PIL.Image.Image, Tensor]:
        """Apply paired geometric augmentation to (img, sp)"""
        size = (self.out_size, self.out_size)
        # Random crop (1:1 aspect ratio)
        i, j, h, w = T.RandomResizedCrop.get_params(
            img_pil, scale=self.crop_scale, ratio=(1.0, 1.0)
        )
        img = TF.resized_crop(img_pil, i, j, h, w, size, InterpolationMode.BICUBIC)

        # SLIC superpixel: same crop and resize
        sp = torch.from_numpy(spix_np)  # (H0,W0), int64
        sp = sp[i : i + h, j : j + w]
        sp = F.interpolate(sp[None, None].float(), size=size, mode="nearest")[0, 0].long()

        if random.random() < self.hflip_p:
            img = TF.hflip(img)
            sp = torch.flip(sp, dims=[1])

        return img, sp

    def __len__(self) -> int:
        return len(self.img_name_list)

    def __getitem__(self, idx: int):
        name = self.img_name_list[idx]
        # COCO image path
        img_path = os.path.join(self.coco_root, f"{self.split}2014", f"{name}.jpg")
        img_pil = Image.open(img_path).convert("RGB")

        # Generate SLIC superpixels on original image
        img_np = np.array(img_pil)
        spix_np = get_spix_label(img_np, num_components=self.num_sp_tokens).astype(np.int64)  # (H0,W0)

        # Label
        label = torch.as_tensor(self.label_list[idx], dtype=torch.float32)

        # Round size to unit
        rounded_size = (
            int(round(img_pil.size[0] / self.unit) * self.unit),  # W
            int(round(img_pil.size[1] / self.unit) * self.unit),  # H
        )

        ms_img_list: List[Tensor | PIL.Image.Image] = []
        ms_sp_list: List[Tensor] = []

        for s in self.scales:
            # Target size (H, W)
            target_W = int(round(rounded_size[0] * s))
            target_H = int(round(rounded_size[1] * s))
            target_size_hw = (target_H, target_W)

            if self.pair_geom:
                # Paired geometric augmentation
                s_img, s_sp = self._paired_geom(img_pil, spix_np)
                # Resize to target scale if needed
                if target_size_hw != (self.out_size, self.out_size):
                    s_img = TF.resize(
                        s_img, target_size_hw, interpolation=InterpolationMode.BICUBIC
                    )
                    s_sp = F.interpolate(
                        s_sp[None, None].float(), size=target_size_hw, mode="nearest"
                    )[0, 0].long()
            else:
                # Only scale (no random crop/flip)
                s_img = TF.resize(
                    img_pil, target_size_hw, interpolation=InterpolationMode.BICUBIC
                )
                s_sp = torch.from_numpy(spix_np)
                s_sp = F.interpolate(
                    s_sp[None, None].float(), size=target_size_hw, mode="nearest"
                )[0, 0].long()

            ms_img_list.append(s_img)
            ms_sp_list.append(s_sp)

        # Test-time horizontal flip: duplicate each scale
        msf_img_list: List[Tensor] = []
        msf_sp_list: List[Tensor] = []
        for img, sp in zip(ms_img_list, ms_sp_list):
            # Original
            if self.transform is not None:
                img_t = self.transform(img)
            else:
                img_t = TF.pil_to_tensor(img).float() / 255.0
            msf_img_list.append(img_t)
            msf_sp_list.append(sp)
            
            # Horizontal flip
            img_flip = TF.hflip(img)
            if self.transform is not None:
                img_flip_t = self.transform(img_flip)
            else:
                img_flip_t = TF.pil_to_tensor(img_flip).float() / 255.0
            msf_img_list.append(img_flip_t)
            msf_sp_list.append(torch.flip(sp, dims=[1]))

        # ---- 生成 GT 分割标签 annotation 并对齐到参考分辨率 ----
        # 参考分辨率来自第一个尺度的图像张量
        ref_h, ref_w = msf_img_list[0].shape[-2], msf_img_list[0].shape[-1]
        
        # COCO 分割掩码路径（由 convert_coco_to_segmentation_masks.py 生成）
        seg_path = os.path.join(self.coco_root, f"segmentation_masks_{self.split}", f"{name}.png")
        
        if not os.path.exists(seg_path):
            logger.warning(f"[COCODatasetMS] GT mask not found for {name}. Using zeros mask.")
            ann_np = np.zeros((ref_h, ref_w), dtype=np.uint8)
            annotation = torch.from_numpy(ann_np).long()
        else:
            seg_pil = Image.open(seg_path)
            seg_np = np.array(seg_pil)  # palette-encoded integers, shape (H0,W0)
            # 保持原始索引，0=背景，1-80=COCO类别；用最近邻对齐到参考分辨率
            seg_t = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).float()
            seg_t = F.interpolate(seg_t, size=(ref_h, ref_w), mode="nearest")
            annotation = seg_t[0, 0].long()  # (H,W) Long

        return msf_img_list, msf_sp_list, label, annotation


def build_dataset(is_train, args, dataset_type="VOC12", gen_attn=False):
    transform = build_transform(is_train, args)
    dataset = None
    nb_classes = None

    num_sp_tokens = getattr(args, 'num_sp_tokens', 196)
    
    if args.data_set == 'VOC12':
        dataset = VOC12Dataset(
            img_name_list_path=args.img_list,
            voc12_root=args.data_path,
            train=is_train,
            gen_attn=gen_attn,
            transform=transform,
            sp_idx_dir=getattr(args, 'sp_idx_dir', None),
            num_sp_tokens=num_sp_tokens,
        )
        nb_classes = 20
    elif args.data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(
            img_name_list_path=args.img_list,
            voc12_root=args.data_path,
            scales=getattr(args, "scales", [1.0, 0.75, 1.25]),
            train=is_train,
            gen_attn=gen_attn,
            transform=transform,
            num_sp_tokens=num_sp_tokens,
        )
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCODataset(
            img_name_list_path=args.img_list,
            coco_root=args.data_path,
            split=getattr(args, 'coco_split', 'train'),
            train=is_train,
            gen_attn=gen_attn,
            transform=transform,
            sp_idx_dir=getattr(args, 'sp_idx_dir', None),
            num_sp_tokens=num_sp_tokens,
        )
        nb_classes = 80  # COCO has 80 foreground classes
    elif args.data_set == 'COCOMS':
        dataset = COCODatasetMS(
            img_name_list_path=args.img_list,
            coco_root=args.data_path,
            split=getattr(args, 'coco_split', 'train'),
            scales=getattr(args, "scales", [1.0, 0.75, 1.25]),
            train=is_train,
            gen_attn=gen_attn,
            transform=transform,
            num_sp_tokens=num_sp_tokens,
        )
        nb_classes = 80  # COCO has 80 foreground classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    # 在 CAM 生成模式下（包括单张和批量），跳过 Resize/CenterCrop 以保持原始尺寸
    if resize_im and not args.gen_attention_maps and not getattr(args, 'gen_attention_maps_batch', False):
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_spix_label(img_np: np.ndarray, num_components: int = 196):
    # 使用传入的参数而不是硬编码
    N_COMPONENTS = num_components
    COMPACTNESS = 10

    try:
        # 使用 fast_slic 生成超像素
        slic = SlicAvx2(num_components=N_COMPONENTS, compactness=COMPACTNESS)
        sp_label = slic.iterate(img_np)  # np.ndarray
        
        # 验证和修正superpixel标签
        # 确保标签值在有效范围内 [0, N_COMPONENTS-1]
        max_label = sp_label.max()
        min_label = sp_label.min()
        
        # 如果标签超出范围，进行重新映射
        if max_label >= N_COMPONENTS or min_label < 0:
            unique_labels = np.unique(sp_label)
            # 创建映射字典，将标签重新映射到 [0, min(len(unique_labels)-1, N_COMPONENTS-1)]
            max_allowed = min(len(unique_labels) - 1, N_COMPONENTS - 1)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels[:max_allowed + 1])}
            
            # 对于超出范围的标签，映射到最后一个有效标签
            for label in unique_labels[max_allowed + 1:]:
                label_map[label] = max_allowed
            
            # 应用映射
            sp_label_fixed = np.zeros_like(sp_label, dtype=np.int32)
            for old_label, new_label in label_map.items():
                sp_label_fixed[sp_label == old_label] = new_label
            sp_label = sp_label_fixed
        
        # 确保数据类型正确
        sp_label = sp_label.astype(np.int32)
        
        # 最终验证
        assert sp_label.min() >= 0, f"Superpixel labels contain negative values: {sp_label.min()}"
        assert sp_label.max() < N_COMPONENTS, f"Superpixel labels exceed maximum: {sp_label.max()} >= {N_COMPONENTS}"
        
        return sp_label
        
    except Exception as e:
        print(f"Error in get_spix_label: {e}")
        print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"Image min/max: {img_np.min()}/{img_np.max()}")
        # 降级到简单的网格标签作为备选方案
        h, w = img_np.shape[:2]
        grid_h = int(np.sqrt(N_COMPONENTS * h / w))
        grid_w = N_COMPONENTS // grid_h
        sp_label = np.zeros((h, w), dtype=np.int32)
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * h // grid_h
                y_end = (i + 1) * h // grid_h
                x_start = j * w // grid_w
                x_end = (j + 1) * w // grid_w
                sp_label[y_start:y_end, x_start:x_end] = i * grid_w + j
        return sp_label
