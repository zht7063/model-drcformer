"""
离线导出 SLIC 超像素索引图（与训练时在线生成一致）。

输出为 .npy（每像素一个 superpixel id，shape=(H,W), dtype=int32）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger

from drcformer.datasets import get_spix_label


def _read_id_list(img_list: Path, split: str) -> list[str]:
    if img_list.is_dir():
        list_file = img_list / ("train_aug_id.txt" if split == "train" else "val_id.txt")
    else:
        list_file = img_list
    ids = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export SLIC superpixel index maps (.npy) for VOC2012")
    p.add_argument("--data-path", type=str, required=True, help="VOC root containing JPEGImages/")
    p.add_argument(
        "--img-list",
        type=str,
        required=True,
        help="Path to voc12 directory (contains train_aug_id.txt/val_id.txt) or a specific *_id.txt file",
    )
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for npy index maps")
    p.add_argument("--num-components", type=int, default=196, help="Number of SLIC components (default: 196)")
    p.add_argument("--resume", action="store_true", default=True, help="Skip images whose npy already exists")
    p.add_argument("--max-count", type=int, default=None, help="Process at most N images (for quick runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    voc_root = Path(args.data_path)
    img_list = Path(args.img_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _read_id_list(img_list, args.split)
    if args.max_count is not None:
        ids = ids[: args.max_count]

    logger.info(
        f"SLIC export: split={args.split}, num={len(ids)}, num_components={args.num_components}, out={out_dir}"
    )

    n_ok, n_skip, n_err = 0, 0, 0
    for i, name in enumerate(ids, start=1):
        out_path = out_dir / f"{name}.npy"
        if args.resume and out_path.exists():
            n_skip += 1
            continue

        img_path = voc_root / "JPEGImages" / f"{name}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.asarray(img)
            spix = get_spix_label(img_np, num_components=args.num_components).astype(np.int32)
            np.save(out_path, spix)
            n_ok += 1
        except Exception as e:
            logger.error(f"[{i}/{len(ids)}] failed on {name}: {e}")
            n_err += 1

    logger.info(f"Done. OK={n_ok}, SKIP={n_skip}, ERR={n_err}. Output -> {out_dir}")


if __name__ == "__main__":
    main()

