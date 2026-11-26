"""Real checkpoint smoke check; run inside tmux from the repository root."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from software.backend.config import ModelSpec
from software.backend.engine import SegmentationEngine
from software.backend.schemas import InferenceParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-sp-tokens", type=int, default=196)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crf", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        parser.error("real inference must run inside tmux")
    torch.set_num_threads(2)
    spec = ModelSpec("smoke", Path(args.checkpoint).expanduser().resolve(), args.num_sp_tokens)
    engine = SegmentationEngine((spec,), args.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    with tempfile.TemporaryDirectory(prefix="seg-engine-") as temp:
        directory = Path(temp)
        result = engine.run(image, {"segmentation", "slic", "cam"}, directory,
                            params=InferenceParams(crf=args.crf))
        mask = np.asarray(Image.open(directory / "mask.png"))
        assert mask.shape == (image.height, image.width)
        assert mask.dtype == np.uint8 and mask.max() < len(engine.models["smoke"].classes)
        for stage in ("segmentation", "slic", "cam"):
            assert result["stages"][stage]["status"] == "completed", stage
        result["test"] = "passed"
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
