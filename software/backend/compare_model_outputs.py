"""Check that the return_extras change to models.py leaves CAM generation unchanged.

Loads the same checkpoint into the models.py of a reference git revision (default origin/main,
before return_extras existed) and into the current models.py, runs both on the same images and
superpixels, and compares element by element:

1. current tuple return vs reference tuple return (logits, cams, patch_attn) for each attention type;
2. current return_extras dict: logits/cams/patch_attn vs its own tuple return, and
   cams_all[type] vs a separate tuple run with attention_type=type;
3. cls_logits/patch_logits average back to logits.

Run from the repository root on the GPU server, e.g. inside tmux:
  CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/envs/py310/bin/python \
      -m software.backend.compare_model_outputs \
      --checkpoint ~/projects/outputs/drcformer/<实验ID>/checkpoint.pth \
      --images ~/projects/voc2012/JPEGImages/2007_000032.jpg ~/projects/voc2012/JPEGImages/2007_000039.jpg
Exits 0 and prints "all outputs identical" when every comparison matches within --atol.
On CUDA, torch_scatter's mean pooling uses atomic adds, so even the unchanged model differs
by float rounding between runs; the script prints that run-to-run baseline for the reference
model so a max|diff| at that level can be told apart from a real change. On CPU every
comparison is expected to be exactly 0.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "drcformer"
sys.path.insert(0, str(SOURCE))

from datasets import get_spix_label  # noqa: E402
import models as current_models  # noqa: E402

TYPES = ("fused", "patchcam", "mct")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def reference_module(rev):
    code = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:src/drcformer/models.py"],
                          check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp(prefix="models-ref-")) / "models_reference.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("models_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # timm's @register_model looks the module up here
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # re-registering the same model names only warns
        spec.loader.exec_module(module)
    return module


def load(module, checkpoint, num_sp_tokens, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = module.drcformer(num_classes=state["cls_token"].shape[1], num_sp_tokens=num_sp_tokens)
    model.load_state_dict(state, strict=True)  # extras add no parameters: strict must pass
    return model.to(device).eval().requires_grad_(False)


def inputs(path, size, num_sp_tokens):
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    spix = torch.from_numpy(get_spix_label(np.asarray(image), num_components=num_sp_tokens)
                            .astype(np.int64))
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    return ((x - MEAN) / STD)[None], spix[None, None]


class Report:
    def __init__(self, atol):
        self.failures, self.atol = 0, atol

    def same(self, name, a, b):
        diff = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        ok = a.shape == b.shape and diff <= self.atol
        self.failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: shape {tuple(a.shape)} max|diff| {diff:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--num-sp-tokens", type=int, default=196)
    parser.add_argument("--reference-rev", default="origin/main",
                        help="git revision whose models.py predates return_extras")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512, help="longest image side")
    parser.add_argument("--layer-index", type=int, default=12)
    parser.add_argument("--atol", type=float, default=None,
                        help="max allowed |diff|; default 0 on CPU, 1e-4 on CUDA")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(args.checkpoint).expanduser()
    reference = load(reference_module(args.reference_rev), checkpoint, args.num_sp_tokens, args.device)
    current = load(current_models, checkpoint, args.num_sp_tokens, args.device)
    atol = args.atol if args.atol is not None else (0.0 if args.device == "cpu" else 1e-4)
    report = Report(atol)
    print(f"tolerance max|diff| <= {atol:g}")
    with torch.inference_mode():
        for path in args.images:
            print(path)
            x, spix = inputs(Path(path).expanduser(), args.size, args.num_sp_tokens)
            x, spix = x.to(args.device), spix.to(args.device)
            common = dict(spix_label=spix, return_att=True, n_layers=args.layer_index)
            again = [reference(x, attention_type="fused", **common) for _ in range(2)]
            baseline = max((a.float() - b.float()).abs().max().item() for a, b in zip(*again))
            print(f"  reference run-to-run baseline max|diff| {baseline:.3g}")
            for kind in TYPES:
                old = reference(x, attention_type=kind, **common)
                new = current(x, attention_type=kind, **common)
                extras = current(x, attention_type=kind, return_extras=True, **common)
                for name, a, b in zip(("logits", "cams", "patch_attn"), new, old):
                    report.same(f"[{kind}] tuple {name} vs {args.reference_rev}", a, b)
                for name, a in zip(("logits", "cams", "patch_attn"), new):
                    report.same(f"[{kind}] dict {name} vs tuple", extras[name], a)
                report.same(f"[{kind}] (cls_logits + patch_logits) / 2 vs logits",
                            (extras["cls_logits"] + extras["patch_logits"]) / 2, extras["logits"])
                for other in TYPES:
                    single = current(x, attention_type=other, **common)[1]
                    report.same(f"[{kind}] cams_all[{other}] vs attention_type={other}",
                                extras["cams_all"][other], single)
                tokens, mask = extras["token_cls"], extras["token_mask"]
                if kind == "fused":
                    print(f"  token_cls {tuple(tokens.shape)}, non-empty superpixels {int(mask.sum())}")
                # token_cls is head_token's output: its ReLU, painted by superpixel, is patchcam.
                labels = spix[0, 0].clamp(max=int(mask[0].sum()) - 1)
                report.same(f"[{kind}] ReLU(token_cls)[superpixel] vs patchcam",
                            F.relu(tokens[0]).T[:, labels][None], extras["cams_all"]["patchcam"])
    if report.failures:
        print(f"{report.failures} comparisons differ")
        sys.exit(1)
    print("all outputs identical")


if __name__ == "__main__":
    main()
