"""Validate server configuration/assets without loading a model or probing GPU."""
from importlib import util
from importlib.metadata import PackageNotFoundError, version
import json
import os

from software.backend.config import Settings, resolve_gt_dirs


def dist_version(candidates):
    for dist in candidates:
        try:
            return dist, version(dist)
        except PackageNotFoundError:
            continue
    raise SystemExit(f"Missing package metadata for: {', '.join(candidates)}")


def main():
    settings = Settings.from_env()
    for spec in settings.models:
        if not spec.checkpoint.is_file() or not os.access(spec.checkpoint, os.R_OK):
            raise SystemExit(f"Missing or unreadable checkpoint: {spec.id}: {spec.checkpoint}")
    if settings.gt_dir is not None:
        if not settings.gt_dir.is_dir() or not os.access(settings.gt_dir, os.R_OK):
            raise SystemExit(f"Missing or unreadable gt_dir: {settings.gt_dir}")
        for gt_subdir in resolve_gt_dirs(settings.gt_dir):
            if not gt_subdir.is_dir() or not os.access(gt_subdir, os.R_OK):
                raise SystemExit(f"Unreadable gt_dir subdirectory: {gt_subdir}")
    packages = {"fastapi": ("fastapi",), "starlette": ("starlette",), "uvicorn": ("uvicorn",),
                "python_multipart": ("python-multipart",), "PIL": ("Pillow",), "numpy": ("numpy",),
                "torch": ("torch",), "torch_scatter": ("torch-scatter",), "timm": ("timm",),
                "fast_slic": ("fast-slic",), "skimage": ("scikit-image",),
                "pydensecrf": ("pydensecrf", "pydensecrf2")}
    for module in packages:
        if util.find_spec(module) is None:
            raise SystemExit(f"Missing dependency: {module}")
    ancestor = settings.results_dir
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise SystemExit("Result directory is not writable")
    port = int(os.environ.get("SEG_PORT", "8000"))
    if not 1 <= port <= 65535:
        raise SystemExit("SEG_PORT must be between 1 and 65535")
    print(json.dumps({"status": "configuration_valid", "device": settings.device,
                      "gpu_tested": False, "model_loaded": False,
                      "models": {spec.id: {"checkpoint": str(spec.checkpoint),
                                           "num_sp_tokens": spec.num_sp_tokens}
                                 for spec in settings.models},
                      "results_dir": str(settings.results_dir),
                      "gt_dir": str(settings.gt_dir) if settings.gt_dir is not None else None,
                      "dependencies": dict(dist_version(dists) for dists in packages.values()),
                      "api_key": "configured (redacted)"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
