"""Validate server configuration/assets without loading a model or probing GPU."""
from importlib import util
from importlib.metadata import version
import json
import os

from software.backend.config import Settings


def main():
    settings = Settings.from_env()
    for spec in settings.models:
        if not spec.checkpoint.is_file() or not os.access(spec.checkpoint, os.R_OK):
            raise SystemExit(f"Missing or unreadable checkpoint: {spec.id}: {spec.checkpoint}")
    packages = {"fastapi": "fastapi", "starlette": "starlette", "uvicorn": "uvicorn",
                "python_multipart": "python-multipart", "PIL": "Pillow", "numpy": "numpy",
                "torch": "torch", "torch_scatter": "torch-scatter", "timm": "timm",
                "fast_slic": "fast-slic", "skimage": "scikit-image", "pydensecrf": "pydensecrf"}
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
                      "dependencies": {dist: version(dist) for dist in packages.values()},
                      "api_key": "configured (redacted)"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
