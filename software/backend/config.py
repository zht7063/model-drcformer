"""Server-owned configuration. Secrets are read from the environment only."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = Path.home() / "projects" / "outputs" / "drcformer"
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class ModelSpec:
    """One resident DRCFormer checkpoint. num_sp_tokens must match its training run."""
    id: str
    checkpoint: Path
    num_sp_tokens: int = 196


def parse_models(value):
    """SEG_MODELS: JSON list of {"id", "checkpoint", "num_sp_tokens"?}; the first is the default."""
    try:
        entries = json.loads(value)
        specs = tuple(ModelSpec(str(e["id"]), Path(e["checkpoint"]).expanduser().resolve(),
                                int(e.get("num_sp_tokens", 196))) for e in entries)
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise ValueError("SEG_MODELS must be a JSON list of "
                         '{"id": ..., "checkpoint": ..., "num_sp_tokens": ...}') from error
    return specs


def resolve_gt_dirs(gt_dir: Path | None) -> tuple[Path, ...]:
    """Return mask lookup directories in priority order for a configured SEG_GT_DIR.

    Supports pointing directly at a mask folder (e.g. SegmentationClass or
    SegmentationClassAug) or at a VOC root containing those subdirectories,
    searching SegmentationClass before SegmentationClassAug.
    """
    if gt_dir is None:
        return ()
    subdirs = tuple(
        sub for name in ("SegmentationClass", "SegmentationClassAug")
        if (sub := gt_dir / name).is_dir()
    )
    return subdirs if subdirs else (gt_dir,)


@dataclass(frozen=True)
class Settings:
    models: tuple = (ModelSpec("drcformer", OUTPUTS / "default" / "checkpoint.pth"),)
    results_dir: Path = OUTPUTS / "server_results"
    gt_dir: Path | None = None
    api_key: str = field(default="", repr=False)
    device: str = "cuda:0"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 12_000_000
    # Longest side the model runs at; larger uploads are downscaled for inference only.
    max_inference_side: int = 512
    queue_capacity: int = 2
    result_ttl_seconds: int = 3600
    upload_timeout_seconds: int = 30

    def validate(self):
        if not self.api_key or len(self.api_key) < 16 or not self.api_key.isascii():
            raise ValueError("SEG_API_KEY must contain at least 16 ASCII characters")
        if self.device not in ("cuda", "cpu") and not (
                self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("SEG_DEVICE must be cuda, cuda:<index> or cpu")
        for name in ("max_upload_bytes", "max_pixels", "max_inference_side", "queue_capacity",
                     "result_ttl_seconds", "upload_timeout_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.models:
            raise ValueError("SEG_MODELS must list at least one model")
        ids = [spec.id for spec in self.models]
        if len(set(ids)) != len(ids) or not all(MODEL_ID.fullmatch(i) for i in ids):
            raise ValueError("model ids must be unique and match [a-z0-9][a-z0-9._-]{0,63}")
        if any(spec.num_sp_tokens <= 0 for spec in self.models):
            raise ValueError("num_sp_tokens must be positive")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for name in cls.__dataclass_fields__:
            value = os.environ.get("SEG_" + name.upper())
            if value is None:
                continue
            if name == "models":
                values[name] = parse_models(value)
            elif name == "results_dir":
                if not value:
                    raise ValueError("SEG_RESULTS_DIR must not be empty")
                values[name] = Path(value).expanduser().resolve()
            elif name == "gt_dir":
                values[name] = Path(value).expanduser().resolve() if value.strip() else None
            elif isinstance(getattr(defaults, name), int):
                values[name] = int(value)
            else:
                values[name] = value
        settings = cls(**values)
        settings.validate()
        return settings
