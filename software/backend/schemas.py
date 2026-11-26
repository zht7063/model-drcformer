"""Public HTTP contracts; internal paths and secrets never belong here."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: Literal["alive", "ready", "not_ready"]


class InferenceParams(BaseModel):
    """Per-request inference options. Defaults follow main.py's CAM generation defaults."""
    model_config = ConfigDict(extra="forbid")

    scales: list[float] = Field(default=[1.0, 0.75, 1.25], min_length=1, max_length=4)
    flip: bool = True
    attention_type: Literal["fused", "patchcam", "mct"] = "fused"
    layer_index: int = Field(default=12, ge=1, le=12)
    patch_attn_refine: bool = True
    cls_threshold: float = Field(default=0.5, ge=0, le=1)
    bg_threshold: float = Field(default=0.39, ge=0, le=1)
    crf: bool = False
    crf_iters: int = Field(default=5, ge=1, le=20)

    @field_validator("scales")
    @classmethod
    def bounded_scales(cls, scales):
        if any(not 0.25 <= s <= 2.0 for s in scales):
            raise ValueError("each scale must be between 0.25 and 2.0")
        return scales


class ClassLabel(BaseModel):
    id: int = Field(ge=0, le=80)
    name: str


class ClassStatistics(ClassLabel):
    pixels: int = Field(ge=1)
    area_ratio: float = Field(gt=0, le=1)
    score: float | None = None


class ServiceLimits(BaseModel):
    max_upload_bytes: int
    max_pixels: int
    max_inference_side: int
    result_ttl_seconds: int
    upload_timeout_seconds: int
    queue_capacity: int
    accepted_formats: list[str] = ["image/jpeg", "image/png", "image/webp"]


class ModelMetadata(BaseModel):
    id: str
    checkpoint_sha256: str
    supervision: str
    dataset: Literal["VOC12", "COCO"]
    num_sp_tokens: int
    classes: list[ClassLabel]
    stages: dict[str, Literal["prediction", "auxiliary", "unavailable"]]
    default_parameters: InferenceParams


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelMetadata]
    limits: ServiceLimits


class StageResult(BaseModel):
    status: Literal["completed", "unavailable", "failed"]
    used_for_prediction: bool
    duration_ms: float = Field(ge=0)
    method: str | None = None
    source: str | None = None
    reason: str | None = None
    parameters: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    artifacts: dict[str, str] | None = None


class PredictionOutputs(BaseModel):
    mask_url: str
    colored_url: str
    overlay_url: str


class SegmentResponse(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    model_id: str
    checkpoint_sha256: str
    supervision: str
    parameters: InferenceParams
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: float = Field(ge=0)
    expires_at: float
    storage_format: Literal["segmentation-service-v1"]
    outputs: PredictionOutputs
    classes: list[ClassStatistics]
    stages: dict[str, StageResult]
