"""Resident DRCFormer checkpoints and the multi-scale CAM -> mask inference path.

Mirrors engine.generate_attention_maps_ms, minus everything that needs ground truth:
image-level labels are replaced by the model's own class scores (sigmoid > cls_threshold),
and the GT-overlap superpixel amplify/suppress step is skipped.
"""
import hashlib
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[2] / "src" / "drcformer"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from datasets import get_spix_label  # noqa: E402
from engine import JET_CMAP, refine_cam_with_spatch_affinity  # noqa: E402
from evaluation import COCO_CATEGORIES, VOC_CATEGORIES  # noqa: E402
from models import drcformer  # noqa: E402

from software.backend.schemas import InferenceParams  # noqa: E402

LOGGER = logging.getLogger(__name__)
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def palette(count=256):
    colors = np.zeros((count, 3), dtype=np.uint8)
    for index in range(count):
        value = index
        for shift in range(8):
            for channel in range(3):
                colors[index, channel] |= ((value >> channel) & 1) << (7 - shift)
            value >>= 3
    return colors


def save_regions(labels, image, directory, prefix):
    labels = labels.astype(np.int32)
    np.save(directory / f"{prefix}-labels.npy", labels, allow_pickle=False)
    Image.fromarray(palette()[labels % 256]).save(directory / f"{prefix}-regions.png")
    edges = np.zeros(labels.shape, dtype=bool)
    edges[1:] |= labels[1:] != labels[:-1]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    overlay = np.asarray(image).copy()
    overlay[edges] = [255, 255, 0]
    Image.fromarray(overlay).save(directory / f"{prefix}-boundary.png")
    return {"labels": f"{prefix}-labels.npy", "regions": f"{prefix}-regions.png",
            "boundary": f"{prefix}-boundary.png"}


def load_model(spec, device):
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {k.removeprefix("module."): v for k, v in state.items()}
    num_classes = state["cls_token"].shape[1]
    if num_classes not in (20, 80):
        raise ValueError(f"{spec.id}: expected a VOC (20) or COCO (80) checkpoint, got {num_classes}")
    model = drcformer(num_classes=num_classes, num_sp_tokens=spec.num_sp_tokens)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def crf_labels(image, probs, iters):
    """Multi-class DenseCRF over the candidate channels; pairwise terms as in engine._init_densecrf_2d."""
    import pydensecrf.densecrf as dcrf
    channels, height, width = probs.shape
    probs = np.clip(probs, 1e-5, None)
    probs = probs / probs.sum(0, keepdims=True)
    crf = dcrf.DenseCRF2D(width, height, channels)
    crf.setUnaryEnergy(np.ascontiguousarray(-np.log(probs).reshape(channels, -1), dtype=np.float32))
    crf.addPairwiseBilateral(sxy=80, srgb=3, rgbim=np.array(image, dtype=np.uint8, order="C"), compat=3)
    crf.addPairwiseGaussian(sxy=3, compat=5)
    return np.argmax(np.array(crf.inference(iters)).reshape(channels, height, width), 0)


class LoadedModel:
    def __init__(self, spec, device):
        self.spec = spec
        self.model = load_model(spec, device)
        self.checkpoint_sha256 = sha256(spec.checkpoint)
        foreground = self.model.num_classes
        self.dataset = "VOC12" if foreground == 20 else "COCO"
        self.classes = VOC_CATEGORIES if foreground == 20 else COCO_CATEGORIES

    def metadata(self):
        return {"id": self.spec.id, "checkpoint_sha256": self.checkpoint_sha256,
                "supervision": "image-level-labels", "dataset": self.dataset,
                "num_sp_tokens": self.spec.num_sp_tokens,
                "classes": [{"id": i, "name": name} for i, name in enumerate(self.classes)],
                "stages": {"segmentation": "prediction", "slic": "auxiliary", "cam": "auxiliary"},
                "default_parameters": InferenceParams().model_dump()}


class SegmentationEngine:
    def __init__(self, specs, device="cuda", max_inference_side=512):
        self.device = device
        self.max_inference_side = max_inference_side
        self.models = {spec.id: LoadedModel(spec, device) for spec in specs}
        self.default_model = specs[0].id

    def metadata(self):
        return [loaded.metadata() for loaded in self.models.values()]

    @torch.inference_mode()
    def cams(self, loaded, working, spix, params):
        """Sum of per-scale (and flipped) CAMs at working size, plus mean image-level class scores."""
        width, height = working.size
        total, scores = None, []
        autocast = torch.autocast("cuda", enabled=self.device.startswith("cuda"))
        for scale in params.scales:
            size = (max(16, round(width * scale)), max(16, round(height * scale)))
            image = torch.from_numpy(np.asarray(working.resize(size, Image.Resampling.BICUBIC)).copy())
            image = (image.permute(2, 0, 1).float() / 255 - MEAN) / STD
            sp = F.interpolate(torch.from_numpy(spix)[None, None].float(), size=size[::-1],
                               mode="nearest")[0].long()
            for flipped in ((False, True) if params.flip else (False,)):
                x, s = image[None], sp[None]
                if flipped:
                    x, s = x.flip(-1), s.flip(-1)
                x, s = x.to(self.device), s.to(self.device)
                with autocast:
                    logits, cam, patch_attn = loaded.model(
                        x, spix_label=s, return_att=True, n_layers=params.layer_index,
                        attention_type=params.attention_type)
                logits, cam = logits.float(), cam.float()
                if params.patch_attn_refine:
                    cam = refine_cam_with_spatch_affinity(
                        cam, s[:, 0], patch_attn.sum(0).float().squeeze(0), output=logits,
                        pos_thr=0.5, alpha=0.85, steps=10, is_logit=True)
                cam = F.interpolate(cam, size=(height, width), mode="bilinear", align_corners=False)
                cam = cam.flip(-1) if flipped else cam
                total = cam if total is None else total + cam
                scores.append(torch.sigmoid(logits))
        return total[0], torch.stack(scores).mean(0)[0]

    def run(self, image, include, directory, model_id=None, params=None):
        params = params or InferenceParams()
        loaded = self.models[model_id or self.default_model]
        started = time.perf_counter()
        working = image
        if max(image.size) > self.max_inference_side:
            working = image.copy()
            working.thumbnail((self.max_inference_side,) * 2, Image.Resampling.BICUBIC)
        spix = get_spix_label(np.asarray(working), num_components=loaded.spec.num_sp_tokens)
        spix = spix.astype(np.int64)
        cam, scores = self.cams(loaded, working, spix, params)

        detected = [int(i) for i in torch.nonzero(scores > params.cls_threshold).flatten()]
        normalized = {}
        for index in detected:
            channel = cam[index]
            normalized[index] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        # Channel 0 is background at bg_threshold; channel k is foreground class detected[k-1].
        probs = torch.stack([torch.full_like(cam[0], params.bg_threshold)]
                            + [normalized[i] for i in detected])
        lookup = np.array([0] + [i + 1 for i in detected], dtype=np.uint8)
        if params.crf and detected:
            picked = crf_labels(np.asarray(working), probs.cpu().numpy(), params.crf_iters)
            mask = lookup[picked]
            if working.size != image.size:
                mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
        else:
            if working.size != image.size:
                probs = F.interpolate(probs[None], size=image.size[::-1], mode="bilinear",
                                      align_corners=False)[0]
            mask = lookup[probs.argmax(0).cpu().numpy()]

        colors = palette()[mask]
        Image.fromarray(mask).save(directory / "mask.png")
        Image.fromarray(colors).save(directory / "colored.png")
        Image.blend(image, Image.fromarray(colors), .45).save(directory / "overlay.png")
        class_scores = {i + 1: float(scores[i]) for i in range(len(scores))}
        stages = {"segmentation": {
            "status": "completed", "used_for_prediction": True,
            "method": "drcformer_multiscale_cam" + ("_crf" if params.crf else ""),
            "parameters": {**params.model_dump(), "inference_size": list(working.size)},
            "data": {"detected_classes": [
                {"id": i + 1, "name": loaded.classes[i + 1], "score": class_scores[i + 1]}
                for i in detected]},
            "duration_ms": round((time.perf_counter() - started) * 1000, 2)}}
        for name in ("slic", "cam"):
            if name not in include:
                continue
            start = time.perf_counter()
            stage = {"used_for_prediction": name == "slic"}
            try:
                if name == "slic":
                    labels = np.asarray(Image.fromarray(spix.astype(np.int32)).resize(
                        image.size, Image.Resampling.NEAREST))
                    stage.update(method="fast_slic_avx2", source="uploaded_image",
                                 parameters={"num_components": loaded.spec.num_sp_tokens,
                                             "compactness": 10,
                                             "working_size": list(working.size)},
                                 data={"region_count": int(np.unique(labels).size)},
                                 artifacts=save_regions(labels, image, directory, "slic"))
                else:
                    artifacts = {}
                    for index in detected:
                        heat = normalized[index][None, None]
                        heat = F.interpolate(heat, size=image.size[::-1], mode="bilinear",
                                             align_corners=False)[0, 0].clamp(0, 1).cpu().numpy()
                        jet = (JET_CMAP(heat)[..., :3] * 255).astype(np.uint8)
                        Image.blend(image, Image.fromarray(jet), .5).save(directory / f"cam-{index + 1}.png")
                        artifacts[f"class_{index + 1}"] = f"cam-{index + 1}.png"
                    stage.update(method=f"drcformer_{params.attention_type}_cam",
                                 source="detected_classes",
                                 data={"class_ids": [i + 1 for i in detected]}, artifacts=artifacts)
                stage["status"] = "completed"
            except Exception:
                LOGGER.exception("auxiliary stage failed: %s", name)
                stage.update(status="failed", reason="analysis_failed")
                stage.pop("artifacts", None)
            stage["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            stages[name] = stage
        counts = np.bincount(mask.ravel(), minlength=len(loaded.classes))
        return {"model_id": loaded.spec.id, "checkpoint_sha256": loaded.checkpoint_sha256,
                "supervision": "image-level-labels", "parameters": params.model_dump(),
                "width": image.width, "height": image.height,
                "outputs": {"mask": "mask.png", "colored": "colored.png", "overlay": "overlay.png"},
                "classes": [{"id": i, "name": loaded.classes[i], "pixels": int(count),
                             "area_ratio": float(count / mask.size),
                             **({"score": class_scores[i]} if i else {})}
                            for i, count in enumerate(counts) if count],
                "stages": stages}
