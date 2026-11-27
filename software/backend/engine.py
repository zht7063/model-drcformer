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


ATTENTION_TYPES = ("fused", "patchcam", "mct")
LOW_CONFIDENCE = 0.1  # top1 - top2 margin below which a pixel counts as uncertain


def minmax(channel):
    return (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)


def save_heat(heat, image, path):
    """Upsample a [0, 1] working-size map to the image, save it as a JET overlay, return it."""
    heat = F.interpolate(heat[None, None].float(), size=image.size[::-1], mode="bilinear",
                         align_corners=False)[0, 0].clamp(0, 1).cpu().numpy()
    jet = (JET_CMAP(heat)[..., :3] * 255).astype(np.uint8)
    Image.blend(image, Image.fromarray(jet), .5).save(path)
    return heat


def save_token_classes(token_cls, token_mask, spix, image, path, classes):
    """Paint each superpixel with its own argmax foreground class when that logit is positive.

    token_cls (T, K) are the head_token logits of superpixel label t; empty superpixels (masked
    tokens) and superpixels whose best logit is <= 0 stay background. Returns per-class counts.
    """
    tokens = token_cls.shape[0]
    best, index = token_cls.max(1)
    label = torch.where((best > 0) & token_mask, index + 1, torch.zeros_like(index)).cpu().numpy()
    regions = np.clip(spix, 0, tokens - 1).astype(np.int32)
    regions = np.asarray(Image.fromarray(regions).resize(image.size, Image.Resampling.NEAREST))
    per_pixel = label[regions].astype(np.uint8)
    painted = np.asarray(Image.blend(image, Image.fromarray(palette()[per_pixel]), .55)).copy()
    edges = np.zeros(regions.shape, dtype=bool)
    edges[1:] |= regions[1:] != regions[:-1]
    edges[:, 1:] |= regions[:, 1:] != regions[:, :-1]
    painted[edges] = [255, 255, 255]
    Image.fromarray(painted).save(path)
    valid = label[token_mask.cpu().numpy()]
    counts = np.bincount(valid, minlength=len(classes))
    return {"measure": "argmax_token_logit_gt_0", "superpixels": int(valid.size),
            "classes": [{"id": i, "name": classes[i], "superpixels": int(n)}
                        for i, n in enumerate(counts) if n]}


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

    def sync(self):
        """Wait for queued GPU work so wall-clock stage timings are not shifted to a later stage."""
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def cams(self, loaded, working, spix, params, all_types=False):
        """Run every scale (and flip) branch once and average/sum its outputs at working size.

        Returns a dict: `cam` (K,H,W) summed selected CAM, refined when patch_attn_refine;
        `raw` the same before refinement (None when refinement is off); `types` per attention
        type summed CAMs (all three when all_types, else only the selected one), each refined
        like `cam`; `scores`, `cls_scores`, `patch_scores` (K,) mean sigmoid of the final,
        class-token and superpixel-token logits; `token_cls` (T,K) mean per-superpixel class
        logits with `token_mask` (T,) for non-empty superpixels; `refine_ms`.
        """
        width, height = working.size
        kinds = ATTENTION_TYPES if all_types else (params.attention_type,)
        totals = dict.fromkeys(kinds)
        raw_total, refine_ms = None, 0.0
        scores, cls_scores, patch_scores, tokens = [], [], [], []
        autocast = torch.autocast("cuda", enabled=self.device.startswith("cuda"))

        def to_working(cam, flipped):
            cam = F.interpolate(cam, size=(height, width), mode="bilinear", align_corners=False)
            return cam.flip(-1) if flipped else cam

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
                    out = loaded.model(
                        x, spix_label=s, return_att=True, return_extras=True,
                        n_layers=params.layer_index, attention_type=params.attention_type)
                logits = out["logits"].float()
                affinity = out["patch_attn"].sum(0).float().squeeze(0)
                for kind in kinds:
                    cam = (out["cams"] if kind == params.attention_type
                           else out["cams_all"][kind]).float()
                    if params.patch_attn_refine:
                        if kind == params.attention_type:
                            raw = to_working(cam, flipped)
                            raw_total = raw if raw_total is None else raw_total + raw
                        self.sync()
                        start = time.perf_counter()
                        cam = refine_cam_with_spatch_affinity(
                            cam, s[:, 0], affinity, output=logits,
                            pos_thr=0.5, alpha=0.85, steps=10, is_logit=True)
                        self.sync()
                        refine_ms += (time.perf_counter() - start) * 1000
                    cam = to_working(cam, flipped)
                    totals[kind] = cam if totals[kind] is None else totals[kind] + cam
                scores.append(torch.sigmoid(logits))
                cls_scores.append(torch.sigmoid(out["cls_logits"].float()))
                patch_scores.append(torch.sigmoid(out["patch_logits"].float()))
                # Token t is superpixel label t in every branch (resizing and flipping keep ids).
                tokens.append(out["token_cls"].float())
                token_mask = out["token_mask"][0]
        self.sync()
        mean = lambda values: torch.stack(values).mean(0)[0]
        return {"cam": totals[params.attention_type][0],
                "raw": None if raw_total is None else raw_total[0],
                "types": {kind: total[0] for kind, total in totals.items()},
                "scores": mean(scores), "cls_scores": mean(cls_scores),
                "patch_scores": mean(patch_scores), "token_cls": mean(tokens),
                "token_mask": token_mask, "refine_ms": refine_ms}

    def run(self, image, include, directory, model_id=None, params=None):
        params = params or InferenceParams()
        loaded = self.models[model_id or self.default_model]
        started = time.perf_counter()
        timings = {}

        def lap(name, since):
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000, 2)
            return now

        working = image
        if max(image.size) > self.max_inference_side:
            working = image.copy()
            working.thumbnail((self.max_inference_side,) * 2, Image.Resampling.BICUBIC)
        mark = lap("resize", started)
        spix = get_spix_label(np.asarray(working), num_components=loaded.spec.num_sp_tokens)
        spix = spix.astype(np.int64)
        mark = lap("superpixels", mark)
        out = self.cams(loaded, working, spix, params, all_types="cam" in include)
        cam, raw_cam, scores = out["cam"], out["raw"], out["scores"]
        mark = lap("forward", mark)
        timings["forward"] = round(timings["forward"] - out["refine_ms"], 2)
        if params.patch_attn_refine:
            timings["refine"] = round(out["refine_ms"], 2)

        detected = [int(i) for i in torch.nonzero(scores > params.cls_threshold).flatten()]
        normalized = {index: minmax(cam[index]) for index in detected}
        # Channel 0 is background at bg_threshold; channel k is foreground class detected[k-1].
        probs = torch.stack([torch.full_like(cam[0], params.bg_threshold)]
                            + [normalized[i] for i in detected])
        # Pixel confidence: winning channel minus runner-up, before CRF; undefined with one channel.
        margin = None
        if detected:
            top = probs.topk(2, dim=0).values
            margin = (top[0] - top[1]).clamp(0, 1)
        lookup = np.array([0] + [i + 1 for i in detected], dtype=np.uint8)
        mark = lap("postprocess", mark)
        if params.crf and detected:
            picked = crf_labels(np.asarray(working), probs.cpu().numpy(), params.crf_iters)
            mask = lookup[picked]
            if working.size != image.size:
                mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
            mark = lap("crf", mark)
        else:
            if working.size != image.size:
                probs = F.interpolate(probs[None], size=image.size[::-1], mode="bilinear",
                                      align_corners=False)[0]
            mask = lookup[probs.argmax(0).cpu().numpy()]
            timings["postprocess"] = round(timings["postprocess"]
                                           + (time.perf_counter() - mark) * 1000, 2)
            mark = time.perf_counter()

        colors = palette()[mask]
        Image.fromarray(mask).save(directory / "mask.png")
        Image.fromarray(colors).save(directory / "colored.png")
        Image.blend(image, Image.fromarray(colors), .45).save(directory / "overlay.png")
        data, artifacts = {}, {}
        if margin is not None:
            confidence = save_heat(margin, image, directory / "confidence.png")
            artifacts["confidence"] = "confidence.png"
            data["confidence"] = {"measure": "top1_minus_top2", "mean": float(confidence.mean()),
                                  "low_threshold": LOW_CONFIDENCE,
                                  "low_ratio": float((confidence < LOW_CONFIDENCE).mean())}
        data["token_classes"] = save_token_classes(
            out["token_cls"], out["token_mask"], spix, image, directory / "token-classes.png",
            loaded.classes)
        artifacts["token_classes"] = "token-classes.png"
        mark = lap("save_outputs", mark)
        class_scores = {i + 1: float(scores[i]) for i in range(len(scores))}
        data.update(
            detected_classes=[{"id": i + 1, "name": loaded.classes[i + 1],
                               "score": class_scores[i + 1]} for i in detected],
            class_scores=[{"id": i + 1, "name": loaded.classes[i + 1], "score": class_scores[i + 1],
                           "cls_score": float(out["cls_scores"][i]),
                           "patch_score": float(out["patch_scores"][i])}
                          for i in range(len(scores))],
            timings_ms=timings)
        stages = {"segmentation": {
            "status": "completed", "used_for_prediction": True,
            "method": "drcformer_multiscale_cam" + ("_crf" if params.crf else ""),
            "parameters": {**params.model_dump(), "inference_size": list(working.size)},
            "data": data, **({"artifacts": artifacts} if artifacts else {}),
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
                        save_heat(normalized[index], image, directory / f"cam-{index + 1}.png")
                        artifacts[f"class_{index + 1}"] = f"cam-{index + 1}.png"
                        if raw_cam is not None:
                            save_heat(minmax(raw_cam[index]), image,
                                      directory / f"cam-raw-{index + 1}.png")
                            artifacts[f"raw_class_{index + 1}"] = f"cam-raw-{index + 1}.png"
                        for kind, typed in out["types"].items():
                            if kind == params.attention_type:
                                continue  # identical to cam-<id>.png
                            save_heat(minmax(typed[index]), image,
                                      directory / f"cam-{kind}-{index + 1}.png")
                            artifacts[f"{kind}_class_{index + 1}"] = f"cam-{kind}-{index + 1}.png"
                    stage.update(method=f"drcformer_{params.attention_type}_cam",
                                 source="detected_classes",
                                 data={"class_ids": [i + 1 for i in detected],
                                       "attention_type": params.attention_type,
                                       "types": list(out["types"]),
                                       "refined": raw_cam is not None}, artifacts=artifacts)
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
