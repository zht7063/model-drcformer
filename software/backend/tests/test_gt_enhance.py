"""Unit and API tests for VOC superpixel-GT CAM enhancement/suppression."""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from software.backend import check_config
from software.backend.app import create_app
from software.backend.config import ModelSpec, Settings, resolve_gt_dirs
from software.backend.schemas import InferenceParams, SegmentResponse


def _import_backend_engine():
    """Import software.backend.engine, stubbing heavy optional model deps only when absent."""
    stubbed = {}

    def _ensure_stub(mod_name, **attrs):
        top = mod_name.split(".")[0]
        if importlib.util.find_spec(top) is not None or mod_name in sys.modules:
            return sys.modules.get(mod_name)
        mod = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        stubbed[mod_name] = mod
        return mod

    jet = lambda arr: np.stack([arr, np.zeros_like(arr), 1.0 - arr, np.ones_like(arr)], axis=-1)

    _ensure_stub("fast_slic")
    _ensure_stub("fast_slic.avx2", SlicAvx2=type("SlicAvx2", (), {
        "__init__": lambda self, *a, **k: None,
        "iterate": lambda self, img_np: np.zeros(img_np.shape[:2], dtype=np.int32),
    }))
    _ensure_stub("skimage")
    _ensure_stub("skimage.io")
    _ensure_stub("skimage.transform", rescale=lambda *a, **k: None)
    _ensure_stub("skimage.segmentation", slic=lambda *a, **k: None, find_boundaries=lambda *a, **k: None)
    _ensure_stub("sklearn")
    _ensure_stub("sklearn.metrics", average_precision_score=lambda *a, **k: 0.0)
    _ensure_stub("torchvision")
    _ensure_stub("torchvision.datasets")
    _ensure_stub("torchvision.datasets.folder", ImageFolder=object, default_loader=lambda *a, **k: None)
    _ensure_stub(
        "torchvision.transforms",
        Compose=object, RandomResizedCrop=object, RandomCrop=object,
        Resize=object, ToTensor=object, Normalize=object,
        InterpolationMode=SimpleNamespace(BICUBIC=3),
    )
    _ensure_stub("torchvision.transforms.functional")
    _ensure_stub("timm")
    _ensure_stub("timm.data", create_transform=lambda *a, **k: None)
    _ensure_stub("timm.data.constants", IMAGENET_DEFAULT_MEAN=(0.485, 0.456, 0.406), IMAGENET_DEFAULT_STD=(0.229, 0.224, 0.225))
    _ensure_stub("timm.models", register_model=lambda fn: fn)
    _ensure_stub("timm.models.vision_transformer", VisionTransformer=object, _cfg=lambda *a, **k: {}, LayerScale=object)
    _ensure_stub("timm.layers", Mlp=object, DropPath=object, get_act_layer=lambda *a, **k: None, get_norm_layer=lambda *a, **k: None, use_fused_attn=lambda *a, **k: False)
    _ensure_stub("timm.models.layers", DropPath=object, to_2tuple=lambda x: (x, x), trunc_normal_=lambda *a, **k: None)
    _ensure_stub("timm.models.helpers", build_model_with_cfg=lambda *a, **k: None)
    _ensure_stub("timm.models.registry", register_model=lambda fn: fn)
    _ensure_stub("torch_scatter", scatter_max=None, scatter_mean=None, scatter_sum=None, scatter_softmax=None, scatter_min=None)
    _ensure_stub("torch_scatter.composite", scatter_std=None)
    _ensure_stub("einops", repeat=lambda *a, **k: None, rearrange=lambda *a, **k: None)
    _ensure_stub("gdown")
    _ensure_stub("loguru", logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))
    _ensure_stub("pandas")
    _ensure_stub("tqdm", tqdm=lambda x, *a, **k: x)
    _ensure_stub("matplotlib")
    _ensure_stub("matplotlib.cm", get_cmap=lambda name: jet)
    _ensure_stub("matplotlib.pyplot", get_cmap=lambda name: jet)

    # Link parent -> submodule attributes for dotted stubs.
    for full, mod in stubbed.items():
        if "." in full:
            parent, child = full.rsplit(".", 1)
            if parent in stubbed:
                setattr(stubbed[parent], child, mod)

    sys.modules.update(stubbed)
    import software.backend.engine as backend_engine
    return backend_engine


backend_engine = _import_backend_engine()


def _write_palette_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    palette_data = [0] * 768
    for idx in range(256):
        palette_data[idx * 3 : idx * 3 + 3] = [idx, idx, idx]
    img.putpalette(palette_data)
    img.save(path)


def _jpeg_bytes(size=(16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_settings_gt_dir_and_lookup_order(monkeypatch, tmp_path):
    monkeypatch.setenv("SEG_API_KEY", "a" * 20)
    monkeypatch.delenv("SEG_GT_DIR", raising=False)
    assert Settings.from_env().gt_dir is None

    monkeypatch.setenv("SEG_GT_DIR", "   ")
    assert Settings.from_env().gt_dir is None

    voc_root = tmp_path / "voc2012"
    seg_cls = voc_root / "SegmentationClass"
    seg_aug = voc_root / "SegmentationClassAug"
    seg_cls.mkdir(parents=True)
    seg_aug.mkdir(parents=True)

    monkeypatch.setenv("SEG_GT_DIR", str(voc_root))
    settings = Settings.from_env()
    assert settings.gt_dir == voc_root.resolve()
    assert resolve_gt_dirs(settings.gt_dir) == (seg_cls.resolve(), seg_aug.resolve())

    # Direct mask folder returns itself without requiring subdirectories.
    assert resolve_gt_dirs(seg_aug) == (seg_aug,)

    # VOC root falls back to SegmentationClassAug if SegmentationClass is missing.
    seg_cls.rmdir()
    assert resolve_gt_dirs(voc_root) == (seg_aug,)


def test_check_config_gt_dir_and_pydensecrf2(monkeypatch, tmp_path, capsys):
    ckpt = tmp_path / "checkpoint.pth"
    ckpt.write_bytes(b"stub")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    missing_gt = tmp_path / "does_not_exist"

    monkeypatch.setenv("SEG_API_KEY", "a" * 20)
    monkeypatch.setenv("SEG_MODELS", json.dumps([{"id": "drcformer", "checkpoint": str(ckpt)}]))
    monkeypatch.setenv("SEG_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("SEG_GT_DIR", str(missing_gt))

    with pytest.raises(SystemExit, match="gt_dir"):
        check_config.main()

    gt_dir = tmp_path / "voc_gt"
    gt_dir.mkdir()
    monkeypatch.setenv("SEG_GT_DIR", str(gt_dir))

    def fake_find_spec(name):
        return object()

    def fake_version(dist):
        if dist == "pydensecrf":
            raise check_config.PackageNotFoundError(dist)
        if dist == "pydensecrf2":
            return "1.0.0"
        return "0.0.1"

    monkeypatch.setattr(check_config.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(check_config, "version", fake_version)
    check_config.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "configuration_valid"
    assert payload["gt_dir"] == str(gt_dir.resolve())
    assert payload["dependencies"]["pydensecrf2"] == "1.0.0"


def test_inference_params_backward_compatibility_with_client_v012():
    legacy_payload = json.dumps({
        "scales": [1.0, 0.75, 1.25],
        "flip": True,
        "attention_type": "fused",
        "layer_index": 12,
        "patch_attn_refine": True,
        "cls_threshold": 0.5,
        "bg_threshold": 0.39,
        "crf": False,
        "crf_iters": 5,
    })
    params = InferenceParams.model_validate_json(legacy_payload)
    assert params.cam_gt_enhance is True
    assert params.cam_sp_gamma == pytest.approx(0.45)
    assert params.cam_sp_theta == pytest.approx(0.38)
    assert params.cam_amp_alpha == pytest.approx(0.06)
    assert params.cam_amp_beta == pytest.approx(0.08)

    empty_params = InferenceParams.model_validate_json("{}")
    assert empty_params == params

    with pytest.raises(Exception):
        InferenceParams.model_validate_json('{"unknown_field": 1}')


def test_filename_matching_and_traversal_protection(tmp_path):
    voc_root = tmp_path / "voc2012"
    seg_cls = voc_root / "SegmentationClass"
    seg_aug = voc_root / "SegmentationClassAug"
    _write_palette_mask(seg_cls / "2010_000270.png", np.ones((4, 4), dtype=np.uint8))
    _write_palette_mask(seg_aug / "2010_000270.png", np.full((4, 4), 2, dtype=np.uint8))
    _write_palette_mask(seg_aug / "2010_000271.png", np.full((4, 4), 3, dtype=np.uint8))
    # Also plant a mask outside gt_dir to prove traversal cannot reach it.
    _write_palette_mask(tmp_path / "2010_000270.png", np.full((4, 4), 9, dtype=np.uint8))

    # Standard VOC filename hits SegmentationClass first, then SegmentationClassAug.
    hit = backend_engine.find_voc_gt_mask(voc_root, "2010_000270.jpg")
    assert hit == ("2010_000270", (seg_cls / "2010_000270.png").resolve())

    hit_aug = backend_engine.find_voc_gt_mask(voc_root, "2010_000271.jpg")
    assert hit_aug == ("2010_000271", (seg_aug / "2010_000271.png").resolve())

    # Normal path prefix is stripped safely.
    hit_prefixed = backend_engine.find_voc_gt_mask(voc_root, "/Users/test/2010_000270.jpg")
    assert hit_prefixed == ("2010_000270", (seg_cls / "2010_000270.png").resolve())

    # Normal non-VOC photo misses cleanly.
    assert backend_engine.find_voc_gt_mask(voc_root, "cat.jpg") is None

    # Path traversal and malformed filenames are rejected as misses without raising.
    for bad_name in (
        "../2010_000270.jpg",
        "subdir/../../2010_000270.jpg",
        "..\\2010_000270.jpg",
        "2010_000270.tar.gz",
        "2010 000270.jpg",
        ".2010_000270.jpg",
        "",
        None,
    ):
        assert backend_engine.find_voc_gt_mask(voc_root, bad_name) is None


def test_superpixel_gt_enhancement_math_and_ignore_255(tmp_path):
    # 4 superpixels (0, 1, 2, 3), each of size 10 pixels (4 rows x 10 cols).
    spix = torch.tensor([
        [0] * 10,
        [1] * 10,
        [2] * 10,
        [3] * 10,
    ], dtype=torch.long)
    # Class index 0 -> seg_id = 1.
    # sp 0: 5 pixels of seg_id=1 -> ratio=0.50 >= gamma(0.45) -> * (1 + alpha) = 1.06
    # sp 1: 4 pixels of seg_id=1 + 6 pixels of ignore(255) -> ratio=0.40 in (theta, gamma) -> * 1.0
    # sp 2: 3 pixels of seg_id=1 -> ratio=0.30 <= theta(0.38) -> * (1 - beta) = 0.92
    # sp 3: 10 pixels of ignore(255) -> ratio=0.0 <= theta(0.38) -> * (1 - beta) = 0.92
    gt_np = np.zeros((4, 10), dtype=np.uint8)
    gt_np[0, :5] = 1
    gt_np[1, :4] = 1
    gt_np[1, 4:] = 255
    gt_np[2, :3] = 1
    gt_np[3, :] = 255

    mask_path = tmp_path / "2010_000270.png"
    _write_palette_mask(mask_path, gt_np)
    gt_mask = backend_engine.load_voc_gt_mask(mask_path)
    assert gt_mask is not None
    assert int(gt_mask[3, 0]) == 255

    cam = torch.ones((20, 4, 10), dtype=torch.float32) * 2.0
    enhanced, eff = backend_engine.apply_superpixel_gt_enhancement(
        cam, spix, gt_mask, active_classes=[0], gamma=0.45, theta=0.38, alpha=0.06, beta=0.08
    )
    assert eff == {"gamma": 0.45, "theta": 0.38, "alpha": 0.06, "beta": 0.08}
    assert torch.allclose(enhanced[0, 0], torch.full((10,), 2.0 * 1.06))
    assert torch.allclose(enhanced[0, 1], torch.full((10,), 2.0))
    assert torch.allclose(enhanced[0, 2], torch.full((10,), 2.0 * 0.92))
    assert torch.allclose(enhanced[0, 3], torch.full((10,), 2.0 * 0.92))
    # Inactive class 1 remains untouched.
    assert torch.allclose(enhanced[1], torch.full((4, 10), 2.0))

    # Swapping fallback when theta > gamma:
    _, swapped_eff = backend_engine.apply_superpixel_gt_enhancement(
        cam, spix, gt_mask, active_classes=[0], gamma=0.30, theta=0.50, alpha=0.10, beta=0.20
    )
    assert swapped_eff["theta"] == pytest.approx(0.30)
    assert swapped_eff["gamma"] == pytest.approx(0.50)


def _make_stub_engine(gt_dir: Path | None, dataset: str = "VOC12"):
    engine = object.__new__(backend_engine.SegmentationEngine)
    engine.device = "cpu"
    engine.max_inference_side = 512
    engine.gt_dir = gt_dir.resolve() if gt_dir is not None else None
    num_classes = 20 if dataset == "VOC12" else 80
    classes = (
        ["background"] + [f"voc_{i}" for i in range(1, 21)]
        if dataset == "VOC12"
        else ["background"] + [f"coco_{i}" for i in range(1, 81)]
    )
    loaded = SimpleNamespace(
        spec=SimpleNamespace(id="drcformer", num_sp_tokens=4),
        checkpoint_sha256="a" * 64,
        dataset=dataset,
        classes=classes,
        metadata=lambda: {
            "id": "drcformer",
            "checkpoint_sha256": "a" * 64,
            "supervision": "image-level-labels",
            "dataset": dataset,
            "num_sp_tokens": 4,
            "classes": [{"id": i, "name": name} for i, name in enumerate(classes)],
            "stages": {"segmentation": "prediction", "slic": "auxiliary", "cam": "auxiliary"},
            "default_parameters": InferenceParams().model_dump(),
        },
    )
    engine.models = {"drcformer": loaded}
    engine.default_model = "drcformer"

    def fake_cams(loaded_model, working, spix, params, all_types=False):
        h, w = working.size[1], working.size[0]
        cam = torch.zeros((num_classes, h, w), dtype=torch.float32)
        # spix 0 (rows 0..5): 0.42 -> below bg_threshold=0.45 before enhance, > 0.45 after 1.5x amp
        # spix 1 (rows 6..11): 0.48 -> above bg_threshold=0.45 before enhance, < 0.45 after 0.5x suppr
        # spix 2 (rows 12..15): anchor min=0.0 and max=1.0 in neutral (theta, gamma) superpixel
        cam[0, :6, :] = 0.42
        cam[0, 6:12, :] = 0.48
        cam[0, 12:, 0] = 1.0
        scores = torch.zeros((num_classes,), dtype=torch.float32)
        scores[0] = 0.9
        return {
            "cam": cam,
            "raw": cam.clone(),
            "types": {k: cam.clone() for k in (("fused", "patchcam", "mct") if all_types else (params.attention_type,))},
            "scores": scores,
            "cls_scores": scores.clone(),
            "patch_scores": scores.clone(),
            "token_cls": torch.zeros((4, num_classes), dtype=torch.float32),
            "token_mask": torch.ones((4,), dtype=torch.bool),
            "refine_ms": 0.0,
        }

    engine.cams = fake_cams
    return engine


def test_engine_run_gt_enhance_conditions_and_mask_consistency(monkeypatch, tmp_path):
    # spix 0: rows 0..5, spix 1: rows 6..11, spix 2: rows 12..15.
    def fake_spix(img_np, num_components=196):
        sp = np.zeros(img_np.shape[:2], dtype=np.int32)
        sp[6:12, :] = 1
        sp[12:, :] = 2
        return sp

    monkeypatch.setattr(backend_engine, "get_spix_label", fake_spix)

    gt_dir = tmp_path / "SegmentationClass"
    gt_arr = np.zeros((16, 16), dtype=np.uint8)
    gt_arr[:6, :] = 1      # spix 0 -> ratio 1.0 >= gamma -> amplified
    # spix 1 -> ratio 0.0 <= theta -> suppressed
    gt_arr[12:14, :6] = 1  # spix 2 -> 12/64? Wait: 4*16=64 pixels -> 26/64 = 0.40625 in (0.38, 0.45)
    gt_arr[12, :] = 1      # 16 pixels
    gt_arr[13, :10] = 1    # +10 = 26 pixels -> ratio = 26/64 = 0.40625 (neutral!)
    _write_palette_mask(gt_dir / "2010_000270.png", gt_arr)

    image = Image.new("RGB", (16, 16), (100, 100, 100))

    # 1. Triggered on VOC12 when gt_dir configured, cam_gt_enhance=True, and filename hits.
    engine_voc = _make_stub_engine(gt_dir, dataset="VOC12")
    out_hit = tmp_path / "out_hit"
    out_hit.mkdir()
    params_strong = InferenceParams(bg_threshold=0.45, cam_amp_alpha=0.5, cam_amp_beta=0.5)
    res_hit = engine_voc.run(image, {"segmentation", "cam"}, out_hit, params=params_strong, filename="2010_000270.jpg")
    assert res_hit["stages"]["cam"]["data"]["gt_enhanced"] is True
    assert res_hit["stages"]["cam"]["data"]["gt_matched_id"] == "2010_000270"
    assert res_hit["stages"]["cam"]["parameters"]["gt_enhanced"] is True
    assert res_hit["stages"]["cam"]["parameters"]["gamma"] == pytest.approx(0.45)

    # 2. Not triggered when cam_gt_enhance=False; mask differs in both amplified and suppressed regions.
    out_disabled = tmp_path / "out_disabled"
    out_disabled.mkdir()
    params_off = InferenceParams(bg_threshold=0.45, cam_gt_enhance=False, cam_amp_alpha=0.5, cam_amp_beta=0.5)
    res_off = engine_voc.run(image, {"segmentation", "cam"}, out_disabled, params=params_off, filename="2010_000270.jpg")
    assert res_off["stages"]["cam"]["data"]["gt_enhanced"] is False
    assert res_off["stages"]["cam"]["data"]["gt_matched_id"] is None

    mask_hit = np.asarray(Image.open(out_hit / "mask.png"))
    mask_off = np.asarray(Image.open(out_disabled / "mask.png"))
    assert np.all(mask_off[:6, :] == 0) and np.all(mask_hit[:6, :] == 1)
    assert np.all(mask_off[6:12, :] == 1) and np.all(mask_hit[6:12, :] == 0)

    # Exported CAM heatmaps (cam-1.png and alternate types) also reflect enhancement.
    cam_hit_img = np.asarray(Image.open(out_hit / "cam-1.png"))
    cam_off_img = np.asarray(Image.open(out_disabled / "cam-1.png"))
    assert not np.array_equal(cam_hit_img, cam_off_img)
    mct_hit_img = np.asarray(Image.open(out_hit / "cam-mct-1.png"))
    mct_off_img = np.asarray(Image.open(out_disabled / "cam-mct-1.png"))
    assert not np.array_equal(mct_hit_img, mct_off_img)

    # CRF path also consumes the enhanced CAM probabilities.
    seen_crf_probs = []
    monkeypatch.setattr(
        backend_engine,
        "crf_labels",
        lambda img, probs, iters: (seen_crf_probs.append(probs.copy()), np.argmax(probs, 0))[1],
    )
    out_crf = tmp_path / "out_crf"
    out_crf.mkdir()
    params_crf = InferenceParams(bg_threshold=0.45, cam_amp_alpha=0.5, cam_amp_beta=0.5, crf=True, crf_iters=2)
    engine_voc.run(image, {"segmentation", "cam"}, out_crf, params=params_crf, filename="2010_000270.jpg")
    assert len(seen_crf_probs) == 1
    assert seen_crf_probs[0][1, 0, 0] == pytest.approx(0.42 * 1.5)
    assert seen_crf_probs[0][1, 6, 0] == pytest.approx(0.48 * 0.5)

    # 3. Not triggered when gt_dir is unconfigured.
    engine_no_gt = _make_stub_engine(None, dataset="VOC12")
    out_no_gt = tmp_path / "out_no_gt"
    out_no_gt.mkdir()
    res_no_gt = engine_no_gt.run(image, {"segmentation", "cam"}, out_no_gt, params=params_strong, filename="2010_000270.jpg")
    assert res_no_gt["stages"]["cam"]["data"]["gt_enhanced"] is False
    assert res_no_gt["stages"]["cam"]["data"]["gt_matched_id"] is None

    # 4. Not triggered when loaded.dataset != "VOC12".
    engine_coco = _make_stub_engine(gt_dir, dataset="COCO")
    out_coco = tmp_path / "out_coco"
    out_coco.mkdir()
    res_coco = engine_coco.run(image, {"segmentation", "cam"}, out_coco, params=params_strong, filename="2010_000270.jpg")
    assert res_coco["stages"]["cam"]["data"]["gt_enhanced"] is False
    assert res_coco["stages"]["cam"]["data"]["gt_matched_id"] is None

    # 5. Not triggered for non-matching filename or traversal filename.
    for fname in ("cat.jpg", "../2010_000270.jpg"):
        out_miss = tmp_path / f"out_miss_{abs(hash(fname))}"
        out_miss.mkdir()
        res_miss = engine_voc.run(image, {"segmentation", "cam"}, out_miss, params=params_strong, filename=fname)
        assert res_miss["stages"]["cam"]["data"]["gt_enhanced"] is False
        assert res_miss["stages"]["cam"]["data"]["gt_matched_id"] is None


def test_segment_endpoint_passes_filename_and_preserves_v012_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backend_engine,
        "get_spix_label",
        lambda img_np, num_components=196: np.zeros(img_np.shape[:2], dtype=np.int32),
    )
    gt_dir = tmp_path / "gt"
    _write_palette_mask(gt_dir / "2010_000270.png", np.ones((16, 16), dtype=np.uint8))

    settings = Settings(
        models=(ModelSpec("drcformer", tmp_path / "ckpt.pth"),),
        results_dir=tmp_path / "results",
        gt_dir=gt_dir,
        api_key="secret-api-key-1234",
    )
    engine = _make_stub_engine(gt_dir, dataset="VOC12")
    app = create_app(settings=settings, engine_factory=lambda: engine)
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    legacy_params = json.dumps({
        "scales": [1.0, 0.75, 1.25],
        "flip": True,
        "attention_type": "fused",
        "layer_index": 12,
        "patch_attn_refine": True,
        "cls_threshold": 0.5,
        "bg_threshold": 0.39,
        "crf": False,
        "crf_iters": 5,
    })

    with TestClient(app) as client:
        models_resp = client.get("/api/v1/models", headers=headers)
        assert models_resp.status_code == 200
        assert models_resp.json()["models"][0]["default_parameters"]["cam_gt_enhance"] is True

        # VOC filename hit with legacy client-v0.1.2 params
        resp_hit = client.post(
            "/api/v1/segment",
            headers=headers,
            files={"image": ("2010_000270.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"include": '["segmentation","cam"]', "model": "drcformer", "params": legacy_params},
        )
        assert resp_hit.status_code == 200
        body_hit = resp_hit.json()
        SegmentResponse.model_validate(body_hit)
        assert body_hit["stages"]["cam"]["data"]["gt_enhanced"] is True
        assert body_hit["stages"]["cam"]["data"]["gt_matched_id"] == "2010_000270"

        # Traversal filename and ordinary photo both succeed silently with gt_enhanced=False
        for fname in ("../2010_000270.jpg", "cat.jpg"):
            resp_miss = client.post(
                "/api/v1/segment",
                headers=headers,
                files={"image": (fname, _jpeg_bytes(), "image/jpeg")},
                data={"include": '["segmentation","cam"]', "model": "drcformer", "params": legacy_params},
            )
            assert resp_miss.status_code == 200
            body_miss = resp_miss.json()
            SegmentResponse.model_validate(body_miss)
            assert body_miss["stages"]["cam"]["data"]["gt_enhanced"] is False
            assert body_miss["stages"]["cam"]["data"]["gt_matched_id"] is None
