// Client-side mirror of software/backend/schemas.py InferenceParams; the server stays authoritative.
export const ATTENTION_TYPES = ["fused", "patchcam", "mct"];

export const FALLBACK_PARAMS = {
  scales: [1.0, 0.75, 1.25],
  flip: true,
  attention_type: "fused",
  layer_index: 12,
  patch_attn_refine: true,
  cls_threshold: 0.5,
  bg_threshold: 0.39,
  crf: false,
  crf_iters: 5,
};

export function formatScales(scales) {
  return scales.map((s) => String(s)).join(", ");
}

/** Turns the form state (scales as text) into the `params` JSON object, or throws a Chinese message. */
export function buildParams(form) {
  const parts = String(form.scales).split(/[,，\s]+/).filter(Boolean);
  const scales = parts.map(Number);
  if (!scales.length || scales.length > 4 || scales.some((s) => !Number.isFinite(s)))
    throw new Error("多尺度须为 1–4 个数字，用逗号分隔。");
  if (scales.some((s) => s < 0.25 || s > 2))
    throw new Error("每个尺度须在 0.25–2.0 之间。");
  const layer = Number(form.layer_index);
  if (!Number.isInteger(layer) || layer < 1 || layer > 12)
    throw new Error("注意力层数须为 1–12 的整数。");
  for (const [key, label] of [
    ["cls_threshold", "类别阈值"],
    ["bg_threshold", "背景阈值"],
  ]) {
    const v = Number(form[key]);
    if (form[key] === "" || !Number.isFinite(v) || v < 0 || v > 1)
      throw new Error(`${label}须在 0–1 之间。`);
  }
  const iters = Number(form.crf_iters);
  if (!Number.isInteger(iters) || iters < 1 || iters > 20)
    throw new Error("CRF 迭代次数须为 1–20 的整数。");
  if (!ATTENTION_TYPES.includes(form.attention_type))
    throw new Error("注意力类型无效。");
  return {
    scales,
    flip: Boolean(form.flip),
    attention_type: form.attention_type,
    layer_index: layer,
    patch_attn_refine: Boolean(form.patch_attn_refine),
    cls_threshold: Number(form.cls_threshold),
    bg_threshold: Number(form.bg_threshold),
    crf: Boolean(form.crf),
    crf_iters: iters,
  };
}

/** Limits come from /api/v1/models; missing limits leave validation to the server. */
export function checkImage(image, limits) {
  const formats = limits?.accepted_formats || ["image/jpeg", "image/png", "image/webp"];
  if (!formats.includes(image.mime)) throw new Error("请选择 JPEG、PNG 或 WebP 图片。");
  if (limits?.max_upload_bytes && image.bytes > limits.max_upload_bytes)
    throw new Error(
      `图片超过服务限制（${(limits.max_upload_bytes / 1048576).toFixed(1)} MB），请缩小图片。`,
    );
  if (limits?.max_pixels && image.width * image.height > limits.max_pixels)
    throw new Error(
      `图片超过服务像素限制（${limits.max_pixels.toLocaleString("zh-CN")} 像素），请降低分辨率。`,
    );
}

/** VOC/COCO bit-interleaved palette, identical to backend engine.palette(). */
export function color(id) {
  let r = 0,
    g = 0,
    b = 0;
  for (let i = 0; i < 8; i++) {
    r |= ((id >> 0) & 1) << (7 - i);
    g |= ((id >> 1) & 1) << (7 - i);
    b |= ((id >> 2) & 1) << (7 - i);
    id >>= 3;
  }
  return `rgb(${r},${g},${b})`;
}

export const ATTENTION_LABELS = { fused: "融合（fused）", patchcam: "PatchCAM", mct: "MCT" };

/**
 * Result views in display order. Each view is `{ id, label, panels: [{ label, url }] }`: one
 * panel is shown next to the original image, two or three panels replace it side by side.
 * Order: prediction outputs, pixel confidence, superpixel classes, SLIC, then per detected
 * class its CAM (before/after refinement when refined) and the three CAM types.
 */
export function resultViews(result, classNames = {}) {
  if (!result) return [];
  const one = (id, label, url) => ({ id, label, panels: [{ label, url }] });
  const views = [
    one("overlay_url", "叠加图", result.outputs.overlay_url),
    one("colored_url", "彩色分割", result.outputs.colored_url),
    one("mask_url", "类别掩码", result.outputs.mask_url),
  ];
  const seg = result.stages?.segmentation || {};
  if (seg.artifacts?.confidence_url)
    views.push(one("confidence", "像素置信度", seg.artifacts.confidence_url));
  if (seg.artifacts?.token_classes_url) {
    const view = one("token_classes", "超像素类别", seg.artifacts.token_classes_url);
    view.legend = (seg.data?.token_classes?.classes || []).filter((c) => c.id);
    views.push(view);
  }
  const slic = result.stages?.slic?.artifacts || {};
  if (slic.boundary_url) views.push(one("slic_boundary", "超像素边界", slic.boundary_url));
  if (slic.regions_url) views.push(one("slic_regions", "超像素区域", slic.regions_url));
  const cam = result.stages?.cam || {};
  const cams = cam.artifacts || {};
  const selected = cam.data?.attention_type;
  for (const [key, url] of Object.entries(cams)) {
    const id = Number(key.match(/^class_(\d+)_url$/)?.[1]);
    if (!id) continue; // raw_ and per-type entries are attached to this class below
    const name = classNames[id] ?? id;
    const raw = cams[`raw_class_${id}_url`];
    views.push(
      raw
        ? { id: `cam_${id}`, label: `注意力 · ${name} · 细化前后`, panels: [
            { label: `细化前`, url: raw }, { label: `细化后`, url }] }
        : one(`cam_${id}`, `注意力 · ${name}`, url),
    );
    const types = Object.keys(ATTENTION_LABELS)
      .map((kind) => ({ kind, url: kind === selected ? url : cams[`${kind}_class_${id}_url`] }))
      .filter((t) => t.url);
    if (selected && types.length > 1)
      views.push({
        id: `cam_types_${id}`,
        label: `注意力 · ${name} · 三种 CAM${cam.data?.refined ? "（均已细化）" : ""}`,
        panels: types.map((t) => ({
          label: ATTENTION_LABELS[t.kind] + (t.kind === selected ? "（当前）" : ""),
          url: t.url,
        })),
      });
  }
  return views;
}

/**
 * Image-level class scores, highest first: the top `limit` plus every detected class.
 * Older servers without `class_scores` fall back to the classes present in the mask.
 */
export function scoreBars(result, limit = 8) {
  const seg = result?.stages?.segmentation?.data || {};
  const detected = new Set((seg.detected_classes || []).map((c) => c.id));
  const all =
    seg.class_scores ||
    (result?.classes || []).filter((c) => c.score != null).map(({ id, name, score }) => ({ id, name, score }));
  const sorted = [...all].sort((a, b) => b.score - a.score);
  return sorted
    .filter((c, rank) => rank < limit || detected.has(c.id))
    .map((c) => ({ ...c, detected: detected.has(c.id) }));
}

const TIMING_LABELS = {
  resize: "缩放",
  superpixels: "超像素",
  forward: "模型前向",
  refine: "亲和力细化",
  postprocess: "后处理",
  crf: "CRF",
  save_outputs: "保存结果",
  slic: "超像素产物",
  cam: "注意力图产物",
  other: "其他",
};

const TIMING_ORDER = Object.keys(TIMING_LABELS);

/**
 * Per-stage server time in pipeline order (the Rust side re-sorts JSON object keys, so order is
 * imposed here); `other` is what the total spends outside them. Sub-millisecond stages are dropped.
 */
export function stageTimings(result) {
  if (!result) return [];
  const entries = Object.entries(result.stages?.segmentation?.data?.timings_ms || {});
  for (const name of ["slic", "cam"]) {
    const ms = result.stages?.[name]?.duration_ms;
    if (ms != null) entries.push([name, ms]);
  }
  if (!entries.length) return [];
  const other = result.duration_ms - entries.reduce((sum, [, ms]) => sum + ms, 0);
  if (other > 0) entries.push(["other", other]);
  const rank = (key) => (TIMING_ORDER.includes(key) ? TIMING_ORDER.indexOf(key) : TIMING_ORDER.length);
  return entries
    .filter(([, ms]) => ms >= 0.5)
    .sort(([a], [b]) => rank(a) - rank(b))
    .map(([key, ms]) => ({ key, label: TIMING_LABELS[key] ?? key, ms }));
}
