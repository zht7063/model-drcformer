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

/**
 * Result views in display order: prediction outputs, pixel confidence, SLIC, then one CAM per
 * detected class. A view with `compare` shows that image instead of the original on the left:
 * refined CAMs are paired with the same class before superpixel-affinity refinement.
 */
export function resultViews(result, classNames = {}) {
  if (!result) return [];
  const views = [
    { id: "overlay_url", label: "叠加图", url: result.outputs.overlay_url },
    { id: "colored_url", label: "彩色分割", url: result.outputs.colored_url },
    { id: "mask_url", label: "类别掩码", url: result.outputs.mask_url },
  ];
  const confidence = result.stages?.segmentation?.artifacts?.confidence_url;
  if (confidence) views.push({ id: "confidence", label: "像素置信度", url: confidence });
  const slic = result.stages?.slic?.artifacts || {};
  if (slic.boundary_url) views.push({ id: "slic_boundary", label: "超像素边界", url: slic.boundary_url });
  if (slic.regions_url) views.push({ id: "slic_regions", label: "超像素区域", url: slic.regions_url });
  const cams = result.stages?.cam?.artifacts || {};
  for (const [key, url] of Object.entries(cams)) {
    const id = Number(key.match(/^class_(\d+)_url$/)?.[1]);
    if (!id) continue; // raw_class_<id>_url entries are attached to their refined view below
    const name = classNames[id] ?? id;
    const raw = cams[`raw_class_${id}_url`];
    views.push(
      raw
        ? { id: `cam_${id}`, label: `注意力 · ${name}（细化后）`, url,
            compare: { label: `注意力 · ${name}（细化前）`, url: raw } }
        : { id: `cam_${id}`, label: `注意力 · ${name}`, url },
    );
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
