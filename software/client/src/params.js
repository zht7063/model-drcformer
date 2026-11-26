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

/** Result views in display order: prediction outputs, then SLIC, then one CAM per detected class. */
export function resultViews(result, classNames = {}) {
  if (!result) return [];
  const views = [
    ["overlay_url", "叠加图", result.outputs.overlay_url],
    ["colored_url", "彩色分割", result.outputs.colored_url],
    ["mask_url", "类别掩码", result.outputs.mask_url],
  ];
  const slic = result.stages?.slic?.artifacts || {};
  if (slic.boundary_url) views.push(["slic_boundary", "超像素边界", slic.boundary_url]);
  if (slic.regions_url) views.push(["slic_regions", "超像素区域", slic.regions_url]);
  for (const [key, url] of Object.entries(result.stages?.cam?.artifacts || {})) {
    const id = Number(key.match(/^class_(\d+)_url$/)?.[1]);
    views.push([`cam_${id}`, `注意力 · ${classNames[id] ?? id}`, url]);
  }
  return views.map(([id, label, url]) => ({ id, label, url }));
}
