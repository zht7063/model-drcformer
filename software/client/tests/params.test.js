import { test } from "node:test";
import assert from "node:assert/strict";
import { buildParams, checkImage, color, resultViews } from "../src/params.js";

const form = {
  scales: "1.0, 0.75，1.25",
  flip: true,
  attention_type: "patchcam",
  layer_index: 10,
  patch_attn_refine: false,
  cls_threshold: 0.6,
  bg_threshold: 0.3,
  crf: true,
  crf_iters: 8,
};

test("buildParams emits exactly the backend InferenceParams fields", () => {
  const params = buildParams(form);
  assert.deepEqual(params, { ...form, scales: [1.0, 0.75, 1.25] });
});

test("buildParams rejects values the backend would answer with 400 invalid_params", () => {
  for (const bad of [
    { scales: "1, 3" },
    { scales: "1,1,1,1,1" },
    { scales: "" },
    { layer_index: 13 },
    { layer_index: 2.5 },
    { cls_threshold: 1.2 },
    { bg_threshold: "" },
    { crf_iters: 0 },
    { attention_type: "gradcam" },
  ])
    assert.throws(() => buildParams({ ...form, ...bad }), Error, JSON.stringify(bad));
});

test("checkImage enforces server limits", () => {
  const limits = { max_upload_bytes: 1000, max_pixels: 100 };
  assert.doesNotThrow(() => checkImage({ mime: "image/png", bytes: 999, width: 10, height: 10 }, limits));
  assert.throws(() => checkImage({ mime: "image/gif", bytes: 1, width: 1, height: 1 }, limits));
  assert.throws(() => checkImage({ mime: "image/png", bytes: 1001, width: 1, height: 1 }, limits));
  assert.throws(() => checkImage({ mime: "image/png", bytes: 1, width: 11, height: 10 }, limits));
});

test("palette matches the VOC colors used by the backend", () => {
  assert.equal(color(0), "rgb(0,0,0)");
  assert.equal(color(1), "rgb(128,0,0)");
  assert.equal(color(12), "rgb(64,0,128)"); // dog
  assert.equal(color(15), "rgb(192,128,128)"); // person
});

test("resultViews lists outputs, SLIC and per-class CAM in display order", () => {
  const base = "/api/v1/results/" + "a".repeat(32) + "/";
  const views = resultViews(
    {
      outputs: { mask_url: base + "mask.png", colored_url: base + "colored.png", overlay_url: base + "overlay.png" },
      stages: {
        slic: { artifacts: { labels_url: base + "slic-labels.npy", boundary_url: base + "slic-boundary.png" } },
        cam: { artifacts: { class_12_url: base + "cam-12.png" } },
      },
    },
    { 12: "dog" },
  );
  assert.deepEqual(
    views.map((v) => v.label),
    ["叠加图", "彩色分割", "类别掩码", "超像素边界", "注意力 · dog"],
  );
});
