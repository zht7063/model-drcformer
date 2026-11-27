import { test } from "node:test";
import assert from "node:assert/strict";
import { buildParams, checkImage, color, resultViews, scoreBars, stageTimings } from "../src/params.js";

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
  // A server without the three-type CAMs gives one CAM view per class, beside the original.
  assert.equal(views[4].panels.length, 1);
});

test("resultViews adds confidence, superpixel classes, refined CAM pairs and the three CAM types", () => {
  const base = "/api/v1/results/" + "b".repeat(32) + "/";
  const views = resultViews(
    {
      outputs: { mask_url: base + "mask.png", colored_url: base + "colored.png", overlay_url: base + "overlay.png" },
      stages: {
        segmentation: {
          artifacts: { confidence_url: base + "confidence.png", token_classes_url: base + "token-classes.png" },
          data: { token_classes: { classes: [{ id: 0, superpixels: 150 }, { id: 12, name: "dog", superpixels: 30 }] } },
        },
        cam: {
          // patchcam is the selected type, so its CAM is class_<id>_url and has no patchcam_ entry.
          data: { attention_type: "patchcam", refined: true },
          artifacts: {
            class_12_url: base + "cam-12.png",
            raw_class_12_url: base + "cam-raw-12.png",
            fused_class_12_url: base + "cam-fused-12.png",
            mct_class_12_url: base + "cam-mct-12.png",
          },
        },
      },
    },
    { 12: "dog" },
  );
  const file = (url) => url.split("/").pop();
  assert.deepEqual(
    views.map((v) => [v.label, v.panels.map((p) => [p.label, file(p.url)])]),
    [
      ["叠加图", [["叠加图", "overlay.png"]]],
      ["彩色分割", [["彩色分割", "colored.png"]]],
      ["类别掩码", [["类别掩码", "mask.png"]]],
      ["像素置信度", [["像素置信度", "confidence.png"]]],
      ["超像素类别", [["超像素类别", "token-classes.png"]]],
      ["注意力 · dog · 细化前后", [["细化前", "cam-raw-12.png"], ["细化后", "cam-12.png"]]],
      [
        "注意力 · dog · 三种 CAM（均已细化）",
        [["融合（fused）", "cam-fused-12.png"], ["PatchCAM（当前）", "cam-12.png"], ["MCT", "cam-mct-12.png"]],
      ],
    ],
  );
  // Background superpixels are not listed in the superpixel-class legend.
  assert.deepEqual(views[4].legend.map((c) => c.id), [12]);
});

test("scoreBars keeps the top scores plus every detected class, highest first", () => {
  const class_scores = Array.from({ length: 20 }, (_, i) => ({ id: i + 1, name: `c${i + 1}`, score: (i + 1) / 40 }));
  // A detected class can rank below the cut-off when the threshold is low; it must still be shown.
  const bars = scoreBars(
    { stages: { segmentation: { data: { class_scores, detected_classes: [{ id: 20 }, { id: 2 }] } } } },
    3,
  );
  assert.deepEqual(
    bars.map((b) => [b.id, b.detected]),
    [[20, true], [19, false], [18, false], [2, true]],
  );
});

test("stageTimings lists stages in pipeline order and attributes the remainder to other", () => {
  const timings = stageTimings({
    duration_ms: 1000,
    stages: {
      // Keys arrive alphabetically sorted after the Rust round trip; resize took no time.
      segmentation: { data: { timings_ms: { forward: 600, refine: 50, resize: 0, superpixels: 100 } } },
      cam: { duration_ms: 150 },
    },
  });
  assert.deepEqual(
    timings.map((t) => [t.label, t.ms]),
    [["超像素", 100], ["模型前向", 600], ["亲和力细化", 50], ["注意力图产物", 150], ["其他", 100]],
  );
});
