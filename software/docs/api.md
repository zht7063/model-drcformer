# HTTP API v1

所有 `/api/v1/*` 接口使用 `Authorization: Bearer <SEG_API_KEY>`，结果下载使用相同凭据。健康检查公开。服务为单租户，持有该 key 的调用方共享结果访问权限。

## POST /api/v1/segment

`multipart/form-data` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image` | 文件，必填 | 单帧 JPEG、PNG、WebP；以真实解码格式为准 |
| `include` | JSON 数组字符串，可选 | 默认 `["segmentation"]`；可加 `slic`、`cam` |
| `model` | 字符串，可选 | `/api/v1/models` 返回的模型 id；默认 `default_model`（`SEG_MODELS` 第一项） |
| `params` | JSON 对象字符串，可选 | 推理参数，见下表；未给出的字段取默认值，未知字段返回 400 |

### 推理参数 `params`

默认值与 `src/drcformer/main.py` 生成 CAM 时的默认值一致。

| 字段 | 默认 | 范围 | 含义 |
| --- | --- | --- | --- |
| `scales` | `[1.0, 0.75, 1.25]` | 1–4 个，各 0.25–2.0 | 多尺度推理，CAM 求和 |
| `flip` | `true` | | 每个尺度追加水平翻转分支 |
| `attention_type` | `fused` | `fused`/`patchcam`/`mct` | CAM 类型，同 `--attention-type` |
| `layer_index` | `12` | 1–12 | 取最后若干层注意力，同 `--layer-index` |
| `patch_attn_refine` | `true` | | 超像素亲和力 RWR 细化（alpha 0.85，10 步） |
| `cls_threshold` | `0.5` | 0–1 | 图像级 sigmoid 分数超过阈值的类别才参与分割 |
| `bg_threshold` | `0.39` | 0–1 | 背景通道常数，同 `--seed-bg-thr` |
| `crf` | `false` | | 多类 DenseCRF 后处理（成对项参数同 `engine._init_densecrf_2d`） |
| `crf_iters` | `5` | 1–20 | CRF 迭代次数 |
| `cam_gt_enhance` | `true` | | 服务端配置 `SEG_GT_DIR` 且上传 VOC2012 同名图片时，启用超像素-GT 增强/抑制 |
| `cam_sp_gamma` | `0.45` | 0–1 | 超像素-GT 增强重叠上阈值 `gamma`（`ratio >= gamma` 乘以 `1 + alpha`） |
| `cam_sp_theta` | `0.38` | 0–1 | 超像素-GT 抑制重叠下阈值 `theta`（`ratio <= theta` 乘以 `1 - beta`；若 `theta > gamma` 自动交换） |
| `cam_amp_alpha` | `0.06` | 0–1 | 超像素-GT 增强比例 `alpha` |
| `cam_amp_beta` | `0.08` | 0–1 | 超像素-GT 抑制比例 `beta` |

推理流程对应 `generate_attention_maps_ms`：训练脚本使用的图像级 GT 标签由模型自己的类别分数（多尺度平均 sigmoid > `cls_threshold`）代替。超像素-GT 增强/抑制仅在同时满足以下条件时自动触发：服务端配置了 `SEG_GT_DIR`、`cam_gt_enhance` 为 `true`、当前模型 `dataset == "VOC12"`、且上传文件名 stem（严格校验 `^[A-Za-z0-9_-]+$`、拒绝路径穿越）在 `SEG_GT_DIR` 中命中 `<stem>.png`；未命中或未配置时静默跳过，不报错。每个检测到的类别 CAM（及对应增强结果）做 min-max 归一化，与背景常数一起取 argmax。超像素由 fast_slic 在推理图上生成，数量为该模型的 `num_sp_tokens`（须与训练时一致，由服务端配置，不接受请求覆盖）。最长边超过 `max_inference_side` 的图片先缩小再推理，结果插值回原图尺寸。图片先做 EXIF 方向校正。

主要响应字段：

| 字段 | 含义 |
| --- | --- |
| `request_id` | 32 位随机十六进制 ID |
| `model_id` / `checkpoint_sha256` | 所用模型 id 与实际权重指纹 |
| `supervision` | `image-level-labels` |
| `parameters` | 本次生效的完整推理参数 |
| `width` / `height` | 校正方向后的图像尺寸 |
| `duration_ms` | 不含网络与排队的处理时间 |
| `expires_at` | 结果过期 UTC Unix 秒时间戳 |
| `outputs` | `mask_url`、`colored_url`、`overlay_url` |
| `classes` | 非零像素类别的 id、name、pixels、area_ratio；前景类另含图像级 `score` |
| `stages` | segmentation 和本次请求的分析节点 |

| 节点 | 返回 |
| --- | --- |
| segmentation | `parameters`（含实际推理尺寸 `inference_size`）；`data.detected_classes`（超过阈值的类别及分数）；`data.class_scores`（全部前景类别的图像级分数）；`data.timings_ms`（`resize`、`superpixels`、`forward`、`refine`、`postprocess`、`crf`、`save_outputs`，毫秒，未执行的阶段不出现）；有检测类别时另有 `data.confidence`（`measure=top1_minus_top2`、`mean`、`low_threshold`、`low_ratio`）与 `artifacts.confidence_url`（CRF 前的像素置信度 JET 叠加图）；`class_scores[]` 每项另含 `cls_score`（类 token 路）、`patch_score`（超像素 token 路），均为多尺度平均 sigmoid，`score` 为两路 logit 平均后的 sigmoid；`data.token_classes`（`measure=argmax_token_logit_gt_0`、非空超像素数 `superpixels`、各类别超像素个数 `classes[]`，0 为背景）与 `artifacts.token_classes_url`（每个超像素按自身类别着色的叠加图，多尺度平均） |
| slic | 模型实际使用的超像素：标签 NPY（原图尺寸 int32）、彩色区域图、黄色边界图；`used_for_prediction=true` |
| cam | 每个检测类别一张 JET 热力叠加图，`artifacts.class_<id>_url`；开启 `patch_attn_refine` 时另有细化前的 `artifacts.raw_class_<id>_url`，`data.refined` 标明是否细化；另外两种 CAM 类型为 `artifacts.<type>_class_<id>_url`（`type` ∈ `fused`/`patchcam`/`mct`，不含当前 `data.attention_type`，同一次前向得到，开启细化时同样细化），`data.types` 列出全部类型；`data.gt_enhanced` / `parameters.gt_enhanced`（布尔）与 `gt_matched_id`（命中的 VOC 图片 ID，未命中为 `null`）及 `parameters` 中的 `gamma`/`theta`/`alpha`/`beta` 记录是否实际触发超像素-GT 增强；`used_for_prediction=false` |

节点 `status` 取 `completed`、`unavailable`、`failed`；辅助节点失败不影响已生成的预测。

## GET /api/v1/results/{request_id}/{artifact}

仅下载响应实际列出的产物，或 `result.json` 完整响应。掩码 PNG 为 uint8 类别编号（VOC 0–20，COCO 0–80）。不存在、已过期或不在白名单内返回 404。下载禁用缓存，后台每 60 秒清理到期结果。

## GET /api/v1/models

返回 `default_model`、每个常驻模型的 id、权重 SHA256、数据集（`VOC12`/`COCO`，由权重类别数推断）、`num_sp_tokens`、类别表、节点能力和 `default_parameters`，以及顶层 `limits`（`max_upload_bytes`、`max_pixels`、`max_inference_side`、`result_ttl_seconds`、`upload_timeout_seconds`、`queue_capacity`、`accepted_formats`）。全部 JSON 响应类型在 `backend/schemas.py` 定义，并公开于受鉴权的 `/openapi.json`。

## 健康检查

- `/health/live`：进程存活，200。
- `/health/ready`：全部模型加载完成时 200；否则 503。

## 错误

错误响应为 `{"error": "..."}`。

| HTTP | 示例原因 |
| --- | --- |
| 400 | 无效图片、`invalid_include`、`unknown_model`、`invalid_params`、缺少文件、未知或重复字段、错误 multipart |
| 401 | Bearer key 缺失或错误 |
| 404 | 结果不存在或过期 |
| 408 | 上传超过时间限制 |
| 413 | 文件、请求体或解码像素超过限制 |
| 415 | 不支持的格式、动画图片、错误 Content-Type |
| 429 | 上传/排队/推理总容量已满，附 Retry-After |
| 500 | 推理失败；内部异常保留在服务器日志 |
| 503 | 模型未就绪 |
