<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import {
  ATTENTION_TYPES,
  FALLBACK_PARAMS,
  buildParams,
  checkImage,
  color,
  formatScales,
  resultViews,
  scoreBars,
  stageTimings,
} from "./params";

// Page structure follows the user manual (figs. 4.1, 4.2): two tabs, “参数配置” then “语义分割”.
const pages = { config: "参数配置", segment: "语义分割" };
const activePage = ref("config");
const stageNames = { slic: "超像素分割", cam: "模型注意力图" };
const attentionNames = { fused: "融合（fused）", patchcam: "PatchCAM", mct: "MCT" };

// Connection: the key is handed to the Rust side, which keeps it in memory only.
const connectionDialog = ref(null);
const baseUrl = ref(localStorage.getItem("seg.baseUrl") || "http://127.0.0.1:8000");
const key = ref("");
const showKey = ref(false);
const connected = ref(false);
const connecting = ref(false);
const service = ref(null); // /api/v1/models response
const connectionError = ref("");

const image = ref(null); // { path, name, bytes, mime, width, height, data_url }
const modelId = ref("");
const include = ref(["slic", "cam"]);
const form = reactive({ ...FALLBACK_PARAMS, scales: formatScales(FALLBACK_PARAMS.scales) });
const confirmed = ref(null);

const running = ref(false);
const saving = ref(false);
const picking = ref(false); // a native file/folder dialog is open
const outcome = ref(null); // { result, cache_dir, files, failed }
const shownOutcome = ref(null); // revealed after the completion dialog, as in the manual
const viewIndex = ref(0);
const error = ref("");
const notice = ref("");
const messageDialog = ref(null);
const message = reactive({ title: "", body: "", pending: false, isError: false });

const locked = computed(
  () => connecting.value || running.value || saving.value || picking.value,
);
const model = computed(() => service.value?.models.find((m) => m.id === modelId.value) || null);
const limits = computed(() => service.value?.limits || null);
const classNames = computed(() =>
  Object.fromEntries((model.value?.classes || []).map((c) => [c.id, c.name])),
);
const views = computed(() =>
  shownOutcome.value
    ? resultViews(shownOutcome.value.result, classNames.value).filter(
        (v) => shownOutcome.value.files[v.url],
      )
    : [],
);
const currentView = computed(() => views.value[viewIndex.value % (views.value.length || 1)]);
const currentSrc = computed(() =>
  currentView.value ? convertFileSrc(shownOutcome.value.files[currentView.value.url]) : "",
);
const compareSrc = computed(() => {
  const path = shownOutcome.value?.files[currentView.value?.compare?.url];
  return path ? convertFileSrc(path) : "";
});
const resultSummary = computed(() => shownOutcome.value?.result);
const scores = computed(() => scoreBars(resultSummary.value));
const scoreThreshold = computed(() => resultSummary.value?.parameters.cls_threshold ?? 0.5);
const timings = computed(() => stageTimings(resultSummary.value));
const confidence = computed(
  () => resultSummary.value?.stages.segmentation?.data?.confidence || null,
);

function stageAvailable(stage) {
  return model.value?.stages?.[stage] !== "unavailable";
}
function applyDefaults() {
  const defaults = model.value?.default_parameters || FALLBACK_PARAMS;
  Object.assign(form, { ...defaults, scales: formatScales(defaults.scales) });
}
function say(text) {
  error.value = "";
  notice.value = text;
}
function fail(e) {
  error.value = String(e?.message ?? e);
}

function openSettings() {
  connectionError.value = "";
  connectionDialog.value.showModal();
}
async function connect() {
  if (locked.value) return;
  connecting.value = true;
  connected.value = false;
  connectionError.value = "";
  try {
    const data = await invoke("connect", { baseUrl: baseUrl.value, key: key.value });
    localStorage.setItem("seg.baseUrl", baseUrl.value.trim());
    service.value = data;
    const ids = data.models.map((m) => m.id);
    if (!ids.includes(modelId.value)) modelId.value = data.default_model;
    connected.value = true;
    applyDefaults();
    include.value = include.value.filter(stageAvailable);
    say("服务已连接，可以选择图像并配置参数。");
  } catch (e) {
    connectionError.value = String(e);
    service.value = null;
  } finally {
    connecting.value = false;
  }
}
function invalidateConnection() {
  if (!connected.value) return;
  connected.value = false;
  invoke("disconnect");
  say("连接参数已修改，请重新连接服务。");
}
watch(activePage, () => window.scrollTo(0, 0));
watch(modelId, (next, prev) => {
  if (prev && next !== prev) applyDefaults();
});

async function loadImage(path) {
  if (locked.value || !path) return;
  try {
    const info = await invoke("inspect_image", { path });
    checkImage(info, limits.value);
    image.value = info;
    say(`已导入 ${info.name}（${info.width} × ${info.height}）。`);
  } catch (e) {
    fail(e);
  }
}
async function pick(options) {
  if (locked.value) return null;
  picking.value = true;
  try {
    return await open(options);
  } finally {
    picking.value = false;
  }
}
async function chooseImage() {
  const path = await pick({
    title: "选择图像",
    multiple: false,
    directory: false,
    filters: [{ name: "图像", extensions: ["jpg", "jpeg", "png", "webp"] }],
  });
  if (path) await loadImage(path);
}
function resetAll() {
  if (locked.value) return;
  image.value = null;
  confirmed.value = null;
  outcome.value = null;
  shownOutcome.value = null;
  include.value = ["slic", "cam"].filter(stageAvailable);
  applyDefaults();
  say("已重置图像与参数。");
}
function confirmData() {
  if (locked.value) return;
  try {
    if (!connected.value) throw new Error("请先通过右上角“服务连接”连接服务。");
    if (!image.value) throw new Error("请先选择待分割图像。");
    checkImage(image.value, limits.value);
    const params = buildParams(form);
    confirmed.value = {
      image: image.value,
      model: model.value,
      include: include.value.filter(stageAvailable),
      params,
    };
    outcome.value = null;
    shownOutcome.value = null;
    activePage.value = "segment";
    say("参数已确认并锁定，可以执行程序。");
  } catch (e) {
    fail(e);
    showMessage("参数检查未通过", error.value, { isError: true });
  }
}

function showMessage(title, body, { pending = false, isError = false } = {}) {
  Object.assign(message, { title, body, pending, isError });
  if (!messageDialog.value.open) messageDialog.value.showModal();
}
function acknowledge() {
  messageDialog.value.close();
}
function onMessageClosed() {
  if (!message.pending && outcome.value) {
    shownOutcome.value = outcome.value;
    viewIndex.value = 0;
  }
}
async function run() {
  if (locked.value || !confirmed.value) return;
  running.value = true;
  outcome.value = null;
  shownOutcome.value = null;
  error.value = "";
  showMessage("程序开始执行", "图像已提交到服务器，正在进行语义分割，请稍候。", { pending: true });
  try {
    const c = confirmed.value;
    const data = await invoke("segment", {
      path: c.image.path,
      model: c.model.id,
      include: c.include,
      params: c.params,
    });
    outcome.value = data;
    const partial = data.failed.length ? `，${data.failed.length} 个产物下载失败` : "";
    say(`分割已完成${partial}。结果已缓存到本地。`);
    showMessage(
      "程序执行完毕",
      `耗时 ${data.result.duration_ms.toFixed(0)} ms，检测到 ${data.result.classes.length} 个非空类别${partial}。点击“确定”查看结果。`,
    );
  } catch (e) {
    fail(e);
    showMessage("执行失败", error.value, { isError: true });
  } finally {
    running.value = false;
  }
}
function switchView() {
  if (views.value.length) viewIndex.value = (viewIndex.value + 1) % views.value.length;
}
async function saveAs() {
  if (locked.value || !shownOutcome.value) return;
  const dest = await pick({ title: "结果另存为（选择目标文件夹）", directory: true });
  if (!dest) return;
  saving.value = true;
  try {
    const target = await invoke("save_results", {
      cacheDir: shownOutcome.value.cache_dir,
      dest,
    });
    say(`结果已另存到 ${target}`);
  } catch (e) {
    fail(e);
  } finally {
    saving.value = false;
  }
}

function expiry(seconds) {
  return new Date(seconds * 1000).toLocaleString("zh-CN", { hour12: false });
}
function paramSummary(p) {
  return [
    `尺度 ${formatScales(p.scales)}`,
    p.flip ? "翻转" : "不翻转",
    `${attentionNames[p.attention_type]} L${p.layer_index}`,
    `阈值 ${p.cls_threshold}/${p.bg_threshold}`,
    p.patch_attn_refine ? "亲和力细化" : "无细化",
    p.crf ? `CRF×${p.crf_iters}` : "无 CRF",
  ].join(" · ");
}

let unlistenDrop = null;
onMounted(async () => {
  try {
    unlistenDrop = await getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "drop" && activePage.value === "config")
        loadImage(event.payload.paths?.[0]);
    });
  } catch {
    // Outside Tauri (plain browser preview) there is no webview drag-drop channel.
  }
});
onBeforeUnmount(() => unlistenDrop?.());
</script>

<template>
  <div class="window">
    <h1 class="app-title">Weakly Supervised Semantic Segmentation System</h1>

    <div class="tabbar">
      <div class="tabs" role="tablist" aria-label="功能页面">
        <button
          v-for="(title, id) in pages"
          :key="id"
          role="tab"
          class="tab"
          :class="{ active: activePage === id }"
          :aria-selected="activePage === id"
          @click="activePage = id"
        >
          {{ title }}
        </button>
      </div>
      <button
        class="connection-settings"
        aria-label="服务连接设置"
        aria-haspopup="dialog"
        @click="openSettings"
      >
        <i class="connection-dot" :class="{ online: connected }" aria-hidden="true"></i
        >{{ connected ? "服务已连接" : "服务连接" }}
      </button>
    </div>

    <!-- 参数配置（图 4.1 / 4.4 / 4.5）：左侧配置参数，右侧预览图片 -->
    <div v-show="activePage === 'config'" class="tab-pane config-page" role="tabpanel">
      <fieldset class="group config-group">
        <legend>配置参数</legend>
        <div class="form-row path-row">
          <span>输入图片路径：</span>
          <input class="mono" readonly :value="image?.path || ''" aria-label="输入图片路径" />
        </div>
        <div class="button-row">
          <button :disabled="locked" @click="chooseImage">选择图片</button>
          <button :disabled="locked" @click="resetAll">重置图片</button>
        </div>
        <label class="form-row">
          <span>选择分割模型：</span>
          <select v-model="modelId" :disabled="locked || !connected">
            <option v-if="!service" value="">请先连接服务</option>
            <option v-for="m in service?.models || []" :key="m.id" :value="m.id">
              {{ m.id }}{{ m.id === service.default_model ? "（默认）" : "" }}
            </option>
          </select>
        </label>
        <div class="form-row">
          <span>附加分析：</span>
          <div class="checks">
            <label v-for="stage in ['slic', 'cam']" :key="stage">
              <input
                type="checkbox"
                :value="stage"
                v-model="include"
                :disabled="locked || !stageAvailable(stage)"
              />{{ stageNames[stage] }}{{ stageAvailable(stage) ? "" : "（未配置）" }}
            </label>
          </div>
        </div>
        <div class="form-row short">
          <span>超像素数量（由模型决定）：</span>
          <input readonly :value="model?.num_sp_tokens ?? ''" aria-label="超像素数量" />
        </div>
        <div class="form-row short">
          <span>超像素紧凑度：</span>
          <input readonly :value="model ? 10 : ''" aria-label="超像素紧凑度" />
        </div>

        <details class="advanced">
          <summary>
            推理参数<span>默认值与训练脚本一致</span>
            <button class="link-button" :disabled="locked" @click.prevent="applyDefaults">
              恢复默认
            </button>
          </summary>
          <label class="form-row">
            <span>多尺度：</span>
            <input v-model="form.scales" :disabled="locked" placeholder="1.0, 0.75, 1.25" />
          </label>
          <label class="form-row">
            <span>注意力类型：</span>
            <select v-model="form.attention_type" :disabled="locked">
              <option v-for="t in ATTENTION_TYPES" :key="t" :value="t">{{ attentionNames[t] }}</option>
            </select>
          </label>
          <label class="form-row short">
            <span>注意力层数：</span>
            <input type="number" min="1" max="12" step="1" v-model.number="form.layer_index" :disabled="locked" />
          </label>
          <label class="form-row">
            <span>类别阈值：<b class="mono">{{ Number(form.cls_threshold).toFixed(2) }}</b></span>
            <input type="range" min="0" max="1" step="0.01" v-model.number="form.cls_threshold" :disabled="locked" />
          </label>
          <label class="form-row">
            <span>背景阈值：<b class="mono">{{ Number(form.bg_threshold).toFixed(2) }}</b></span>
            <input type="range" min="0" max="1" step="0.01" v-model.number="form.bg_threshold" :disabled="locked" />
          </label>
          <div class="form-row">
            <span>增强与后处理：</span>
            <div class="checks">
              <label><input type="checkbox" v-model="form.flip" :disabled="locked" />水平翻转</label>
              <label><input type="checkbox" v-model="form.patch_attn_refine" :disabled="locked" />亲和力细化</label>
              <label><input type="checkbox" v-model="form.crf" :disabled="locked" />DenseCRF</label>
            </div>
          </div>
          <label class="form-row short">
            <span>CRF 迭代次数：</span>
            <input
              type="number"
              min="1"
              max="20"
              step="1"
              v-model.number="form.crf_iters"
              :disabled="locked || !form.crf"
            />
          </label>
        </details>

        <button class="primary confirm" :disabled="locked || !image || !connected" @click="confirmData">
          确认数据
        </button>
      </fieldset>

      <fieldset class="group preview-group">
        <legend>预览图片</legend>
        <img v-if="image" :src="image.data_url" alt="输入图片预览" class="preview-image" />
        <p v-else class="placeholder">输入图片预览</p>
      </fieldset>
    </div>

    <!-- 语义分割（图 4.2 / 4.9 / 4.10）：上方参数列表与执行按钮，下方输出效果展示 -->
    <div v-show="activePage === 'segment'" class="tab-pane" role="tabpanel">
      <div class="group segment-frame">
        <div class="segment-top">
          <fieldset class="group param-list">
            <legend>参数列表</legend>
            <div class="list-row">
              <span>输入图片路径：</span>
              <input class="mono" readonly :value="confirmed?.image.path || ''" aria-label="输入图片路径" />
            </div>
            <div class="list-row">
              <span>分割模型：</span>
              <input
                readonly
                :value="confirmed ? `${confirmed.model.id} · ${confirmed.model.dataset}` : ''"
                aria-label="分割模型"
              />
            </div>
            <div class="list-row">
              <span>附加分析：</span>
              <input
                readonly
                :value="confirmed ? confirmed.include.map((s) => stageNames[s]).join('、') || '无' : ''"
                aria-label="附加分析"
              />
            </div>
            <div class="list-row">
              <span>推理参数：</span>
              <input
                readonly
                :value="confirmed ? paramSummary(confirmed.params) : ''"
                :title="confirmed ? paramSummary(confirmed.params) : ''"
                aria-label="推理参数"
              />
            </div>
            <div class="list-row pair">
              <span>超像素数</span>
              <input readonly :value="confirmed?.model.num_sp_tokens ?? ''" aria-label="超像素数" />
              <span>紧凑度：</span>
              <input readonly :value="confirmed ? 10 : ''" aria-label="紧凑度" />
            </div>
          </fieldset>

          <div class="run-column">
            <span>分割结果输出路径：</span>
            <input class="mono" readonly :value="shownOutcome?.cache_dir || ''" aria-label="分割结果输出路径" />
            <div class="button-row three">
              <button class="primary" :disabled="locked || !confirmed || !connected" @click="run">
                {{ running ? "正在处理…" : "执行程序" }}
              </button>
              <button :disabled="locked || views.length < 2" @click="switchView">切换视图</button>
              <button :disabled="locked || !shownOutcome" @click="saveAs">
                {{ saving ? "保存中…" : "结果另存为" }}
              </button>
            </div>
          </div>
        </div>

        <hr />

        <section class="output" :aria-busy="running" aria-label="输出效果展示">
          <template v-if="shownOutcome">
            <div class="output-body">
            <div class="compare">
              <figure v-if="compareSrc">
                <img :src="compareSrc" :alt="currentView.compare.label" />
                <figcaption>{{ currentView.compare.label }}</figcaption>
              </figure>
              <figure v-else>
                <img :src="confirmed.image.data_url" alt="原图" />
                <figcaption>原图</figcaption>
              </figure>
              <figure>
                <img
                  v-if="currentView"
                  :src="currentSrc"
                  :alt="currentView.label"
                  :class="{ pixelated: currentView.id === 'mask_url' }"
                />
                <figcaption>
                  {{ currentView?.label || "无可用预览" }}
                  <span v-if="views.length">（{{ (viewIndex % views.length) + 1 }}/{{ views.length }}）</span>
                </figcaption>
              </figure>
            </div>
            <aside class="insights">
              <section v-if="scores.length" aria-label="类别置信度">
                <h2>类别置信度 <small>阈值 {{ scoreThreshold.toFixed(2) }}</small></h2>
                <div class="score-chart" :style="{ '--threshold': scoreThreshold }">
                  <div
                    v-for="item in scores"
                    :key="item.id"
                    class="score-row"
                    :class="{ detected: item.detected }"
                    :title="`${item.name}：${item.score.toFixed(3)}${item.detected ? '（已检测）' : ''}`"
                  >
                    <span class="score-name">{{ item.name }}</span>
                    <span class="score-track">
                      <i
                        :style="{
                          width: `${item.score * 100}%`,
                          background: item.detected ? color(item.id) : undefined,
                        }"
                      ></i>
                    </span>
                    <b class="mono">{{ item.score.toFixed(2) }}</b>
                  </div>
                </div>
              </section>
              <section v-if="timings.length" aria-label="分阶段耗时">
                <h2>分阶段耗时 <small>共 {{ resultSummary.duration_ms.toFixed(0) }} ms</small></h2>
                <div class="timing-row" v-for="t in timings" :key="t.key">
                  <span>{{ t.label }}</span>
                  <span class="timing-track">
                    <i :style="{ width: `${(t.ms / resultSummary.duration_ms) * 100}%` }"></i>
                  </span>
                  <b class="mono">{{ t.ms.toFixed(0) }} ms</b>
                </div>
              </section>
            </aside>
            </div>
            <div class="legend">
              <span v-for="item in resultSummary.classes" :key="item.id" class="legend-item">
                <i :style="{ background: color(item.id) }"></i>{{ item.name }}
                <b class="mono">{{ (item.area_ratio * 100).toFixed(1) }}%</b>
              </span>
              <span
                v-if="confidence"
                class="legend-item"
                :title="`像素置信度 = 第一、第二候选之差；低于 ${confidence.low_threshold} 视为低置信`"
              >
                像素置信度均值 <b class="mono">{{ confidence.mean.toFixed(2) }}</b> · 低置信
                <b class="mono">{{ (confidence.low_ratio * 100).toFixed(1) }}%</b>
              </span>
              <span class="legend-meta">
                {{ resultSummary.width }} × {{ resultSummary.height }} ·
                {{ resultSummary.duration_ms.toFixed(0) }} ms · 服务端结果过期
                {{ expiry(resultSummary.expires_at) }}
              </span>
            </div>
          </template>
          <p v-else class="placeholder">{{ running ? "服务器处理中…" : "输出效果展示" }}</p>
        </section>
      </div>
    </div>

    <footer class="statusbar" aria-live="polite">
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <p v-else>{{ notice || "请先点击右上角“服务连接”填写服务地址与访问密钥。" }}</p>
      <span>{{ model ? `${model.dataset} · ${model.classes.length} 类语义标签` : "未加载模型" }}</span>
    </footer>

    <dialog
      ref="connectionDialog"
      class="connection-dialog"
      aria-labelledby="connection-title"
      @close="showKey = false"
    >
      <div class="dialog-heading">
        <h2 id="connection-title">服务连接</h2>
        <button class="close-settings" @click="connectionDialog.close()">关闭</button>
      </div>
      <section class="connection">
        <label class="field">
          <span>服务地址</span>
          <input
            class="mono"
            v-model="baseUrl"
            :disabled="locked"
            placeholder="http://127.0.0.1:8000"
            @input="invalidateConnection"
          />
        </label>
        <div class="field">
          <label for="api-key">访问密钥</label>
          <div class="key-input">
            <input
              id="api-key"
              :type="showKey ? 'text' : 'password'"
              v-model="key"
              :disabled="locked"
              autocomplete="off"
              placeholder="仅保存在本次运行的内存中"
              @input="invalidateConnection"
              @keydown.enter="connect"
            /><button @click="showKey = !showKey" :aria-pressed="showKey">
              {{ showKey ? "隐藏" : "显示" }}
            </button>
          </div>
        </div>
        <button class="primary connect" @click="connect" :disabled="locked">
          {{ connecting ? "连接中…" : "连接服务" }}
        </button>
      </section>
      <div class="dialog-note" aria-live="polite">
        <p v-if="connectionError" class="error">{{ connectionError }}</p>
        <p v-else-if="connected">
          已连接，{{ service.models.length }} 个模型可用；上传上限
          {{ (service.limits.max_upload_bytes / 1048576).toFixed(1) }} MB，结果保留
          {{ Math.round(service.limits.result_ttl_seconds / 60) }} 分钟。
        </p>
        <p v-else>服务地址会保存在本机；访问密钥不落盘，关闭程序后需重新填写。</p>
      </div>
      <div v-if="connected" class="model-table">
        <div v-for="m in service.models" :key="m.id" class="model-row">
          <strong>{{ m.id }}</strong>
          <span>{{ m.dataset }} · {{ m.num_sp_tokens }} 超像素</span>
          <span class="mono">{{ m.checkpoint_sha256.slice(0, 16) }}…</span>
        </div>
      </div>
    </dialog>

    <dialog
      ref="messageDialog"
      class="connection-dialog message-dialog"
      aria-labelledby="message-title"
      @close="onMessageClosed"
    >
      <h2 id="message-title">{{ message.title }}</h2>
      <p :class="{ error: message.isError }">{{ message.body }}</p>
      <p v-if="message.pending" class="muted">关闭此提示不会取消任务，完成后会再次提示。</p>
      <div class="dialog-actions">
        <button class="primary" @click="acknowledge">确定</button>
      </div>
    </dialog>
  </div>
</template>
