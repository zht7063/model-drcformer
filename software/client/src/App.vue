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
} from "./params";

// Page structure follows the user manual: “参数配置” then “语义分割”.
const pages = {
  config: {
    title: "参数配置",
    description: "选择待分割图像，配置模型与推理参数，确认后进入语义分割。",
    icon: "◈",
  },
  segment: {
    title: "语义分割",
    description: "核对参数列表，执行分割并对比原图与模型输出。",
    icon: "▦",
  },
};
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
const resultSummary = computed(() => shownOutcome.value?.result);

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
    if (!connected.value) throw new Error("请先通过左下角“服务连接”连接服务。");
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

function megabytes(bytes) {
  return `${(bytes / 1048576).toFixed(2)} MB`;
}
function expiry(seconds) {
  return new Date(seconds * 1000).toLocaleString("zh-CN", { hour12: false });
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
  <div class="shell">
    <aside class="rail">
      <div class="rail-label">弱监督语义分割</div>
      <nav aria-label="功能导航">
        <button
          v-for="(page, id) in pages"
          :key="id"
          class="nav"
          :class="{ active: activePage === id }"
          :aria-current="activePage === id ? 'page' : undefined"
          @click="activePage = id"
        >
          <i aria-hidden="true">{{ page.icon }}</i><span>{{ page.title }}</span>
        </button>
      </nav>
      <button
        class="connection-settings"
        aria-label="服务连接设置"
        aria-haspopup="dialog"
        @click="openSettings"
      >
        <span aria-hidden="true">⚙</span><span class="settings-label">服务连接</span
        ><i class="connection-dot" :class="{ online: connected }" aria-hidden="true"></i>
      </button>
      <div class="rail-foot">
        <span class="dot"></span>
        {{ model ? `${model.dataset} · ${model.classes.length} 类语义标签` : "未加载模型" }}
      </div>
    </aside>

    <main>
      <header>
        <div>
          <p class="eyebrow">WEAKLY SUPERVISED SEMANTIC SEGMENTATION SYSTEM</p>
          <h1>{{ pages[activePage].title }}<span class="version">v1.0</span></h1>
          <p class="muted">{{ pages[activePage].description }}</p>
        </div>
        <span class="pill" :class="{ online: connected }">{{
          connected ? "服务已连接" : "服务未连接"
        }}</span>
      </header>

      <div class="feedback" aria-live="polite">
        <p v-if="error" class="error" role="alert">{{ error }}</p>
        <p v-else>{{ notice || "请先通过左下角“服务连接”填写服务地址与访问密钥。" }}</p>
      </div>

      <!-- 参数配置 -->
      <div v-show="activePage === 'config'" class="workspace-grid">
        <section class="panel input-panel">
          <div class="section-heading">
            <h2>输入与参数</h2>
            <span>01 / PARAMS</span>
          </div>
          <div class="action-row">
            <button class="primary" :disabled="locked" @click="chooseImage">选择图像</button>
            <button :disabled="locked" @click="resetAll">重置</button>
          </div>
          <label class="field">
            <span>输入图片路径</span>
            <input class="mono" readonly :value="image?.path || ''" placeholder="尚未选择图像" />
          </label>

          <div class="options">
            <h3>模型 <span>由服务端提供</span></h3>
            <label class="field">
              <span>分割模型</span>
              <select v-model="modelId" :disabled="locked || !connected">
                <option v-if="!service" value="">请先连接服务</option>
                <option v-for="m in service?.models || []" :key="m.id" :value="m.id">
                  {{ m.id }}{{ m.id === service.default_model ? "（默认）" : "" }}
                </option>
              </select>
            </label>
            <p v-if="model" class="muted model-meta">
              {{ model.dataset }} · {{ model.classes.length }} 类 · 超像素数
              {{ model.num_sp_tokens }} · 紧凑度 10 ·
              <span class="mono">{{ model.checkpoint_sha256.slice(0, 12) }}</span>
            </p>
          </div>

          <div class="options">
            <h3>附加分析 <span>可选</span></h3>
            <label v-for="stage in ['slic', 'cam']" :key="stage">
              <input
                type="checkbox"
                :value="stage"
                v-model="include"
                :disabled="locked || !stageAvailable(stage)"
              /><span>{{ stageNames[stage] }}</span
              ><small>{{ stageAvailable(stage) ? "辅助节点" : "未配置" }}</small>
            </label>
          </div>

          <div class="options">
            <h3>
              推理参数 <span>默认值与训练脚本一致</span>
              <button class="link-button" :disabled="locked" @click="applyDefaults">恢复默认</button>
            </h3>
            <div class="param-grid">
              <label class="field">
                <span>多尺度</span>
                <input v-model="form.scales" :disabled="locked" placeholder="1.0, 0.75, 1.25" />
              </label>
              <label class="field">
                <span>注意力类型</span>
                <select v-model="form.attention_type" :disabled="locked">
                  <option v-for="t in ATTENTION_TYPES" :key="t" :value="t">
                    {{ attentionNames[t] }}
                  </option>
                </select>
              </label>
              <label class="field">
                <span>注意力层数</span>
                <input type="number" min="1" max="12" step="1" v-model.number="form.layer_index" :disabled="locked" />
              </label>
              <label class="field">
                <span>CRF 迭代次数</span>
                <input
                  type="number"
                  min="1"
                  max="20"
                  step="1"
                  v-model.number="form.crf_iters"
                  :disabled="locked || !form.crf"
                />
              </label>
              <label class="field">
                <span>类别阈值 <b class="mono">{{ Number(form.cls_threshold).toFixed(2) }}</b></span>
                <input type="range" min="0" max="1" step="0.01" v-model.number="form.cls_threshold" :disabled="locked" />
              </label>
              <label class="field">
                <span>背景阈值 <b class="mono">{{ Number(form.bg_threshold).toFixed(2) }}</b></span>
                <input type="range" min="0" max="1" step="0.01" v-model.number="form.bg_threshold" :disabled="locked" />
              </label>
            </div>
            <label><input type="checkbox" v-model="form.flip" :disabled="locked" /><span>水平翻转增强</span></label>
            <label
              ><input type="checkbox" v-model="form.patch_attn_refine" :disabled="locked" /><span
                >超像素亲和力细化</span
              ></label
            >
            <label><input type="checkbox" v-model="form.crf" :disabled="locked" /><span>DenseCRF 后处理</span></label>
          </div>

          <button class="primary run" :disabled="locked || !image || !connected" @click="confirmData">
            确认数据 <span>→</span>
          </button>
        </section>

        <section class="panel result-panel">
          <div class="section-heading">
            <h2>输入图像预览</h2>
            <span>{{ image ? `${image.width} × ${image.height}` : "02 / PREVIEW" }}</span>
          </div>
          <div class="upload preview-area">
            <img v-if="image" :src="image.data_url" alt="输入图像预览" class="preview-image" />
            <div v-else class="upload-empty">
              <span class="upload-icon" aria-hidden="true">↥</span>
              <h3>选择一张待分割图像</h3>
              <p>拖放图像到窗口，或点击左侧“选择图像”</p>
            </div>
          </div>
          <div class="result-summary">
            <span>{{ image ? image.name : "JPEG / PNG / WebP · 单帧图片" }}</span>
            <span>{{
              image
                ? `${megabytes(image.bytes)} · ${image.mime}`
                : limits
                  ? `最大 ${(limits.max_upload_bytes / 1048576).toFixed(1)} MB · ${limits.max_pixels.toLocaleString("zh-CN")} 像素`
                  : "连接后显示服务限制"
            }}</span>
          </div>
        </section>
      </div>

      <!-- 语义分割 -->
      <div v-show="activePage === 'segment'" class="segment-page">
        <p v-if="!confirmed" class="panel page-empty">
          尚未确认参数。请前往
          <button class="link-button" @click="activePage = 'config'">参数配置</button>
          选择图像并点击“确认数据”。
        </p>
        <div class="segment-top">
          <section class="panel param-list">
            <div class="section-heading">
              <h2>参数列表</h2>
              <span>LOCKED</span>
            </div>
            <dl v-if="confirmed">
              <dt>输入图片路径</dt>
              <dd class="mono">{{ confirmed.image.path }}</dd>
              <dt>图片尺寸</dt>
              <dd>{{ confirmed.image.width }} × {{ confirmed.image.height }}</dd>
              <dt>分割模型</dt>
              <dd>{{ confirmed.model.id }} · {{ confirmed.model.dataset }}</dd>
              <dt>超像素数</dt>
              <dd>{{ confirmed.model.num_sp_tokens }}（紧凑度 10）</dd>
              <dt>附加分析</dt>
              <dd>{{ confirmed.include.map((s) => stageNames[s]).join("、") || "无" }}</dd>
              <dt>多尺度 / 翻转</dt>
              <dd>{{ formatScales(confirmed.params.scales) }} · {{ confirmed.params.flip ? "翻转" : "不翻转" }}</dd>
              <dt>注意力</dt>
              <dd>{{ attentionNames[confirmed.params.attention_type] }} · 层数 {{ confirmed.params.layer_index }}</dd>
              <dt>阈值</dt>
              <dd>类别 {{ confirmed.params.cls_threshold }} · 背景 {{ confirmed.params.bg_threshold }}</dd>
              <dt>细化 / CRF</dt>
              <dd>
                {{ confirmed.params.patch_attn_refine ? "亲和力细化" : "无细化" }} ·
                {{ confirmed.params.crf ? `CRF ${confirmed.params.crf_iters} 次` : "无 CRF" }}
              </dd>
            </dl>
            <p v-else class="muted">确认数据后在此显示所选参数。</p>
          </section>

          <section class="panel run-panel">
            <div class="section-heading">
              <h2>执行与导出</h2>
              <span>{{ resultSummary ? `${resultSummary.duration_ms.toFixed(0)} ms` : "RUN" }}</span>
            </div>
            <label class="field">
              <span>分割结果输出路径</span>
              <input class="mono" readonly :value="shownOutcome?.cache_dir || ''" placeholder="执行后显示本地缓存路径" />
            </label>
            <div class="action-row three">
              <button class="primary" :disabled="locked || !confirmed || !connected" @click="run">
                {{ running ? "正在处理…" : "执行程序" }}
              </button>
              <button :disabled="locked || views.length < 2" @click="switchView">切换视图</button>
              <button :disabled="locked || !shownOutcome" @click="saveAs">
                {{ saving ? "保存中…" : "结果另存为" }}
              </button>
            </div>
            <div v-if="views.length" class="view-switch" role="group" aria-label="结果视图">
              <button
                v-for="(v, i) in views"
                :key="v.id"
                :class="{ selected: currentView?.id === v.id }"
                :aria-pressed="currentView?.id === v.id"
                :disabled="locked"
                @click="viewIndex = i"
              >
                {{ v.label }}
              </button>
            </div>
            <p v-if="resultSummary" class="muted fingerprint">
              请求 ID <span class="mono">{{ resultSummary.request_id }}</span> · 服务端结果过期
              {{ expiry(resultSummary.expires_at) }}
            </p>
          </section>
        </div>

        <section class="panel output-panel" :aria-busy="running">
          <div class="section-heading">
            <h2>输出结果展示</h2>
            <span>{{ resultSummary ? `${resultSummary.width} × ${resultSummary.height}` : "OUTPUT" }}</span>
          </div>
          <div v-if="shownOutcome" class="compare">
            <figure>
              <div class="canvas"><img :src="confirmed.image.data_url" alt="原图" class="result-image" /></div>
              <figcaption>原图</figcaption>
            </figure>
            <figure>
              <div class="canvas">
                <img
                  v-if="currentView"
                  :src="currentSrc"
                  :alt="currentView.label"
                  class="result-image"
                  :class="{ pixelated: currentView.id === 'mask_url' }"
                />
              </div>
              <figcaption>{{ currentView?.label || "无可用预览" }}</figcaption>
            </figure>
          </div>
          <div v-else class="canvas">
            <div class="empty-output">
              <span class="crosshair" aria-hidden="true">⌖</span>
              <h3>{{ running ? "服务器处理中" : "等待执行分割" }}</h3>
              <p>
                {{
                  running
                    ? "当前可能正在排队或执行推理，相关按钮暂不可用。"
                    : "执行完毕后，原图与超像素、注意力、分割结果将在此对比展示。"
                }}
              </p>
            </div>
          </div>
          <div v-if="resultSummary" class="class-list">
            <div v-for="item in resultSummary.classes" :key="item.id" class="class-row">
              <i :style="{ background: color(item.id) }"></i
              ><span
                >{{ item.name
                }}<small class="pixel-count">{{ item.pixels.toLocaleString("zh-CN") }} 像素{{
                  item.score != null ? ` · 分数 ${item.score.toFixed(2)}` : ""
                }}</small></span
              >
              <div class="bar">
                <span :style="{ width: `${item.area_ratio * 100}%`, background: color(item.id) }"></span>
              </div>
              <strong>{{ (item.area_ratio * 100).toFixed(1) }}%</strong>
            </div>
          </div>
        </section>
      </div>

      <dialog
        ref="connectionDialog"
        class="connection-dialog"
        aria-labelledby="connection-title"
        @close="showKey = false"
      >
        <div class="section-heading">
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
          <div class="key-field">
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
          <button class="secondary primary" @click="connect" :disabled="locked">
            {{ connecting ? "连接中…" : "连接服务" }}
          </button>
        </section>
        <div class="feedback" aria-live="polite">
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
    </main>
  </div>
</template>
