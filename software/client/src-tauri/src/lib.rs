//! Native side of the client: every HTTP call to software/backend, the Bearer key,
//! and the local result cache live here, so the webview never holds the key or talks to the network.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use base64::Engine as _;
use reqwest::{multipart, Client, RequestBuilder, StatusCode};
use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Manager, State};

const READ_TIMEOUT: Duration = Duration::from_secs(15);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60);
const SEGMENT_TIMEOUT: Duration = Duration::from_secs(180);
const CACHE_RETENTION: Duration = Duration::from_secs(7 * 24 * 3600);

struct Session {
    base: String,
    key: String,
}

#[derive(Default)]
struct AppState {
    client: Client,
    session: Mutex<Option<Session>>,
}

#[derive(Serialize)]
struct ImageInfo {
    path: String,
    name: String,
    bytes: u64,
    mime: &'static str,
    width: usize,
    height: usize,
    data_url: String,
}

#[derive(Serialize)]
struct SegmentOutcome {
    result: Value,
    cache_dir: String,
    /// Artifact URL path (as listed by the server) -> absolute local file.
    files: BTreeMap<String, String>,
    /// Artifact URL paths whose download failed; the prediction itself still succeeded.
    failed: Vec<String>,
}

fn normalize_base(url: &str) -> Result<String, String> {
    let base = url.trim().trim_end_matches('/');
    let parsed = reqwest::Url::parse(base).map_err(|_| "服务地址无效，示例：http://127.0.0.1:8000".to_string())?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none()
        || parsed.query().is_some() || parsed.fragment().is_some()
    {
        return Err("服务地址须为 http(s)://主机[:端口][/前缀]。".into());
    }
    Ok(base.to_string())
}

/// Only result paths the server itself issues may be fetched (mirrors storage.ARTIFACTS naming).
fn artifact_name(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("/api/v1/results/")?;
    let (id, name) = rest.split_once('/')?;
    let id_ok = id.len() == 32 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    let name_ok = !name.is_empty() && !name.starts_with('.')
        && name.bytes().all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'.' | b'-'));
    (id_ok && name_ok).then_some(name)
}

fn sniff_mime(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if data.len() > 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

fn status_message(status: StatusCode, body: &str) -> String {
    let code = serde_json::from_str::<Value>(body).ok()
        .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default();
    let text = match (status.as_u16(), code.as_str()) {
        (400, "invalid_params") => "推理参数无效，请检查参数范围后重试。",
        (400, "unknown_model") => "所选模型在服务端不存在，请重新连接服务。",
        (400, "invalid_include") => "附加分析节点无效，请重新选择。",
        (400, _) => "图片或请求无效，请重新选择单帧图片。",
        (401, _) => "访问密钥无效，请检查后重新连接。",
        (404, _) => "结果不存在或已过期，请重新执行分割。",
        (408, _) => "上传超时，请检查网络后重试。",
        (413, _) => "图片超过服务端限制，请缩小图片。",
        (415, _) => "格式不支持，请使用单帧 JPEG、PNG 或 WebP。",
        (429, _) => "服务繁忙，请稍后手动重试。",
        (500, _) => "推理失败，请检查服务端日志后重试。",
        (503, _) => "模型尚未就绪，请稍后重新连接。",
        (other, _) => return format!("请求失败（{other}），请重试。"),
    };
    text.to_string()
}

fn transport_message(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "请求超时；服务端可能仍在处理，请勿立即重复提交。".into()
    } else if error.is_connect() {
        "无法连接服务，请检查服务地址、网络或 SSH 隧道。".into()
    } else if error.is_decode() {
        "服务返回了无法解析的数据。".into()
    } else {
        format!("网络请求失败：{error}")
    }
}

async fn send(request: RequestBuilder) -> Result<reqwest::Response, String> {
    let response = request.send().await.map_err(transport_message)?;
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(status_message(status, &body))
}

fn session(state: &AppState) -> Result<(String, String), String> {
    state.session.lock().unwrap().as_ref()
        .map(|s| (s.base.clone(), s.key.clone()))
        .ok_or_else(|| "尚未连接服务，请先在“服务连接”中连接。".to_string())
}

fn results_root(app: &AppHandle) -> Result<PathBuf, String> {
    app.path().app_cache_dir().map(|dir| dir.join("results"))
        .map_err(|e| format!("无法定位缓存目录：{e}"))
}

#[tauri::command]
async fn connect(state: State<'_, AppState>, base_url: String, key: String) -> Result<Value, String> {
    *state.session.lock().unwrap() = None;
    let base = normalize_base(&base_url)?;
    let key = key.trim().to_string();
    if key.is_empty() || !key.is_ascii() {
        return Err("请填写有效的访问密钥（ASCII 字符）。".into());
    }
    send(state.client.get(format!("{base}/health/ready")).timeout(READ_TIMEOUT)).await?;
    let models: Value = send(state.client.get(format!("{base}/api/v1/models"))
        .bearer_auth(&key).timeout(READ_TIMEOUT)).await?
        .json().await.map_err(transport_message)?;
    if models["models"].as_array().is_none_or(|m| m.is_empty()) {
        return Err("服务未提供可用模型。".into());
    }
    *state.session.lock().unwrap() = Some(Session { base, key });
    Ok(models)
}

#[tauri::command]
fn disconnect(state: State<'_, AppState>) {
    *state.session.lock().unwrap() = None;
}

#[tauri::command]
fn inspect_image(path: String) -> Result<ImageInfo, String> {
    let data = fs::read(&path).map_err(|e| format!("无法读取图片：{e}"))?;
    let mime = sniff_mime(&data).ok_or("请选择 JPEG、PNG 或 WebP 图片。")?;
    let size = imagesize::blob_size(&data).map_err(|_| "图片无法解码，请重新选择有效图片。")?;
    let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(ImageInfo {
        bytes: data.len() as u64,
        mime,
        width: size.width,
        height: size.height,
        data_url: format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&data)),
        name,
        path,
    })
}

#[tauri::command]
async fn segment(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    model: String,
    include: Vec<String>,
    params: Value,
) -> Result<SegmentOutcome, String> {
    let (base, key) = session(&state)?;
    let data = fs::read(&path).map_err(|e| format!("无法读取图片：{e}"))?;
    let mime = sniff_mime(&data).ok_or("请选择 JPEG、PNG 或 WebP 图片。")?;
    let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".into());
    let mut stages = vec!["segmentation".to_string()];
    stages.extend(include.into_iter().filter(|s| s != "segmentation"));
    let form = multipart::Form::new()
        .part("image", multipart::Part::bytes(data).file_name(name).mime_str(mime).map_err(|e| e.to_string())?)
        .text("include", serde_json::to_string(&stages).unwrap())
        .text("model", model)
        .text("params", params.to_string());
    let body = send(state.client.post(format!("{base}/api/v1/segment"))
        .bearer_auth(&key).multipart(form).timeout(SEGMENT_TIMEOUT)).await?
        .text().await.map_err(transport_message)?;
    let result: Value = serde_json::from_str(&body).map_err(|_| "服务返回了无法解析的数据。")?;

    let request_id = result["request_id"].as_str().unwrap_or_default();
    if request_id.len() != 32 || !request_id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("服务返回了无效的请求 ID。".into());
    }
    let dir = results_root(&app)?.join(request_id);
    fs::create_dir_all(&dir).map_err(|e| format!("无法创建缓存目录：{e}"))?;
    fs::write(dir.join("result.json"), &body).map_err(|e| format!("无法写入缓存：{e}"))?;

    let mut urls: Vec<String> = result["outputs"].as_object().into_iter().flatten()
        .chain(result["stages"].as_object().into_iter().flatten()
            .flat_map(|(_, stage)| stage["artifacts"].as_object().into_iter().flatten()))
        .filter_map(|(_, url)| url.as_str().map(str::to_owned))
        .collect();
    urls.sort();
    urls.dedup();
    let (mut files, mut failed) = (BTreeMap::new(), Vec::new());
    for url in urls {
        let Some(file) = artifact_name(&url).filter(|_| url.contains(request_id)) else {
            failed.push(url);
            continue;
        };
        let target = dir.join(file);
        let fetched = async {
            let bytes = send(state.client.get(format!("{base}{url}")).bearer_auth(&key)
                .timeout(DOWNLOAD_TIMEOUT)).await?.bytes().await.map_err(transport_message)?;
            fs::write(&target, bytes).map_err(|e| format!("无法写入缓存：{e}"))
        }.await;
        match fetched {
            Ok(()) => { files.insert(url, target.to_string_lossy().into_owned()); }
            Err(_) => failed.push(url),
        }
    }
    Ok(SegmentOutcome { result, cache_dir: dir.to_string_lossy().into_owned(), files, failed })
}

/// Copies one cached result folder into `<dest>/<request_id>/` and returns that folder.
#[tauri::command]
fn save_results(app: AppHandle, cache_dir: String, dest: String) -> Result<String, String> {
    let root = results_root(&app)?.canonicalize().map_err(|_| "缓存目录不存在。")?;
    let source = Path::new(&cache_dir).canonicalize().map_err(|_| "缓存结果不存在，请重新执行分割。")?;
    if source.parent() != Some(root.as_path()) {
        return Err("只能另存本客户端缓存的分割结果。".into());
    }
    let target = Path::new(&dest).join(source.file_name().unwrap());
    fs::create_dir_all(&target).map_err(|e| format!("无法创建目标目录：{e}"))?;
    for entry in fs::read_dir(&source).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_file() {
            fs::copy(entry.path(), target.join(entry.file_name())).map_err(|e| format!("复制失败：{e}"))?;
        }
    }
    Ok(target.to_string_lossy().into_owned())
}

fn prune_cache(root: &Path) {
    let Ok(entries) = fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        let old = entry.metadata().and_then(|m| m.modified()).ok()
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .is_some_and(|age| age > CACHE_RETENTION);
        if old && entry.path().is_dir() {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .setup(|app| {
            if let Ok(root) = results_root(app.handle()) {
                prune_cache(&root);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![connect, disconnect, inspect_image, segment, save_results])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_paths_are_whitelisted() {
        let id = "0123456789abcdef0123456789abcdef";
        assert_eq!(artifact_name(&format!("/api/v1/results/{id}/cam-12.png")), Some("cam-12.png"));
        assert_eq!(artifact_name(&format!("/api/v1/results/{id}/slic-labels.npy")), Some("slic-labels.npy"));
        assert_eq!(artifact_name(&format!("/api/v1/results/{id}/../x.png")), None);
        assert_eq!(artifact_name(&format!("/api/v1/results/{id}/a/b.png")), None);
        assert_eq!(artifact_name("/api/v1/results/XYZ/mask.png"), None);
        assert_eq!(artifact_name(&format!("https://evil/api/v1/results/{id}/mask.png")), None);
    }

    #[test]
    fn base_url_is_normalized() {
        assert_eq!(normalize_base(" http://10.0.0.2:8000/ ").unwrap(), "http://10.0.0.2:8000");
        assert_eq!(normalize_base("https://h/seg").unwrap(), "https://h/seg");
        assert!(normalize_base("ftp://h").is_err());
        assert!(normalize_base("127.0.0.1:8000").is_err());
    }

    #[test]
    fn server_error_codes_map_to_specific_messages() {
        let msg = status_message(StatusCode::BAD_REQUEST, r#"{"error":"invalid_params"}"#);
        assert!(msg.contains("推理参数"));
        assert!(status_message(StatusCode::BAD_REQUEST, "").contains("图片或请求无效"));
        assert!(status_message(StatusCode::IM_A_TEAPOT, "").contains("418"));
    }
}
