# 弱监督语义分割系统 · 桌面客户端

基于 Tauri 2（Rust + 系统 WebView）的原生桌面客户端，对接 `software/backend` 的 FastAPI 推理服务。
页面结构按《基于超像素引导ViT的弱监督语义分割系统》用户手册第 4 章实现，视觉风格沿用
`ref/model_rgrp/software/frontend`（Vue 3 + Vite，`src/style.css` 的设计 token 直接复用）。

## 页面与手册对应

| 手册 | 客户端 |
| --- | --- |
| 图 4.1 / 4.4 / 4.5「参数配置」 | 顶部居中标题与“参数配置 / 语义分割”标签页（右上角为服务连接）；左侧“配置参数”分组：输入图片路径、选择图片 / 重置图片、分割模型、附加分析、超像素数量与紧凑度（只读）、可展开的推理参数，“确认数据”；右侧“预览图片”分组 |
| 图 4.2「语义分割」 | 上方左侧“参数列表”分组（只读输入框），右侧分割结果输出路径与执行程序 / 切换视图 / 结果另存为；分隔线下方为原图与结果对比、类别图例 |
| 图 4.6 / 4.7 | “程序开始执行”“程序执行完毕”模态提示 |
| 图 4.8 | 执行中三个按钮禁用 |
| 图 4.9 / 4.10 | 点击“确定”后显示结果；“切换视图”在叠加图、彩色分割、类别掩码、超像素边界/区域、各类别注意力图之间循环 |
| 图 4.11 | “结果另存为”选择文件夹，复制本次全部产物到 `<目录>/<request_id>/` |

手册中的“超像素数 / 紧凑度”由服务端模型决定（`num_sp_tokens`，紧凑度固定 10），客户端只显示不提交；
可调参数与 `software/docs/api.md` 的 `params` 一一对应（多尺度、翻转、注意力类型、层数、亲和力细化、类别/背景阈值、CRF 及迭代次数），
默认值取自 `/api/v1/models` 的 `default_parameters`，切换模型时自动重置。

## 架构

- 所有 HTTP 请求在 Rust 端（`src-tauri/src/lib.rs`，reqwest）完成：`/health/ready`、`/api/v1/models`、`/api/v1/segment`（multipart：`image`、`include`、`model`、`params`）、`/api/v1/results/{id}/{artifact}`。
- 访问密钥只保存在 Rust 进程内存中，WebView 不持有密钥、也没有网络权限（CSP 仅允许 IPC）；服务地址保存在本机。
- 产物只按服务端返回的 `/api/v1/results/<32位hex>/<文件名>` 白名单下载，缓存到系统缓存目录
  （Linux `~/.cache/cn.edu.sxu.drcformer.client/results/<request_id>/`），启动时清理 7 天前的缓存。
- 前端：`src/App.vue`（页面）、`src/params.js`（参数校验与视图顺序，镜像 `backend/schemas.py`）、`src/style.css`。

## 运行

依赖：Node 18+、Rust 1.82+，以及 Tauri 的系统依赖：

- Linux（Debian/Ubuntu）：`sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libxdo-dev libssl-dev build-essential pkg-config`
- Windows：WebView2（Windows 10/11 自带）与 MSVC 生成工具
- macOS：Xcode Command Line Tools

```bash
cd software/client
npm install
npm run tauri dev            # 开发模式（Vite 热更新，端口 1420）
npm run tauri build          # 生成安装包：src-tauri/target/release/bundle/
npm test                     # 前端参数校验单元测试
(cd src-tauri && cargo test) # Rust 端地址白名单、错误映射测试
```

后端启动与连接的完整步骤见 [../docs/usage.md](../docs/usage.md)。

启动后点击右上角“服务连接”，填写服务地址（默认 `http://127.0.0.1:8000`）与 `SEG_API_KEY`。
后端默认只监听服务器本机，远程使用时建议 SSH 隧道：

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<gpu-server>
```
