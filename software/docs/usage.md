# 使用说明：启动后端与连接桌面客户端

本文说明两件事：

1. 在 4090 服务器上启动推理后端（`software/backend`）。
2. 在本地电脑上运行桌面客户端（`software/client`），并连接到后端。

接口细节见 [api.md](api.md)，部署细节见 [deployment.md](deployment.md)。

## 一、在 4090 服务器上启动后端

以下命令都在仓库根目录执行。

### 1. 安装 Web 依赖（只需一次）

后端用项目已有的 conda `py310` 环境，只需在其中补装 Web 依赖：

```bash
/home/ubuntu/miniconda3/envs/py310/bin/pip install -r software/backend/requirements.txt
```

### 2. 准备 checkpoint

后端加载的权重，是训练（`run_drcformer.py` / `main.py`）在输出目录里写出的 `checkpoint.pth`：

```
~/projects/outputs/drcformer/<实验ID>/checkpoint.pth
```

后端会根据权重自动识别数据集：20 类为 VOC，80 类为 COCO。

`num_sp_tokens` 必须和训练时的 `--num_sp_tokens` / `--num-sp-tokens` 一致，默认是 196。填错会导致推理结果错误。

### 3. 编写 `software/.env`

`software/.env` 已被 Git 忽略，不会提交，访问密钥只写在这里。示例：

```bash
# 访问密钥：至少 16 个 ASCII 字符，客户端连接时需要填写同一个值
SEG_API_KEY=请替换为随机生成的长密钥

# 常驻模型列表（JSON）。第一项为默认模型；id 只能用小写字母、数字和 . _ -
SEG_MODELS=[{"id": "drcformer", "checkpoint": "/home/ubuntu/projects/outputs/drcformer/<实验ID>/checkpoint.pth", "num_sp_tokens": 196}]

SEG_DEVICE=cuda:0
# 可选：
# SEG_HOST=127.0.0.1                 # 默认只监听本机；远程请用 SSH 隧道
# SEG_PORT=8000
# SEG_MAX_INFERENCE_SIDE=512
# SEG_RESULTS_DIR=/home/ubuntu/projects/outputs/drcformer/server_results
```

生成密钥可以用：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

需要同时提供多个模型时，在 `SEG_MODELS` 里追加多项。所有模型会在启动时一次性加载，并常驻显存。例如：

```bash
SEG_MODELS=[{"id": "drcformer", "checkpoint": "/home/ubuntu/projects/outputs/drcformer/expA/checkpoint.pth", "num_sp_tokens": 196}, {"id": "drcformer-sp256", "checkpoint": "/home/ubuntu/projects/outputs/drcformer/expB/checkpoint.pth", "num_sp_tokens": 256}]
```

### 4. 检查配置

这一步不加载模型，也不占用 GPU。它会检查以下几项：
- 密钥格式；
- checkpoint 文件是否存在且可读；
- 依赖包是否齐全；
- 结果目录是否可写。

```bash
bash software/backend/run.sh --check-config
```

输出中出现 `"status": "configuration_valid"` 即为通过。密钥在输出中显示为 `configured (redacted)`。

### 5. 在 tmux 中启动服务

服务必须在 tmux 中运行，否则 `run.sh` 会拒绝启动：

```bash
tmux new -s drcformer-seg
cd ~/projects/model-drcformer        # 仓库根目录
CUDA_VISIBLE_DEVICES=0 bash software/backend/run.sh
```

按 `Ctrl-b d` 可以离开 tmux，服务继续运行；之后用 `tmux attach -t drcformer-seg` 回到该会话。

`run.sh` 默认使用 `/home/ubuntu/miniconda3/envs/py310/bin/python`，可以用 `SERVER_PYTHON` 覆盖。

### 6. 确认服务可用

在服务器上另开一个终端，依次执行：

```bash
curl http://127.0.0.1:8000/health/live    # {"status":"alive"}
curl http://127.0.0.1:8000/health/ready   # 模型加载完成后为 {"status":"ready"}，加载中为 503
curl -H "Authorization: Bearer $SEG_API_KEY" http://127.0.0.1:8000/api/v1/models
```

上面的 `$SEG_API_KEY` 需要先在当前终端里设置成你的密钥。

## 二、运行桌面客户端并连接后端

客户端是 Tauri 桌面程序，运行在你自己的电脑上，支持 Windows、macOS 和 Linux。

### 1. 安装依赖（只需一次）

- Node.js 18 及以上；
- Rust 1.82 及以上，用 rustup 安装；
- 系统依赖，按操作系统安装：
  - Windows：Visual Studio 生成工具（勾选“使用 C++ 的桌面开发”）。Windows 10/11 自带 WebView2。
  - macOS：在终端执行 `xcode-select --install`。
  - Debian/Ubuntu：

    ```bash
    sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libxdo-dev libssl-dev build-essential pkg-config
    ```

### 2. 启动

```bash
cd software/client
npm install
npm run tauri dev        # 开发运行
# 或打包安装程序，产物在 src-tauri/target/release/bundle/
npm run tauri build
```

### 3. 连接后端

后端默认只监听服务器的 `127.0.0.1:8000`。从本地电脑访问时，先建立 SSH 隧道，把本地 8000 端口转发到服务器：

```bash
ssh -N -L 8000:127.0.0.1:8000 <用户名>@<4090服务器地址>
```

保持这个终端开着，然后在客户端里：

1. 点击右上角“服务连接”。
2. 服务地址填 `http://127.0.0.1:8000`。如果本地 8000 端口已被占用，可以把隧道改成 `-L 18000:127.0.0.1:8000`，这里相应填 `http://127.0.0.1:18000`。
3. 访问密钥填 `software/.env` 中的 `SEG_API_KEY`。
4. 点击“连接服务”。

连接成功后，右上角按钮变为“服务已连接”且圆点变绿，弹窗中显示可用模型列表。

关于保存：
- 服务地址会保存在本机。
- 访问密钥只保存在程序内存中，每次重新打开客户端都要重新填写。

如果客户端和后端在同一台机器上，不需要隧道，直接填 `http://127.0.0.1:8000`。

### 4. 使用流程

流程和用户手册第 4 章一致：

1. 在「参数配置」页面：
   - 点击“选择图像”，或把图片拖进窗口；
   - 选择模型、附加分析（超像素分割、模型注意力图）和推理参数；
   - 点击“确认数据”。
2. 进入「语义分割」页面后：
   - 核对参数列表，点击“执行程序”；
   - 完成提示框中点击“确定”后，下方会对比显示原图和分割结果。
3. 查看和导出结果：
   - “切换视图”在以下视图之间循环：叠加图、彩色分割、类别掩码、像素置信度、超像素类别、超像素边界、超像素区域，以及每个检测类别的注意力图（细化前后对比、三种 CAM 并排）；
   - “结果另存为”会把本次全部产物（PNG、NPY、result.json）复制到所选目录下的 `<request_id>/` 文件夹。

### 5. 结果页中的分析信息

结果区右侧和下方的图例会显示以下推理中间信息：

- **类别置信度**：模型对每个类别给出的图像级分数（多尺度、翻转平均后的 sigmoid），按分数从高到低列出前 8 类，已检测的类别一定列出。红色虚线是本次的“类别阈值”，越过虚线的类别会参与分割，并以该类别的颜色加粗显示。某个类别没被分割出来时，先看它的分数是否低于阈值；需要时可在“推理参数”里调低类别阈值后重跑。
- **两路分数对比**：类别置信度标题右侧可在“最终 / 两路”之间切换。DRCFormer 的图像级分数由两路平均得到：类 token 路（橙色，各类别 token 的输出均值）和超像素 token 路（青色，每个超像素的类别分数排序后衰减加权）。“两路”模式下每个类别显示两条细条，数值列仍是最终分数；鼠标悬停可看三个具体数值。两路差异大的类别，说明全局（类 token）与局部（超像素）证据不一致。
- **分阶段耗时**：服务端处理时间的拆分，依次为缩放、超像素、模型前向（全部尺度和翻转分支）、亲和力细化、后处理、CRF、保存结果、超像素产物、注意力图产物，以及不属于以上各项的“其他”（图片解码、写结果文件等）。不含网络传输和排队时间。未执行的阶段不显示。
- **像素置信度**（视图）：每个像素第一候选与第二候选的得分差（背景通道为常数“背景阈值”），按 JET 色表叠加在原图上：红、黄表示模型很确定，蓝色表示两个候选接近，分割结果容易出错。图例中给出全图均值，以及得分差低于 0.1 的低置信像素比例。开启 CRF 时，该图反映的是 CRF 之前的置信度。未检测到任何前景类别时没有此视图。
- **细化前后注意力图对比**（视图）：开启“亲和力细化”且选择了“模型注意力图”时，切换到某个类别的注意力图，左侧显示该类别细化前的 CAM，右侧显示超像素亲和力细化后的 CAM，便于观察细化的作用。只有分数超过 0.5 的类别会被细化；低于 0.5 的类别两侧相同。关闭细化时，左侧仍显示原图。
- **三种 CAM 并排**（视图）：选择了“模型注意力图”时，每个检测类别还有一个视图，并排显示融合（fused）、PatchCAM、MCT 三种 CAM，标“（当前）”的是推理参数中选定、用于分割的那一种。三种 CAM 来自同一次前向，不额外跑模型；开启亲和力细化时三种都经过细化，因此“亲和力细化”耗时约为只算一种时的 3 倍。
- **超像素类别**（视图）：模型对每个超像素单独给出的类别判断（超像素 token 的类别分数取最大类，分数为正才着色，否则视为背景），白线为超像素边界。它是“超像素 token 路”分数的来源，也是 PatchCAM 的原始形式。视图下方列出各类别占用的超像素个数。它不经过阈值、多尺度归一化和背景阈值，可能出现分割结果里没有的类别。

### 6. 常见错误

| 提示 | 处理 |
| --- | --- |
| 无法连接服务 | 检查 SSH 隧道是否仍在运行、服务地址和端口是否正确、后端 tmux 会话是否还在。 |
| 访问密钥无效 | 密钥与 `SEG_API_KEY` 不一致。 |
| 模型尚未就绪 | 后端仍在加载权重，稍后重新连接。 |
| 服务繁忙 | 排队已满（`SEG_QUEUE_CAPACITY`），稍后手动重试。 |
| 结果不存在或已过期 | 服务端结果默认保留 1 小时（`SEG_RESULT_TTL_SECONDS`）。本地缓存的副本不受影响，可以继续另存。 |

## 三、模型输出一致性检查（4090）

后端通过 `models.py` 中 `forward(..., return_att=True, return_extras=True)` 取两路分数、三种 CAM 和超像素类别分数。该参数默认关闭，不增加任何模型参数，旧 checkpoint 照常加载，训练和 CAM 生成脚本使用的三元组返回不变。用真实权重确认这一点：

```bash
cd ~/projects/model-drcformer
git fetch origin main
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/miniconda3/envs/py310/bin/python -m software.backend.compare_model_outputs \
    --checkpoint ~/projects/outputs/drcformer/<实验ID>/checkpoint.pth \
    --images ~/projects/voc2012/JPEGImages/2007_000032.jpg ~/projects/voc2012/JPEGImages/2007_000039.jpg
```

脚本把同一个 checkpoint 分别载入 `origin/main`（改动前，可用 `--reference-rev` 换成其他版本）和当前的 `models.py`，对三种注意力类型逐元素比较：当前三元组返回与改动前一致；dict 返回与三元组一致；`cams_all` 中每种 CAM 与单独指定该类型时一致；两路 logit 的平均等于最终 logit；ReLU 后的超像素类别分数回投到像素等于 PatchCAM。全部一致时输出 `all outputs identical` 并以 0 退出，否则列出不一致项并以 1 退出。`num_sp_tokens` 不是 196 时加 `--num-sp-tokens`。
