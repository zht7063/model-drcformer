#!/usr/bin/env bash
# DRCFormer 环境一键安装：复现 py310（CUDA 12.1 + cu121 PyTorch）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-py310}"
CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "${CONDA_BIN}" &>/dev/null; then
  echo "错误: 未找到 conda，请先安装 Miniconda/Anaconda 或将 CONDA_BIN 指向 conda 可执行文件"
  exit 1
fi

echo "==> 目标环境: ${ENV_NAME}"
echo "==> 工作目录: ${SCRIPT_DIR}"

# 可选：检查 GPU（不阻断安装）
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
else
  echo "警告: 未检测到 nvidia-smi，将仍安装 GPU 版 PyTorch（cu121）"
fi

# 1. 创建 conda 环境
if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> 环境 ${ENV_NAME} 已存在，执行 conda env update ..."
  "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${SCRIPT_DIR}/conda-env.yml" --prune -y
else
  echo "==> 创建 conda 环境 ..."
  "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/conda-env.yml" -y
fi

run_in_env() {
  "${CONDA_BIN}" run -n "${ENV_NAME}" "$@"
}

# 2. PyTorch (cu121)
echo "==> 安装 PyTorch (cu121) ..."
run_in_env pip install \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 3. torch-scatter (PyG wheel)
echo "==> 安装 torch-scatter ..."
run_in_env pip install torch-scatter==2.1.2+pt25cu121 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# 4. fast-slic（需 Cython + --no-build-isolation）
echo "==> 安装 fast-slic ..."
run_in_env pip install Cython==3.2.4
run_in_env pip install fast-slic==0.4.0 --no-build-isolation

# 5. 其余 pip 包（排除已安装项）
echo "==> 安装 pip.lock 中其余依赖 ..."
FILTERED_LOCK="$(mktemp)"
grep -vE '^(#|$)' "${SCRIPT_DIR}/pip.lock" | while IFS= read -r line; do
  pkg="${line%%==*}"
  pkg_lower="$(echo "${pkg}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
  skip=0
  while IFS= read -r excl; do
    [[ -z "${excl}" || "${excl}" =~ ^# ]] && continue
    excl_lower="$(echo "${excl}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${pkg_lower}" == "${excl_lower}" ]]; then
      skip=1
      break
    fi
  done < "${SCRIPT_DIR}/pip.exclude"
  [[ "${skip}" -eq 0 ]] && echo "${line}"
done > "${FILTERED_LOCK}"

run_in_env pip install -r "${FILTERED_LOCK}"
rm -f "${FILTERED_LOCK}"

# 6. 校验
echo "==> 运行环境校验 ..."
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME}" PROJECT_ROOT="${PROJECT_ROOT}" \
  "${CONDA_BIN}" run -n "${ENV_NAME}" bash "${SCRIPT_DIR}/verify.sh"

echo ""
echo "安装完成。激活环境: conda activate ${ENV_NAME}"
