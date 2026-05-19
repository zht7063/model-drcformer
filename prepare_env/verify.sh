#!/usr/bin/env bash
# DRCFormer 环境校验脚本
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "==> Python: $(python --version)"
echo "==> PyTorch / CUDA:"
python -c "import torch; print('  version:', torch.__version__); print('  cuda:', torch.version.cuda); print('  available:', torch.cuda.is_available())"

echo "==> 核心依赖 import:"
python -c "
import torch_scatter
import pydensecrf.densecrf
import timm
import einops
import fast_slic
import numpy as np
print('  torch_scatter, pydensecrf, timm, einops, fast_slic, numpy OK')
print('  numpy:', np.__version__)
"

echo "==> 项目入口:"
cd "${PROJECT_ROOT}/src/drcformer"
python main.py --help >/dev/null
echo "  main.py --help OK"

echo "==> 校验通过"
