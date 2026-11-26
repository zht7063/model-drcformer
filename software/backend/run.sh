#!/usr/bin/env bash
set -euo pipefail

SERVER_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${SERVER_PROJECT_ROOT}"
SERVER_ENV_FILE=${SERVER_ENV_FILE:-${SERVER_PROJECT_ROOT}/software/.env}
if [[ -f "${SERVER_ENV_FILE}" ]]; then
    set -a
    source "${SERVER_ENV_FILE}"
    set +a
fi
SERVER_PYTHON=${SERVER_PYTHON:-/home/ubuntu/miniconda3/envs/py310/bin/python}
if [[ ${1:-} == --check-config && $# == 1 ]]; then
    exec "${SERVER_PYTHON}" -m software.backend.check_config
fi
if [[ $# != 0 ]]; then
    echo "Usage: bash software/backend/run.sh [--check-config]" >&2
    exit 2
fi
if [[ -z ${TMUX:-} ]]; then
    echo "Start inside tmux; see software/docs/deployment.md" >&2
    exit 2
fi
exec "${SERVER_PYTHON}" -m software.backend
