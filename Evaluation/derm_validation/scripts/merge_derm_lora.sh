#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-qwen3vl}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR to your LlamaFactory checkout}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/llamafactory_configs}"
LLAMAFACTORY_SAVES="${LLAMAFACTORY_SAVES:-${LLAMAFACTORY_DIR}/saves}"

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache/huggingface}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

RENDER_DIR="$(mktemp -d)"
cleanup() { rm -rf "${RENDER_DIR}"; }
trap cleanup EXIT

render_config() {
  local template="$1"
  local output="$2"
  sed \
    -e "s|@BENCHMARK_ROOT@|${ROOT}|g" \
    -e "s|@LLAMAFACTORY_SAVES@|${LLAMAFACTORY_SAVES}|g" \
    "${template}" > "${output}"
}

case "${MODEL}" in
  qwen3vl)
    TEMPLATE="${CONFIG_DIR}/merge_qwen3vl_derm_binary.yaml"
    ;;
  medgemma)
    TEMPLATE="${CONFIG_DIR}/merge_medgemma_derm_binary.yaml"
    ;;
  medgemma27b)
    TEMPLATE="${CONFIG_DIR}/merge_medgemma27b_derm_binary.yaml"
    ;;
  qwen2_5vl)
    TEMPLATE="${CONFIG_DIR}/merge_qwen2_5vl_derm_binary.yaml"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}."
    exit 1
    ;;
esac

CONFIG="${RENDER_DIR}/merge.yaml"
render_config "${TEMPLATE}" "${CONFIG}"

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli export "${CONFIG}"
