#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-qwen3vl}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR to your LlamaFactory checkout}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/llamafactory_configs}"
IMAGE_ROOT="${IMAGE_ROOT:-${DERM_IMAGE_ROOT:-}}"
LLAMAFACTORY_SAVES="${LLAMAFACTORY_SAVES:-${LLAMAFACTORY_DIR}/saves}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-8}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or DERM_IMAGE_ROOT to the dermatology image directory."
  exit 1
fi

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache/huggingface}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

python "${ROOT}/scripts/prepare_llamafactory_sft.py" \
  --benchmark_dir "${ROOT}" \
  --output_dir "${ROOT}/llamafactory_data" \
  --task differential \
  --max_image_num "${MAX_IMAGE_NUM}"

RENDER_DIR="$(mktemp -d)"
cleanup() { rm -rf "${RENDER_DIR}"; }
trap cleanup EXIT

render_config() {
  local template="$1"
  local output="$2"
  sed \
    -e "s|@BENCHMARK_ROOT@|${ROOT}|g" \
    -e "s|@IMAGE_ROOT@|${IMAGE_ROOT}|g" \
    -e "s|@LLAMAFACTORY_SAVES@|${LLAMAFACTORY_SAVES}|g" \
    "${template}" > "${output}"
}

case "${MODEL}" in
  qwen3vl)
    TEMPLATE="${CONFIG_DIR}/qwen3vl_lora_derm_binary.yaml"
    ;;
  medgemma)
    TEMPLATE="${CONFIG_DIR}/medgemma_lora_derm_binary.yaml"
    ;;
  medgemma27b)
    TEMPLATE="${CONFIG_DIR}/medgemma27b_lora_derm_binary.yaml"
    ;;
  qwen2_5vl)
    TEMPLATE="${CONFIG_DIR}/qwen2_5vl_lora_derm_binary.yaml"
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}."
    echo "MODEL: qwen3vl | medgemma | medgemma27b | qwen2_5vl"
    exit 1
    ;;
esac

CONFIG="${RENDER_DIR}/train.yaml"
render_config "${TEMPLATE}" "${CONFIG}"

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train "${CONFIG}"
