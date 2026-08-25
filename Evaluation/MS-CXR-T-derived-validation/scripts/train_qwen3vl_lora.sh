#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${TASK:-target}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR to your LlamaFactory checkout}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/llamafactory_configs}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
LLAMAFACTORY_SAVES="${LLAMAFACTORY_SAVES:-${LLAMAFACTORY_DIR}/saves}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache/huggingface}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${XDG_CACHE_HOME}/vllm}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${XDG_CACHE_HOME}/flashinfer}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

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

cd "${LLAMAFACTORY_DIR}"

run_train() {
  local name="$1"
  local template="$2"
  local config="${RENDER_DIR}/${name}.yaml"
  render_config "${template}" "${config}"
  echo "Starting ${name} fine-tuning with ${config}"
  llamafactory-cli train "${config}"
}

if [[ "${TASK}" == "target" || "${TASK}" == "both" ]]; then
  run_train "target" "${CONFIG_DIR}/qwen3vl_lora_mscxr_target.yaml"
fi

if [[ "${TASK}" == "progression" || "${TASK}" == "both" ]]; then
  run_train "progression" "${CONFIG_DIR}/qwen3vl_lora_mscxr_progression.yaml"
fi
