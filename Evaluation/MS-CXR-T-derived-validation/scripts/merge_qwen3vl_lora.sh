#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${TASK:-target}"
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

cd "${LLAMAFACTORY_DIR}"

run_export() {
  local name="$1"
  local template="$2"
  local config="${RENDER_DIR}/${name}.yaml"
  render_config "${template}" "${config}"
  echo "Merging ${name} LoRA with ${config}"
  llamafactory-cli export "${config}"
}

if [[ "${TASK}" == "target" || "${TASK}" == "both" ]]; then
  run_export "target" "${CONFIG_DIR}/merge_qwen3vl_mscxr_target.yaml"
fi

if [[ "${TASK}" == "progression" || "${TASK}" == "both" ]]; then
  run_export "progression" "${CONFIG_DIR}/merge_qwen3vl_mscxr_progression.yaml"
fi
