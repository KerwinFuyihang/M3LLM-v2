#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to a MedEvalKit wrapper name}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model checkpoint or HF repo id}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
MEDEVALKIT_PATH="${MEDEVALKIT_PATH:-../MedEvalKit}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_VLLM="${USE_VLLM:-False}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-0}"
TASK="${TASK:-target}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache/huggingface}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${XDG_CACHE_HOME}/vllm}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${XDG_CACHE_HOME}/flashinfer}"

OUT_ROOT="${OUT_ROOT:-results/zero_shot/${MODEL_NAME}}"

run_task() {
  local data_path="$1"
  local output_dir="$2"

  python scripts/run_medevalkit.py \
    --data_path "${data_path}" \
    --image_root "${IMAGE_ROOT}" \
    --output_dir "${output_dir}" \
    --medevalkit_path "${MEDEVALKIT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --use_vllm "${USE_VLLM}" \
    --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_image_num 2 \
    --max_new_tokens 128 \
    --temperature 0 \
    --top_p 0.0001 \
    --batch_size "${BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --max_image_side "${MAX_IMAGE_SIDE}"
}

if [[ "${TASK}" == "target" || "${TASK}" == "both" ]]; then
  run_task \
    data/target_finding_identification/test.jsonl \
    "${OUT_ROOT}/target_finding_test"
fi

if [[ "${TASK}" == "progression" || "${TASK}" == "both" ]]; then
  run_task \
    data/progression_classification/test.jsonl \
    "${OUT_ROOT}/progression_test"
fi
