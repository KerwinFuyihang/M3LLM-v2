#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to a MedEvalKit wrapper name}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model checkpoint or HF repo id}"
IMAGE_ROOT="${IMAGE_ROOT:-${DERM_IMAGE_ROOT:-}}"
MEDEVALKIT_PATH="${MEDEVALKIT_PATH:-../MedEvalKit}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
USE_VLLM="${USE_VLLM:-False}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-0}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-8}"
RESUME="${RESUME:-false}"
INTERNVL_MAX_TILES_PER_IMAGE="${INTERNVL_MAX_TILES_PER_IMAGE:-6}"
TASK="${TASK:-differential}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or DERM_IMAGE_ROOT to the dermatology image directory."
  exit 1
fi

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache/huggingface}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${XDG_CACHE_HOME}/vllm}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${XDG_CACHE_HOME}/flashinfer}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi

if [[ "${MODEL_NAME}" == "InternVL" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export internvl_max_tiles_per_image="${INTERNVL_MAX_TILES_PER_IMAGE}"
fi

if [[ -z "${OUT_NAME:-}" ]]; then
  _path="${MODEL_PATH%/}"
  if [[ "${_path}" == /* ]]; then
    OUT_NAME="$(basename "${_path}")"
    _parent="$(basename "$(dirname "${_path}")")"
    if [[ "${OUT_NAME}" =~ ^(sft|sft_merged|checkpoint-.+|test)$ ]]; then
      OUT_NAME="${_parent}__${OUT_NAME}"
    fi
  elif [[ "${_path}" == */* ]]; then
    OUT_NAME="${_path//\//__}"
  else
    OUT_NAME="${_path}"
  fi
fi

DATA_PATH="${DATA_PATH:-data/binary_differential/test.jsonl}"
DEFAULT_ROOT="results/differential_diagnosis"

OUT_ROOT="${RESULTS_ROOT:-${DEFAULT_ROOT}}/${OUT_NAME}"
echo "Output directory: ${OUT_ROOT}/test"

python scripts/run_medevalkit.py \
  --data_path "${DATA_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUT_ROOT}/test" \
  --medevalkit_path "${MEDEVALKIT_PATH}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --use_vllm "${USE_VLLM}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --max_image_num "${MAX_IMAGE_NUM}" \
  --max_new_tokens 128 \
  --temperature 0 \
  --top_p 0.0001 \
  --batch_size "${BATCH_SIZE}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_image_side "${MAX_IMAGE_SIDE}" \
  $([[ "${RESUME}" == "true" ]] && echo --resume)
