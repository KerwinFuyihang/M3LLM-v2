#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-target}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
CACHE_ROOT="${CACHE_ROOT:-cache/cxr_foundation}"
MODEL_DIR="${MODEL_DIR:-}"
QFORMER_MODEL_DIR="${QFORMER_MODEL_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

if [[ "${TASK}" == "target" ]]; then
  TRAIN_PATH="data/target_finding_identification/train.jsonl"
  TEST_PATH="data/target_finding_identification/test.jsonl"
elif [[ "${TASK}" == "progression" ]]; then
  TRAIN_PATH="data/progression_classification/train.jsonl"
  TEST_PATH="data/progression_classification/test.jsonl"
else
  echo "Unsupported TASK=${TASK}. Use target or progression."
  exit 1
fi

EMBEDDINGS="${CACHE_ROOT}/${TASK}_all_embeddings.npz"
OUT_DIR="results/cxr_foundation/${TASK}_finetune"

if [[ ! -f "${EMBEDDINGS}" ]]; then
  EXTRACT_ARGS=(
    --data_path "${TRAIN_PATH}"
    --data_path "${TEST_PATH}"
    --image_root "${IMAGE_ROOT}"
    --output_path "${EMBEDDINGS}"
  )
  if [[ -n "${HF_HOME:-}" ]]; then
    EXTRACT_ARGS+=(--cache_dir "${HF_HOME}")
  fi
  if [[ -n "${MODEL_DIR}" ]]; then
    EXTRACT_ARGS+=(--model_dir "${MODEL_DIR}")
    EXTRACT_ARGS+=(--qformer_model_dir "${QFORMER_MODEL_DIR}")
  fi
  python scripts/extract_cxr_foundation_embeddings.py "${EXTRACT_ARGS[@]}"
fi

python scripts/train_cxr_foundation.py \
  --train_data_path "${TRAIN_PATH}" \
  --train_embeddings "${EMBEDDINGS}" \
  --output_dir "${OUT_DIR}" \
  --task "$([[ "${TASK}" == "target" ]] && echo binary || echo progression)" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --num_workers "${NUM_WORKERS}"
