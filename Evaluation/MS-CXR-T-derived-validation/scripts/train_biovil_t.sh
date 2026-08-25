#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-progression}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-32}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"
PRECISION="${PRECISION:-bf16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PRETRAINED_ENCODER_CHECKPOINT="${PRETRAINED_ENCODER_CHECKPOINT:-}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

FINDINGS=(
  "consolidation"
  "edema"
  "pleural effusion"
  "pneumonia"
  "pneumothorax"
)

if [[ "${TASK}" == "target" ]]; then
  TRAIN_PATH="data/target_finding_identification/train.jsonl"
elif [[ "${TASK}" == "progression" ]]; then
  TRAIN_PATH="data/progression_classification/train.jsonl"
else
  echo "Unsupported TASK=${TASK}. Use target or progression."
  exit 1
fi

run_finding() {
  local finding="$1"
  local slug="${finding// /_}"
  local out_dir="results/biovil_t/${TASK}_finetune/${slug}"
  local args=(
    --train_data_path "${TRAIN_PATH}"
    --image_root "${IMAGE_ROOT}"
    --output_dir "${out_dir}"
    --task "$([[ "${TASK}" == "target" ]] && echo binary || echo progression)"
    --finding "${finding}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
    --epochs "${EPOCHS}"
    --learning_rate "${LR}"
    --precision "${PRECISION}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --num_workers "${NUM_WORKERS}"
    --val_ratio "${VAL_RATIO}"
  )
  if [[ -n "${PRETRAINED_ENCODER_CHECKPOINT}" ]]; then
    args+=(--pretrained_encoder_checkpoint "${PRETRAINED_ENCODER_CHECKPOINT}")
  fi
  if [[ "${FREEZE_ENCODER}" == "True" ]]; then
    args+=(--freeze_encoder)
  fi
  echo "=== Training BioViL-T: task=${TASK}, finding=${finding} ==="
  python scripts/train_biovil_t.py "${args[@]}"
}

if [[ -n "${FINDING:-}" ]]; then
  run_finding "${FINDING}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_finding "${FINDINGS[${SLURM_ARRAY_TASK_ID}]}"
else
  for finding in "${FINDINGS[@]}"; do
    run_finding "${finding}"
  done
fi
