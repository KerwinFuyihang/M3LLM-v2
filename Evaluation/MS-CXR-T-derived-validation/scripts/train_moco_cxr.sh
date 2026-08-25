#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-progression}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
CHECKPOINT="${CHECKPOINT:-cache/moco_cxr/r8w-00001-v2.pth.tar}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
VAL_RATIO="${VAL_RATIO:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
DISEASE_EMBEDDING_DIM="${DISEASE_EMBEDDING_DIM:-64}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing MoCo-CXR checkpoint: ${CHECKPOINT}"
  echo "Run: python scripts/download_moco_cxr_checkpoint.py"
  exit 1
fi

case "${TASK}" in
  target)
    TRAIN_PATH="data/target_finding_identification/train.jsonl"
    ;;
  progression)
    TRAIN_PATH="data/progression_classification/train.jsonl"
    ;;
  *)
    echo "Unsupported TASK=${TASK}. Use target or progression."
    exit 1
    ;;
esac

OUTPUT_DIR="results/moco_cxr/${TASK}_finetune"
echo "=== Joint MoCo-CXR fine-tuning: task=${TASK} ==="
python scripts/train_moco_cxr.py \
  --train_data_path "${TRAIN_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --checkpoint_path "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --task "$([[ "${TASK}" == "target" ]] && echo binary || echo progression)" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --num_workers "${NUM_WORKERS}" \
  --precision "${PRECISION}" \
  --disease_embedding_dim "${DISEASE_EMBEDDING_DIM}" \
  --seed "${SEED}"
