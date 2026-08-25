#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-target}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT_DIR="${CKPT_DIR:-results/biovil_t/${TASK}_finetune}"
OUT_DIR="${OUT_DIR:-results/biovil_t/${TASK}_eval}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

if [[ "${TASK}" == "target" ]]; then
  DATA_PATH="data/target_finding_identification/test.jsonl"
elif [[ "${TASK}" == "progression" ]]; then
  DATA_PATH="data/progression_classification/test.jsonl"
else
  echo "Unsupported TASK=${TASK}. Use target or progression."
  exit 1
fi

python scripts/eval_biovil_t.py \
  --data_path "${DATA_PATH}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --output_dir "${OUT_DIR}" \
  --image_root "${IMAGE_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --per_finding
