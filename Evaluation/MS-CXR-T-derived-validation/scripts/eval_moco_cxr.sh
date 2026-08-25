#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-progression}"
IMAGE_ROOT="${IMAGE_ROOT:-${MSCXR_IMAGE_ROOT:-}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16}"

if [[ -z "${IMAGE_ROOT}" ]]; then
  echo "Set IMAGE_ROOT or MSCXR_IMAGE_ROOT to the MIMIC-CXR-JPG-sorted directory."
  exit 1
fi

case "${TASK}" in
  target)
    DATA_PATH="data/target_finding_identification/test.jsonl"
    ;;
  progression)
    DATA_PATH="data/progression_classification/test.jsonl"
    ;;
  *)
    echo "Unsupported TASK=${TASK}. Use target or progression."
    exit 1
    ;;
esac

python scripts/eval_moco_cxr.py \
  --data_path "${DATA_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --checkpoint_path "results/moco_cxr/${TASK}_finetune/model.pt" \
  --output_dir "results/moco_cxr/${TASK}_eval" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --precision "${PRECISION}"
