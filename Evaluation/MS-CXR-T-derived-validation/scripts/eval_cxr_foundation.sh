#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-target}"
CACHE_ROOT="${CACHE_ROOT:-cache/cxr_foundation}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ "${TASK}" == "target" ]]; then
  DATA_PATH="data/target_finding_identification/test.jsonl"
elif [[ "${TASK}" == "progression" ]]; then
  DATA_PATH="data/progression_classification/test.jsonl"
else
  echo "Unsupported TASK=${TASK}. Use target or progression."
  exit 1
fi

python scripts/eval_cxr_foundation.py \
  --data_path "${DATA_PATH}" \
  --embeddings "${CACHE_ROOT}/${TASK}_all_embeddings.npz" \
  --checkpoint_dir "results/cxr_foundation/${TASK}_finetune" \
  --output_dir "results/cxr_foundation/${TASK}_eval" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"
