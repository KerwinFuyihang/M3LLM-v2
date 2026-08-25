#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the released model checkpoint}"
TEMPLATE="${TEMPLATE:?Set TEMPLATE to the matching LLaMA-Factory chat template}"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME from dataset_info.json}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR containing dataset_info.json}"
IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT to the authorized local image directory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the final adapter}"

CONFIG="$(mktemp "${TMPDIR:-/tmp}/m3llm-lora.XXXXXX.yaml")"
cleanup() { rm -f "${CONFIG}"; }
trap cleanup EXIT

sed \
  -e "s|@MODEL_PATH@|${MODEL_PATH}|g" \
  -e "s|@TEMPLATE@|${TEMPLATE}|g" \
  -e "s|@DATASET_NAME@|${DATASET_NAME}|g" \
  -e "s|@DATASET_DIR@|${DATASET_DIR}|g" \
  -e "s|@IMAGE_ROOT@|${IMAGE_ROOT}|g" \
  -e "s|@OUTPUT_DIR@|${OUTPUT_DIR}|g" \
  "${ROOT}/Evaluation/common/clinical_lora_sft.yaml" > "${CONFIG}"

cd "${LLAMAFACTORY_DIR}"
llamafactory-cli train "${CONFIG}"
