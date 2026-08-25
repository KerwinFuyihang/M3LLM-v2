#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the actor checkpoint or HF repo id}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the EasyR1 train JSONL}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the EasyR1 val JSONL}"
IMAGE_DIR="${IMAGE_DIR:-}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-checkpoints/m3llm_policy_refinement}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_vl_8b_policy_refinement}"

N_GPUS="${N_GPUS:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${N_GPUS}}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
LR="${LR:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-15000}"
FORMAT_PROMPT="${FORMAT_PROMPT:-${ROOT}/prompts/m3llm_response.jinja}"
REWARD_FUNCTION="${REWARD_FUNCTION:-${ROOT}/reward/m3llm_reward.py:compute_score}"

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache/huggingface}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${XDG_CACHE_HOME}/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${XDG_CACHE_HOME}/vllm_config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${XDG_CACHE_HOME}/torch/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${XDG_CACHE_HOME}/flashinfer}"

RAY_TMP_SHORT="${RAY_TMP_SHORT:-/tmp/ru${UID}}"
mkdir -p "${RAY_TMP_SHORT}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TMP_SHORT}}"
export TMPDIR="${TMPDIR:-${RAY_TMP_SHORT}}"
export TEMP="${TEMP:-${RAY_TMP_SHORT}}"
mkdir -p "${VLLM_CACHE_DIR}" "${VLLM_CONFIG_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${SAVE_CHECKPOINT_PATH}"

EXTRA_ARGS=()
if [[ -n "${IMAGE_DIR}" ]]; then
  EXTRA_ARGS+=(data.image_dir="${IMAGE_DIR}")
fi

python3 -m verl.trainer.main \
    config="${ROOT}/config.yaml" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.lora.rank=0 \
    worker.actor.optim.lr="${LR}" \
    worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    worker.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    worker.reward.reward_function="${REWARD_FUNCTION}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_checkpoint_path="${SAVE_CHECKPOINT_PATH}" \
    "${EXTRA_ARGS[@]}"
