#!/bin/bash
# Evaluate MedGPT-oss / Hulu-Med.
# Usage:
#   bash eval_new_models.sh medgpt
#   bash eval_new_models.sh hulumed
#
# Env notes (separate conda envs recommended):
#   medgpt : transformers>=4.55, gated HF repo (HF_TOKEN), large GPU memory
#   hulumed: transformers==4.51.2, decord/ffmpeg-python/imageio/opencv-python

#SBATCH --job-name=eval-new-models
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=48:00:00

WHICH="${1:-hulumed}"
if [ "$WHICH" = "medgpt" ]; then
  MODEL_NAME="MedGPT-oss"
  MODEL_PATH="UFNLP/MedGPT-oss"
  OUTPUT_PATH="eval_results/new_models/medgpt-oss"
  export medgpt_max_tiles_per_image=12
else
  MODEL_NAME="Hulu-Med"
  MODEL_PATH="ZJU-AI4H/Hulu-Med-7B"
  OUTPUT_PATH="eval_results/new_models/hulu-med-7b"
fi

EVAL_DATASETS="${EVAL_DATASETS:-OmniMedVQA,MMMU-Medical-val}"
DATASETS_PATH="${DATASETS_PATH:-hf}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
USE_VLLM="False"

SEED=42
REASONING="False"
TEST_TIMES=1
EVAL_SAMPLE_NUM=0
EVAL_SAMPLE_SEED=42

MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

USE_LLM_JUDGE="${USE_LLM_JUDGE:-False}"
GPT_MODEL="${GPT_MODEL:-None}"
JUDGE_MODEL_TYPE="${JUDGE_MODEL_TYPE:-openai}"
API_KEY="${API_KEY:-None}"
BASE_URL="${BASE_URL:-None}"

python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed "$SEED" \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
    --eval_sample_num "$EVAL_SAMPLE_NUM" \
    --eval_sample_seed "$EVAL_SAMPLE_SEED"
