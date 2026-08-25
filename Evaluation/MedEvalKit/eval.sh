#!/bin/bash

EVAL_DATASETS="${EVAL_DATASETS:-OmniMedVQA}"
DATASETS_PATH="${DATASETS_PATH:-hf}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_results/OmniMedVQA}"

MODEL_NAME="${MODEL_NAME:-Qwen3-VL}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
USE_VLLM="${USE_VLLM:-False}"

SEED="${SEED:-42}"
REASONING="${REASONING:-False}"
TEST_TIMES="${TEST_TIMES:-1}"
EVAL_SAMPLE_NUM="${EVAL_SAMPLE_NUM:-0}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-42}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_IMAGE_NUM="${MAX_IMAGE_NUM:-8}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.0001}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1}"

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
