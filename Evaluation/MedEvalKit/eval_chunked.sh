#!/bin/bash
# Multi-GPU chunked evaluation.

EVAL_DATASETS="${EVAL_DATASETS:-Medbullets_op4}"
DATASETS_PATH="${DATASETS_PATH:-hf}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_results/{}}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-False}"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=${CHUNKS:-$TOTAL_GPUS}

SEED=42
REASONING="False"
TEST_TIMES=1

MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

USE_LLM_JUDGE="${USE_LLM_JUDGE:-False}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
JUDGE_GPT_MODEL="${JUDGE_GPT_MODEL:-gpt-4.1-2025-04-14}"

for IDX in $(seq 0 $((CHUNKS - 1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed "$SEED" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --num_chunks "$CHUNKS" \
    --chunk_idx "$IDX" \
    --reasoning "$REASONING" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$JUDGE_GPT_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES" &
done

wait
