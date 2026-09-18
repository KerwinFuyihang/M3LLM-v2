# Open-weight response-quality judge

This directory provides an open-weight alternative for the complementary LLM-as-a-judge analysis. It applies the manuscript rubric to a question, a manually verified reference answer, and one anonymized candidate response. Images, auxiliary context, and candidate-model identity are not sent to the judge.

The default configuration runs `deepseek-ai/DeepSeek-V4-Flash` through a local OpenAI-compatible vLLM endpoint. The same runner can use another compatible local server or API by changing the configuration file. Results reported in the manuscript were produced by the three judges specified there; this configuration provides an open substitute using the same evaluation rubric.

## Input

Prepare JSONL records with these fields:

```json
{"id":"item-001","question":"...","reference_answer":"...","candidate_response":"...","candidate_model":"optional metadata"}
```

`candidate_model` is retained in the output for aggregation but is not included in the judge prompt.

## Run locally

```bash
pip install -r requirements.txt
pip install vllm
vllm serve deepseek-ai/DeepSeek-V4-Flash --served-model-name deepseek-ai/DeepSeek-V4-Flash

python scripts/run_judge.py \
  --input examples/input.jsonl \
  --output results/judgements.jsonl \
  --config config/open_weight_deepseek_v4_flash.json

python scripts/summarize_scores.py \
  --input results/judgements.jsonl \
  --output results/summary.json
```

The exact system prompt, item template, model identifier, decoding parameters, output schema, retry limit, and endpoint settings are stored under `prompts/` and `config/`. Set `OPENAI_API_KEY` only when the selected endpoint requires authentication; a local vLLM server accepts the default placeholder key.
