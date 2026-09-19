# PMC-MI instruction construction

This directory provides the five-step construction pipeline described in the
manuscript.  The prompt files are the searchable specifications reproduced in
the Supplementary Information.  Steps 1, 2, 4 and 5 use Qwen2.5-32B; Step 3
uses HuatuoGPT-Vision-34B to describe each constituent subimage separately.

## Contents

- `scripts/step1_inline_summary.py`: summarize figure-linked article passages.
- `scripts/step2_medical_knowledge.py`: extract keywords and add concise medical
  background.
- `scripts/step3_visual_description.py`: generate a grounded description for
  each constituent subimage.
- `scripts/step4_*.py`: six task-specific instruction-construction routes.
- `scripts/step5_refine_context.py`: remove answer-bearing information from
  auxiliary context. Relative-position and multiple-choice records are skipped
  because these formats do not contain an auxiliary context.
- `prompts/`: canonical prompt text for every step and Step 4 task format.

The supplied historical scripts used local Hugging Face checkpoints and fixed
cluster paths.  The release scripts retain a local-checkpoint route and also
support an OpenAI-compatible endpoint, including a locally served vLLM model.
No API key or filesystem path is embedded in the code.

## Installation

For text-only local inference:

```bash
pip install -r DataConstruction/requirements.txt
```

The `huatuo-local` backend reuses the HuatuoGPT-Vision implementation under
`Evaluation/MedEvalKit/models/HuatuoGPT` and requires the dependencies specified
for that model. Alternatively, serve HuatuoGPT-Vision-34B through a compatible
multimodal endpoint and select `--backend openai-compatible`.

## Input and output

Every command accepts either a JSON array or JSONL input and writes JSONL. The
input records may use the historical field names (`caption`, `references`,
`Inline Summary`, `medical knowledge`, and `subcaptions`) or the documented
snake-case aliases. Each subimage record should include an image identifier,
caption, and, after Step 3, `visual perception description`.

Outputs retain the source record and append the generated fields. Each output
line has a stable `_construction_key`. Rerunning a command resumes from an
existing output; use `--overwrite` to start again. Records that cannot satisfy a
task schema and generation failures are written to `<output>.errors.jsonl`.
Pass `--include-raw` to retain model text before parsing.

Generation is greedy by default (`temperature=0`). Steps 1, 2, 3 and 5 allow
512 new tokens, and the Step 4 task scripts allow 1,024; all values can be
overridden from the command line.

Step 3 resolves an image from an absolute path, `--image-root`, or an image
manifest. A JSON manifest maps image identifiers to paths; a CSV manifest uses
`image` (or `filename`) and `path` columns.

For the single-subimage, multi-subimage, and multiple-choice routes, an optional
classification CSV can reproduce the historical medical-subimage selection
(`filename` and `prob` columns; probability greater than 0.70 by default).
Multi-subimage VQA then samples two or three eligible subimages with the
specified seed. Relative-position VQA consumes bounding-box centres and a
precomputed relation; the language model only verbalizes that relation.

## Example

Run the text steps with the local Hugging Face checkpoint:

```bash
python DataConstruction/scripts/step1_inline_summary.py \
  --input records.jsonl --output step1.jsonl

python DataConstruction/scripts/step2_medical_knowledge.py \
  --input step1.jsonl --output step2.jsonl
```

Run Step 3 through a local OpenAI-compatible multimodal server:

```bash
python DataConstruction/scripts/step3_visual_description.py \
  --input step2.jsonl --output step3.jsonl \
  --backend openai-compatible \
  --base-url http://localhost:8000/v1 \
  --model HuatuoGPT-Vision-34B \
  --image-manifest image_paths.csv
```

Construct each Step 4 format independently. For example:

```bash
python DataConstruction/scripts/step4_multi_subimage_vqa.py \
  --input step3.jsonl --output multi_subimage.jsonl \
  --classification-csv medical_subimage_scores.csv --seed 42

python DataConstruction/scripts/step4_text_only_qa.py \
  --input step3.jsonl --output text_only.jsonl
```

Then refine the auxiliary context for formats that use it:

```bash
python DataConstruction/scripts/step5_refine_context.py \
  --input multi_subimage.jsonl --output multi_subimage_refined.jsonl
```

To use a local vLLM endpoint for any text step, add:

```text
--backend openai-compatible --base-url http://localhost:8000/v1 \
--model Qwen/Qwen2.5-32B-Instruct-AWQ
```

The endpoint reads `OPENAI_API_KEY` by default; a local server that does not
authenticate can use any placeholder value.
