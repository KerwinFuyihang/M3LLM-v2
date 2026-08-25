# M3LLM code release

This repository contains the training and evaluation code accompanying M3LLM. It intentionally contains no PMC-MI, PMC-MI-Bench, MS-CXR-T-derived, or YNHH patient data.

## Repository layout

- `Training/SFT/`: Stage I multi-image SFT and Stage III full-mixture SFT configurations.
- `Training/GRPO/`: Stage II policy-refinement data conversion, response schema, reward function, and launch configuration.
- `Evaluation/PMC-MI-Bench/`: six-task PMC-MI-Bench inference and automatic metrics.
- `Evaluation/MedEvalKit/`: MedEvalKit-based OmniMedVQA and MMMU-Med evaluation.
- `Evaluation/MS-CXR-T-derived-validation/`: derived-task construction, MLLM adaptation/inference, and specialized chest-radiograph baselines.
- `Evaluation/derm_validation/`: YNHH differential-diagnosis validity adaptation, inference, and metrics.
- `Evaluation/common/`: shared LoRA template for the matched clinical task-adaptation protocol.

## Environment

The main SFT experiments used LLaMA-Factory `0.9.5.dev0` (`v0.9.4-48-g184304b5`) with DeepSpeed ZeRO-3. Stage II used EasyR1 `v0.3.2-41-gdd71bbd` and verl `0.3.3.dev0`. Public-benchmark evaluation used MedEvalKit commit `9b12e3b`. Install each framework from its upstream repository at the stated revision, then install the requirements in the relevant evaluation directory.

## Three-stage training

Register locally prepared JSONL files with LLaMA-Factory using `Training/SFT/configs/dataset_info.example.json`. Each Stage I input contains the compound figure followed by at most three constituent subimages. Run:

```bash
llamafactory-cli train Training/SFT/configs/stage1_mi_sft.yaml
llamafactory-cli train Training/SFT/configs/stage3_full_sft.yaml
```

Both configurations use one epoch, full-parameter optimization, a 2,048-token cutoff, bfloat16, learning rate `2e-6`, cosine decay, 3% warm-up, and an effective global batch size of 16 on two GPUs. They use the final training checkpoint and no validation split.

Prepare the 10,361/200 Stage II optimization/validation files separately with:

```bash
python Training/GRPO/scripts/prepare_policy_data.py \
  --input policy_instances.jsonl \
  --output policy_train.jsonl \
  --image-root /path/to/images
```

Launch Stage II from an EasyR1 environment:

```bash
MODEL_PATH=/path/to/stage1 \
TRAIN_FILE=/path/to/policy_train.jsonl \
VAL_FILE=/path/to/policy_validation.jsonl \
IMAGE_DIR=/path/to/images \
bash Training/GRPO/train_grpo.sh
```

The default reward operator is `0.6 * answer + 0.3 * selection + 0.1 * format`.

## PMC-MI-Bench

Place benchmark JSON/JSONL files under `Evaluation/PMC-MI-Bench/data/` and images under `Evaluation/PMC-MI-Bench/images/`, or set `PMC_BENCH_IMAGE_ROOT`. The task prompts in `task_config.py` match those reported in the manuscript.

```bash
cd Evaluation/PMC-MI-Bench
python evaluate_all.py \
  --model_name qwen3-vl \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --task multisubimageVQA
```

Valid tasks are `puretext`, `multi-choice`, `single-subimageVQA`, `bboxVQA`, `compoundVQA`, and `multisubimageVQA`. Inference uses greedy decoding, temperature 0, top-p 1, repetition penalty 1, and 2,048 new tokens.

## OmniMedVQA and MMMU-Med

The included MedEvalKit launcher defaults to the manuscript inference settings (seed 42, no reasoning mode, one run, temperature 0, top-p `0.0001`, repetition penalty 1, 8,192 new tokens, and at most eight images).

```bash
cd Evaluation/MedEvalKit
EVAL_DATASETS=OmniMedVQA MODEL_NAME=Qwen3-VL \
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash eval.sh

EVAL_DATASETS=MMMU-Medical-val OUTPUT_PATH=eval_results/MMMU-Medical-val \
MODEL_NAME=Qwen3-VL MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash eval.sh
```

## MS-CXR-T-derived evaluation

The source MIMIC-CXR images are not redistributed. Build the two derived tasks from authorized local inputs:

```bash
cd Evaluation/MS-CXR-T-derived-validation
python scripts/build_benchmark.py \
  --csv_path /path/to/MS-CXR-T.csv \
  --image_root /path/to/MIMIC-CXR-JPG \
  --output_dir data
```

The builder creates a patient-level 1:1 split with seed 42; no patient can
occur in both partitions. For the 1,326-example benchmark this yields exactly
663 training and 663 test examples. Patient groups are shuffled with the fixed
seed and assigned without splitting a patient; construction stops with an error
if the input cannot support an exact half split. Each temporal image-label
example produces one progression item and five target-finding queries. The
builder writes the partition counts, patient counts and progression-class
distribution to `data/summary.json`. Convert training records and run LoRA
adaptation:

```bash
python scripts/prepare_llamafactory_sft.py
LLAMAFACTORY_DIR=/path/to/LLaMA-Factory IMAGE_ROOT=/path/to/images \
TASK=both bash scripts/train_qwen3vl_lora.sh
```

Run a generative model without task adaptation with `scripts/run_zero_shot.sh`. Set `TASK=target`, `progression`, or `both`; `MODEL_NAME` must be a MedEvalKit wrapper name and `MODEL_PATH` a released checkpoint or merged adapter.

The specialized baselines use the released MoCo-CXR repository checkpoint, `google/cxr-foundation`, and `microsoft/BiomedVLP-BioViL-T`. Their training and evaluation entry points are `train_*` and `eval_*` under `scripts/`.

For another generative baseline, apply the same LoRA protocol with its native
LLaMA-Factory template:

```bash
LLAMAFACTORY_DIR=/path/to/LLaMA-Factory \
MODEL_PATH=/path/to/released/checkpoint TEMPLATE=model_template \
DATASET_NAME=mscxr_progression_train \
DATASET_DIR="$PWD/llamafactory_data" IMAGE_ROOT=/path/to/images \
OUTPUT_DIR=/path/to/adapter bash ../common/train_clinical_lora.sh
```

## YNHH dermatology evaluation

YNHH images and annotations are not included because they contain restricted clinical data. Authorized users should provide the private JSONL split and image directory locally. Cases with more than eight photographs must be excluded before conversion; images are never subsampled by this code.

```bash
cd Evaluation/derm_validation
python scripts/prepare_llamafactory_sft.py \
  --benchmark_dir . --output_dir llamafactory_data --task differential

LLAMAFACTORY_DIR=/path/to/LLaMA-Factory DERM_IMAGE_ROOT=/path/to/images \
MODEL=qwen3vl bash scripts/train_derm_lora.sh

MODEL_NAME=Qwen3-VL MODEL_PATH=/path/to/checkpoint \
DERM_IMAGE_ROOT=/path/to/images bash scripts/run_zero_shot.sh
```

The released dermatology workflow includes only differential-diagnosis validity. Accuracy, balanced accuracy, and macro-F1 count unparseable outputs as incorrect.

The generic launcher in `Evaluation/common/` can adapt another generative
baseline by setting `DATASET_NAME=derm_binary_train` and using that model's
native LLaMA-Factory template.

## Data placement

No data files are bundled. Before public release, distribute PMC-derived text and images only according to the accompanying licence manifest. Restricted MIMIC-CXR and YNHH data remain outside this repository. Third-party provenance is summarized in `THIRD_PARTY.md`.


