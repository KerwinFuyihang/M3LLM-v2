# Training

The released training recipes follow the manuscript's three-stage post-training workflow. The [versioned data release](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0) contains the supervised instruction records and separate policy-optimisation training and validation records. Configure local dataset and image paths before running a stage.

| Stage | Recipe | Implementation |
| --- | --- | --- |
| I · MI-SFT | Multi-image supervised fine-tuning | [`SFT/configs/stage1_mi_sft.yaml`](SFT/configs/stage1_mi_sft.yaml) |
| II · GRPO | Selection-aware policy refinement | [`GRPO/train_grpo.sh`](GRPO/train_grpo.sh) |
| III · Full-SFT | Full-mixture supervised fine-tuning | [`SFT/configs/stage3_full_sft.yaml`](SFT/configs/stage3_full_sft.yaml) |

## Stage I and Stage III: supervised fine-tuning

The SFT configurations target Qwen3-VL-8B and use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), installed separately. Stage I starts from `Qwen/Qwen3-VL-8B-Instruct`; Stage III starts from the Stage II checkpoint. Register the local training files using [`SFT/configs/dataset_info.example.json`](SFT/configs/dataset_info.example.json), then adapt `dataset_dir`, checkpoint, and output paths in the YAML files to your environment. The recipes contain the remaining training parameters.

## Stage II: policy refinement

The GRPO recipe uses [EasyR1](https://github.com/hiyouga/EasyR1), installed separately. [`GRPO/config.yaml`](GRPO/config.yaml) contains the rollout and optimisation settings. [`GRPO/prompts/m3llm_response.jinja`](GRPO/prompts/m3llm_response.jinja) specifies the response format, and [`GRPO/reward/m3llm_reward.py`](GRPO/reward/m3llm_reward.py) implements the reward. [`GRPO/scripts/prepare_policy_data.py`](GRPO/scripts/prepare_policy_data.py) prepares the policy records.

Run [`GRPO/train_grpo.sh`](GRPO/train_grpo.sh) after setting `MODEL_PATH`, `TRAIN_FILE`, and `VAL_FILE`; set `IMAGE_DIR` when image references require a local root. The script accepts optional environment variables for the checkpoint destination and training settings. The model path should point to the Stage I checkpoint, and Stage III should use the resulting Stage II checkpoint.

Framework versions and third-party provenance are listed in [`THIRD_PARTY.md`](../THIRD_PARTY.md). Model weights and data have their own release terms.
