# M3LLM

Official code for training and evaluating M3LLM, a medical multi-image multimodal language model.

## Contents

- `Training/SFT`: Stage I multi-image SFT and Stage III full-mixture SFT.
- `Training/GRPO`: Stage II selection-aware policy refinement.
- `Evaluation/PMC-MI-Bench`: PMC-MI-Bench inference and reference-based metrics.
- `Evaluation/LLM-Judge`: response-quality evaluation, including the open-weight DeepSeek-V4-Flash judge.
- `Evaluation/MedEvalKit`: OmniMedVQA and MMMU-Med evaluation.
- `Evaluation/MS-CXR-T-derived-validation`: longitudinal chest-radiograph evaluation.
- `Evaluation/derm_validation`: YNHH dermatology evaluation.

Install the requirements in the relevant subdirectory and follow its scripts or configuration files. Data are released separately through [KerwinFu/M3LLM-data-v1.0.0](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0). Restricted clinical data are not included in this repository.

Third-party components and data provenance are summarized in [`THIRD_PARTY.md`](THIRD_PARTY.md).
