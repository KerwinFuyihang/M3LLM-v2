# M3LLM

**From Compound Figures to Medical Multi-image Reasoning: Scaling Multimodal Large Language Models with Biomedical Literature**

M3LLM is a medical multimodal language model developed for reasoning across multiple images. This repository contains the PMC-MI instruction-construction pipeline, the three-stage post-training recipes, and the evaluation code accompanying the manuscript.

- **Data and benchmark:** [PMC-MI and PMC-MI-Bench](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0)
- **Versioned code:** [v1.0.2](https://github.com/Yale-BIDS-Chen-Lab/M3LLM-v2/tree/v1.0.2)
- **Upstream source resource:** [MedPMC](https://github.com/Yale-BIDS-Chen-Lab/MedPMC)

## Quick start

1. Explore [DataConstruction](DataConstruction/README.md) for the five-step PMC-MI workflow, exact prompts, and example commands.
2. See [Training](Training/README.md) for multi-image SFT, selection-aware policy refinement, and full-mixture SFT.
3. See [Evaluation](Evaluation/README.md) for PMC-MI-Bench, public benchmarks, response-quality assessment, and clinical-validation scripts.

Install dependencies within the relevant component rather than at the repository root. The released data and benchmark are hosted separately on Hugging Face.

## Workflow

| Component | Purpose | Location |
| --- | --- | --- |
| PMC-MI construction | Build task-specific instructions from compound figures and article context | [`DataConstruction/`](DataConstruction/README.md) |
| Stage I: MI-SFT | Multi-image supervised fine-tuning | [`Training/SFT/`](Training/README.md#stage-i-and-stage-iii-supervised-fine-tuning) |
| Stage II: GRPO | Selection-aware policy refinement | [`Training/GRPO/`](Training/README.md#stage-ii-policy-refinement) |
| Stage III: Full-SFT | Full-mixture supervised fine-tuning | [`Training/SFT/`](Training/README.md#stage-i-and-stage-iii-supervised-fine-tuning) |
| Evaluation | Benchmark and validation workflows | [`Evaluation/`](Evaluation/README.md) |

## Repository structure

```text
M3LLM-v2/
├── DataConstruction/   # Five-step pipeline and task-specific prompts
├── Training/           # SFT configurations and GRPO reward/training code
└── Evaluation/         # PMC-MI-Bench, public benchmarks, judge, and validations
```

The Yale New Haven Health System clinical cohort is not distributed here. Its aggregate results and evaluation procedures are described in the manuscript. PMC-derived content, project-generated annotations, model checkpoints, and third-party software have distinct usage terms; consult the [dataset card](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0) and [third-party notes](THIRD_PARTY.md) before reuse.

## Citation

The accompanying manuscript is *From Compound Figures to Medical Multi-image Reasoning: Scaling Multimodal Large Language Models with Biomedical Literature*. Publication details will be added when available.
