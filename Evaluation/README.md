# Evaluation

Evaluation code is organised by benchmark or analysis. The [PMC-MI-Bench release](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0) provides benchmark records and required images; other public benchmarks should be obtained from their original sources. Install each component's dependencies separately.

| Component | Contents |
| --- | --- |
| [`PMC-MI-Bench/`](PMC-MI-Bench/) | Task configurations, model wrappers, inference, and reference-based metrics |
| [`MedEvalKit/`](MedEvalKit/) | Adapted public-benchmark evaluation, including OmniMedVQA and MMMU-Med |
| [`LLM-Judge/`](LLM-Judge/README.md) | Response-quality rubric, prompts, and an open-weight judge route |
| [`MS-CXR-T-derived-validation/`](MS-CXR-T-derived-validation/) | Longitudinal chest-radiograph evaluation and target-task adaptation |
| [`derm_validation/`](derm_validation/) | Dermatology evaluation scripts and target-task adaptation |
| [`common/`](common/) | Shared clinical adaptation configuration |

The manuscript's OmniMedVQA evaluation excludes four source datasets with potential publication- or preprint-derived content: Covid CT, CoronaHack, Covid19 heywhale, and COVIDx CXR-4. Apply this selection to the source JSON files before running the MedEvalKit wrapper, which otherwise loads all files in `QA_information/Open-access`. The Yale New Haven Health System dermatology images and case data are not part of this public repository. For cases containing more than eight images, the dermatology evaluation scripts sample up to eight images across the ordered image sequence.

The MedEvalKit snapshot and adapted model wrappers retain their upstream attribution and terms; see [`THIRD_PARTY.md`](../THIRD_PARTY.md). The open-weight judge route is described in [`LLM-Judge/README.md`](LLM-Judge/README.md).
