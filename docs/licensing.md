# Licensing

M3LLM-v2 resources have separate licence scopes, following the approach used by [MedPMC](https://github.com/Yale-BIDS-Chen-Lab/MedPMC/blob/main/docs/licensing.md).

## Repository code

Original M3LLM-v2 source code, configurations, prompts, and documentation in this repository are licensed under the [Apache License 2.0](../LICENSE), except where a file or component carries different upstream terms. The repository includes adapted third-party code; the root licence does not replace its original notices or licences. See [third-party software](../THIRD_PARTY.md).

## PMC-MI data and PMC-MI-Bench

The [dataset release](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0) uses a mixed-licence declaration. To the extent the authors hold the relevant rights, project-generated questions, answer options, reference answers, structured reference trajectories, task labels, and dataset organisation are released under CC BY-NC-SA 4.0. PMC-derived images, captions, and article text remain subject to their source licences and any figure-specific rights notices. Each released record carries article-level licence and provenance metadata. The dataset's [data-licence statement](https://huggingface.co/datasets/KerwinFu/M3LLM-data-v1.0.0/blob/main/DATA_LICENCE.md) defines the reuse scope.

## Model weights

Model checkpoints are not covered by the code licence. Their release terms are specified in the respective model repository and may also depend on base-model terms.

## Clinical and third-party resources

The Yale New Haven Health System clinical data are not distributed in this repository. Other public or controlled-access datasets, upstream training frameworks, and adapted model wrappers retain their own terms and access conditions. See the manuscript and [third-party notes](../THIRD_PARTY.md) for the resources used in this study.
