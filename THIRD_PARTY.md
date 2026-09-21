# Third-party software

The repository's [Apache-2.0 licence](LICENSE) applies to original M3LLM-v2 code and documentation; it does not replace the terms of the components listed below or other upstream code incorporated into model wrappers. See [licensing details](docs/licensing.md).

This release contains model wrappers adapted from their upstream projects and a
MedEvalKit snapshot used for public-benchmark evaluation. Those files remain
subject to their upstream licences and attribution requirements.

- MedEvalKit: <https://github.com/alibaba-damo-academy/MedEvalKit>, evaluated
  at commit `9b12e3b` (Apache-2.0 as identified by the upstream project).
- LLaMA-Factory: <https://github.com/hiyouga/LLaMA-Factory>, version
  `0.9.5.dev0` (`v0.9.4-48-g184304b5`); installed separately.
- EasyR1: <https://github.com/hiyouga/EasyR1>, revision
  `v0.3.2-41-gdd71bbd`; installed separately.

The upstream training frameworks are not vendored here. Model checkpoints and
datasets retain their own licences and must be obtained from their official
release locations.
