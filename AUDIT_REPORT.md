# Release audit

This audit covers the cleaned release copy. The original directory in
`Downloads` was not modified.

## Resolved issues

- Replaced stale demonstration training configurations with manuscript-aligned
  Stage I, Stage II and Stage III settings.
- Restricted Stage I and Stage III dataset registrations to the intended
  PMC-MI components; no unrelated instruction datasets are registered.
- Implemented the MS-CXR-T-derived split as a deterministic patient-level 1:1
  partition with seed 42. On the 1,326 source examples, the implementation was
  verified to produce 663/663 examples with no patient overlap.
- Removed the dermatology severity workflow. The release contains only the
  differential-diagnosis validity task described in the revised manuscript.
- Replaced stale benchmark prompts with the manuscript prompts and standardized
  the Fig-0/Fig-1... image identifiers used by the supported wrappers.
- Removed silent missing-image fallbacks from the primary PMC-MI-Bench wrappers.
  Missing or malformed required images now stop evaluation instead of producing
  an unintended text-only result.
- Made PMC-MI-Bench multiple-choice scoring an exact, case-insensitive option-
  letter match. Fuzzy answer-text matching is not used.
- Limited the reported automatic open-ended measures to the manuscript set:
  BLEU-1--4, ROUGE-1/2/L, BERTScore precision/recall/F1 and STS.
- Made invalid clinical-task responses count as incorrect while retaining them
  in the metric denominator.
- Removed local absolute paths, embedded credentials and project datasets from
  the cleaned tree. Restricted PMC, MIMIC-CXR and YNHH data are not bundled.
- Reduced first-party explanatory and debugging comments. Vendored upstream
  code remains attributable to its source rather than being stylistically
  rewritten.

## Parameters checked against the manuscript

- Stage I and III: full-parameter Qwen3-VL-8B training, one epoch, two H200
  GPUs, effective global batch 16, learning rate 2e-6, cosine decay, 3% warm-up,
  bfloat16, 2,048-token cutoff, at most four images and seed 42.
- Stage II: 10,361 optimization and 200 validation records, four H200 GPUs,
  15 epochs, 16 rollouts per prompt, global prompt batch 4, learning rate 1e-6,
  0.01 weight decay, seed 1 and reward operator 0.6/0.3/0.1.
- Clinical LoRA: rank 16, alpha 32, dropout 0.05, all eligible language-model
  linear layers, frozen vision tower/projector, three epochs, learning rate
  1e-4, effective batch 8, cosine decay, 5% warm-up and seed 42.
- PMC-MI-Bench inference: greedy decoding, temperature 0, top-p 1.0,
  repetition penalty 1.0 and 2,048 new tokens.
- Public benchmarks: seed 42, reasoning disabled, one run, temperature 0,
  top-p 0.0001, repetition penalty 1.0, 8,192 new tokens and eight-image cap.

## Validation performed

- Python byte-compilation for all release modules.
- Shell syntax validation for all launch scripts.
- Parsing of every JSON and YAML configuration.
- Unit checks for reward aggregation and invalid-response metric handling.
- A split-level check on the 1,326 longitudinal source records: 663 training
  examples, 663 test examples, zero overlapping patients and deterministic
  reproduction with seed 42.
- File scan for bundled project CSV/JSONL/TSV/Parquet/XLSX and clinical images.

## Items to verify before public archiving

1. Pin the exact immutable revision for every released model checkpoint used in
   the reported experiments, rather than relying only on a mutable repository
   name.
2. Restore the MedEvalKit upstream licence file and preserve all other upstream
   notices when publishing the vendored snapshot.
3. Test each architecture-specific LLaMA-Factory template used with the generic
   clinical LoRA launcher in its final software environment.
4. Record the checksum of the final private split manifests used to reproduce
   the clinical tables; do not commit the restricted manifests themselves.
5. The Stage II answer reward uses BERTScore F1 with `roberta-large`, matching
   the supplied training implementation. This model identifier should be stated
   explicitly in the Supplementary Methods.
