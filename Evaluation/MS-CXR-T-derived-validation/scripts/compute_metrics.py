#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.dataset import read_jsonl
from ms_cxr_benchmark.metrics import (
    BINARY_LABELS,
    PROGRESSION_LABELS,
    compute_classification_metrics,
    labels_for_task,
    save_metrics,
)


def infer_labels(records: list[dict]) -> list[str]:
    if records and "meta" in records[0] and "task" in records[0]["meta"]:
        return list(labels_for_task(str(records[0]["meta"]["task"])))
    answers = {str(record["answer"]) for record in records}
    if answers <= set(BINARY_LABELS):
        return list(BINARY_LABELS)
    if answers <= set(PROGRESSION_LABELS):
        return list(PROGRESSION_LABELS)
    return sorted(answers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracy + macro/micro precision/recall/F1 from evaluated JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results_evaluated.jsonl (must have prediction/answer fields).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output metrics JSON path (default: overwrite <input_dir>/metrics.json).",
    )
    parser.add_argument(
        "--extended_only",
        action="store_true",
        help="Write metrics_extended.json instead of overwriting metrics.json.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_jsonl(input_path)
    metrics = compute_classification_metrics(records, labels=infer_labels(records))

    if args.output:
        output_path = Path(args.output)
    elif args.extended_only:
        output_path = input_path.parent / "metrics_extended.json"
    else:
        output_path = input_path.parent / "metrics.json"

    save_metrics(output_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
