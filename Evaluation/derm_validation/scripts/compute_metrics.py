#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derm_benchmark import read_jsonl
from derm_benchmark.metrics import compute_classification_metrics, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracy, precision, recall, F1 from evaluated predictions JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results_evaluated.jsonl (must have prediction/answer fields).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output metrics JSON path (default: <input_dir>/metrics_extended.json).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_jsonl(input_path)
    metrics = compute_classification_metrics(records)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "metrics_extended.json"
    )
    save_metrics(output_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
