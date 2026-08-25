#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derm_benchmark import read_jsonl, write_jsonl
from derm_benchmark.metrics import evaluate_records, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dermatology predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics, evaluated = evaluate_records(read_jsonl(args.predictions))
    write_jsonl(output_dir / "results_evaluated.jsonl", evaluated)
    save_metrics(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
