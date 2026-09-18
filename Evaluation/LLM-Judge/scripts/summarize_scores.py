#!/usr/bin/env python3
"""Summarize mean judge scores by candidate-model metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DIMENSIONS = ("completeness", "clarity", "correctness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {dimension: [] for dimension in DIMENSIONS}
    )
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            group = str(row.get("candidate_model") or "all")
            for dimension in DIMENSIONS:
                grouped[group][dimension].append(row["scores"][dimension]["score"])

    summary = {}
    for group, values in sorted(grouped.items()):
        summary[group] = {
            "n": len(values[DIMENSIONS[0]]),
            **{
                dimension: sum(scores) / len(scores) if scores else None
                for dimension, scores in values.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
