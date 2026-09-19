#!/usr/bin/env python3
"""Step 4: construct four-option multiple-choice VQA instructions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import add_common_arguments, run_pipeline
from stages import TEXT_MODEL, load_classification_scores, transform_step4_mcq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_model=TEXT_MODEL)
    parser.add_argument("--target-mode", choices=("compound", "subimage"), default="subimage")
    parser.add_argument("--classification-csv", type=Path, default=None)
    parser.add_argument("--classification-threshold", type=float, default=0.70)
    parser.add_argument("--classification-filename-column", default="filename")
    parser.add_argument("--classification-score-column", default="prob")
    args = parser.parse_args()
    args._classification_scores = load_classification_scores(
        args.classification_csv,
        args.classification_filename_column,
        args.classification_score_column,
    )
    run_pipeline(args, transform_step4_mcq)


if __name__ == "__main__":
    main()
