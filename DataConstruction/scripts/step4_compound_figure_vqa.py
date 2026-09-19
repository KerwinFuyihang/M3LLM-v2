#!/usr/bin/env python3
"""Step 4: construct VQA instructions for the complete compound figure."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import add_common_arguments, run_pipeline
from stages import TEXT_MODEL, transform_step4_compound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_model=TEXT_MODEL)
    args = parser.parse_args()
    run_pipeline(args, transform_step4_compound)


if __name__ == "__main__":
    main()
