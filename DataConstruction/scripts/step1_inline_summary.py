#!/usr/bin/env python3
"""Step 1: summarize article passages associated with each compound figure."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import add_common_arguments, run_pipeline
from stages import TEXT_MODEL, transform_step1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_model=TEXT_MODEL)
    parser.set_defaults(max_new_tokens=512)
    args = parser.parse_args()
    run_pipeline(args, transform_step1)


if __name__ == "__main__":
    main()
