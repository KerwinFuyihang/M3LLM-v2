#!/usr/bin/env python3
"""Step 3: generate one grounded visual description per constituent subimage."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import add_common_arguments, run_pipeline
from stages import VISION_MODEL, load_image_manifest, transform_step3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        default_model=VISION_MODEL,
        default_backend="huatuo-local",
    )
    parser.set_defaults(max_new_tokens=512, repetition_penalty=1.2)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Directory joined with relative subimage filenames",
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=None,
        help="JSON mapping or CSV (image/path columns) for resolving images",
    )
    args = parser.parse_args()
    args._image_manifest = load_image_manifest(args.image_manifest)
    run_pipeline(args, transform_step3, images_required=True)


if __name__ == "__main__":
    main()
