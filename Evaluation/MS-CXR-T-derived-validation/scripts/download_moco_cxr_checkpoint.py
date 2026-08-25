#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.moco_cxr import load_moco_cxr_resnet18


DEFAULT_URL = "https://storage.googleapis.com/moco-cxr/r8w-00001-v2.pth.tar"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate the released MoCo-CXR ResNet-18 checkpoint."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        default="cache/moco_cxr/r8w-00001-v2.pth.tar",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        _, metadata = load_moco_cxr_resnet18(output)
        print(f"Checkpoint already exists and is valid: {output}")
        print(metadata)
        return

    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading {args.url} -> {output}")
    try:
        with urllib.request.urlopen(args.url) as response, temporary.open(
            "wb"
        ) as target:
            shutil.copyfileobj(response, target, length=16 * 1024 * 1024)
        temporary.replace(output)
        _, metadata = load_moco_cxr_resnet18(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    print(f"Validated checkpoint: {output}")
    print(metadata)


if __name__ == "__main__":
    main()
