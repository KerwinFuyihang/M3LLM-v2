#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def to_llamafactory_record(record: Dict) -> Dict:
    prompt = record["conversations"][0]["value"]
    answer = record["answer"]
    images = []
    for image_name in record["image"]:
        image_path = Path(image_name)
        if image_path.is_absolute():
            images.append(str(image_path))
        else:
            images.append(image_name)

    return {
        "id": record["id"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": images,
    }


def convert_split(input_path: Path, output_path: Path) -> int:
    records = read_jsonl(input_path)
    converted = (
        to_llamafactory_record(record)
        for record in tqdm(records, desc=f"Converting {input_path.name}", unit="sample")
    )
    return write_jsonl(output_path, converted)


def write_dataset_info(output_dir: Path) -> None:
    dataset_info = {
        "mscxr_target_finding_train": {
            "file_name": "mscxr_target_finding_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
        "mscxr_progression_train": {
            "file_name": "mscxr_progression_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        },
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MS-CXR-T SFT data for LlamaFactory."
    )
    parser.add_argument("--benchmark_dir", default=str(ROOT))
    parser.add_argument("--output_dir", default=str(ROOT / "llamafactory_data"))
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "mscxr_target_finding_train": convert_split(
            benchmark_dir / "data/target_finding_identification/train.jsonl",
            output_dir / "mscxr_target_finding_train.jsonl",
        ),
        "mscxr_progression_train": convert_split(
            benchmark_dir / "data/progression_classification/train.jsonl",
            output_dir / "mscxr_progression_train.jsonl",
        ),
    }
    write_dataset_info(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
