#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
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


def select_images(image_names: List[str], max_image_num: int) -> List[str]:
    if not image_names:
        raise ValueError("Each case must contain at least one image.")
    if max_image_num > 0 and len(image_names) > max_image_num:
        indices = np.linspace(0, len(image_names) - 1, max_image_num, dtype=int)
        return [image_names[index] for index in indices]
    return image_names


def rebuild_prompt(prompt: str, num_images: int) -> str:
    if "\n\nQuestion:" not in prompt:
        raise ValueError("Prompt missing Question block.")
    _, question_part = prompt.split("\n\nQuestion:", 1)
    image_block = "\n".join([f"Image-{i}:<image>" for i in range(num_images)])
    header = (
        "You are a medical AI assistant interpreting clinical dermatology images.\n\n"
    )
    return f"{header}{image_block}\n\nQuestion:{question_part}"


def to_llamafactory_record(record: Dict, max_image_num: int) -> Dict:
    prompt = record["conversations"][0]["value"]
    answer = record["answer"]
    image_names = select_images(record["image"], max_image_num)
    prompt = rebuild_prompt(prompt, len(image_names))

    images = []
    for image_name in image_names:
        image_path = Path(image_name)
        images.append(str(image_path) if image_path.is_absolute() else image_name)

    return {
        "id": record["id"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": images,
    }


def convert_split(input_path: Path, output_path: Path, max_image_num: int) -> int:
    records = read_jsonl(input_path)
    converted = (
        to_llamafactory_record(record, max_image_num=max_image_num)
        for record in tqdm(records, desc=f"Converting {input_path.name}", unit="sample")
    )
    return write_jsonl(output_path, converted)


def write_dataset_info(output_dir: Path, dataset_key: str, file_name: str) -> None:
    path = output_dir / "dataset_info.json"
    info = {}
    if path.exists():
        info = json.loads(path.read_text(encoding="utf-8"))
    info[dataset_key] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=True)


TASK_SPECS = {
    "differential": {
        "input": "data/binary_differential/train.jsonl",
        "output": "derm_binary_train.jsonl",
        "dataset_key": "derm_binary_train",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dermatology clinical validation SFT data for LlamaFactory."
    )
    parser.add_argument("--benchmark_dir", default=str(ROOT))
    parser.add_argument("--output_dir", default=str(ROOT / "llamafactory_data"))
    parser.add_argument(
        "--task",
        choices=tuple(TASK_SPECS.keys()),
        default="differential",
        help="Benchmark task to convert for LlamaFactory SFT.",
    )
    parser.add_argument(
        "--max_image_num",
        type=int,
        default=8,
        help=(
            "Maximum number of images supplied to the model. Cases above the limit "
            "are sampled approximately uniformly across their ordered image sequence."
        ),
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TASK_SPECS[args.task]
    input_path = benchmark_dir / spec["input"]
    output_name = spec["output"]
    dataset_key = spec["dataset_key"]

    count = convert_split(
        input_path,
        output_dir / output_name,
        max_image_num=args.max_image_num,
    )
    write_dataset_info(output_dir, dataset_key=dataset_key, file_name=output_name)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "task": args.task,
                "max_image_num": args.max_image_num,
                dataset_key: count,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
