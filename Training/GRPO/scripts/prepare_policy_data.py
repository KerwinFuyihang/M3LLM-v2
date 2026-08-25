#!/usr/bin/env python3
"""Convert policy-refinement instances to the EasyR1 JSONL schema.

Input record (json/jsonl):
{
  "id": "...",
  "image": ["compound.jpg", "sub1.jpg", ...],
  "conversations": [{"from": "human", "value": "..."}, ...],
  "gt_selection": ["Fig-1", "Fig-3"],
  "reference_answer": "..."
}

Output record (jsonl):
{
  "sample_id": "...",
  "prompt": "...",
  "images": ["/abs/path/compound.jpg", ...],
  "answer": ["Fig-1", "Fig-3"],
  "ground_truth_answer_text": "..."
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Stage II policy-refinement dataset for EasyR1."
    )
    parser.add_argument(
        "--input", required=True, help="Path to source JSON or JSONL file."
    )
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument(
        "--image-root",
        default="",
        help="Prefix for relative image paths. Leave empty if images are already absolute.",
    )
    parser.add_argument("--max-images", type=int, default=4)
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {i}: {exc}") from exc
        return rows

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of records.")
    return data


def extract_human_prompt(conversations: Any) -> str | None:
    if not isinstance(conversations, list):
        return None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        if message.get("from") == "human" and isinstance(message.get("value"), str):
            return message["value"]
    return None


def normalize_fig_set(figs: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen = set()
    for figure in figs:
        if not isinstance(figure, str):
            continue
        match = re.search(r"(\d+)", figure)
        if not match:
            continue
        key = f"Fig-{int(match.group(1))}"
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def extract_answer(record: dict[str, Any]) -> list[str]:
    answer = record.get("gt_selection")
    if isinstance(answer, list):
        answer_norm = normalize_fig_set(answer)
        if answer_norm:
            return answer_norm

    return []


def resolve_image_path(image_name: str, image_root: Path | None) -> str:
    path = Path(image_name)
    if path.is_absolute():
        return str(path)
    if image_root is None:
        return str(path)
    return str((image_root / path).resolve())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root).resolve() if args.image_root else None

    records = read_json_or_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                skipped += 1
                continue

            conversations = record.get("conversations")
            prompt = extract_human_prompt(conversations)
            images = record.get("image")
            if not isinstance(prompt, str) or not isinstance(images, list):
                skipped += 1
                continue

            image_paths = [
                resolve_image_path(x, image_root) for x in images if isinstance(x, str)
            ]
            if not image_paths or len(image_paths) > args.max_images:
                skipped += 1
                continue

            answer_norm = extract_answer(record)

            if not answer_norm:
                skipped += 1
                continue

            row = {
                "sample_id": str(record.get("id", idx)),
                "prompt": prompt,
                "images": image_paths,
                "answer": answer_norm,
            }
            reference_answer = record.get("reference_answer") or record.get("answer")
            if isinstance(reference_answer, str) and reference_answer.strip():
                row["ground_truth_answer_text"] = reference_answer.strip()
            else:
                skipped += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} malformed/empty records")


if __name__ == "__main__":
    main()
