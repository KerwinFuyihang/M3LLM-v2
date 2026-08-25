#!/usr/bin/env python
"""Construct the two MS-CXR-T-derived tasks used in the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ms_cxr_benchmark import write_jsonl


FINDINGS = (
    ("consolidation", "consolidation"),
    ("edema", "edema"),
    ("pleural_effusion", "pleural effusion"),
    ("pneumonia", "pneumonia"),
    ("pneumothorax", "pneumothorax"),
)

PROGRESSION_PROMPT = (
    "You are a medical AI assistant interpreting paired longitudinal chest radiographs.\n"
    "Previous examination Fig-0: <image>\n"
    "Current examination Fig-1: <image>\n"
    "Question: Relative to the prior examination, is the patient's {finding} "
    "improving, worsening, or stable?\n"
    "Respond with exactly one label: improving, stable, or worsening."
)

TARGET_FINDING_PROMPT = (
    "You are a medical AI assistant interpreting paired longitudinal chest radiographs.\n"
    "Previous examination Fig-0: <image>\n"
    "Current examination Fig-1: <image>\n"
    "Question: Is {finding} the annotated target finding for this image pair?\n"
    "Answer with exactly one label: Yes or No."
)


def _image_name(dicom_id: str) -> str:
    return Path(dicom_id).name + ".jpg"


def _validate_images(
    row: Dict[str, str], image_root: Path
) -> tuple[list[str], list[int], list[int]]:
    names = [_image_name(row["previous_dicom_id"]), _image_name(row["dicom_id"])]
    missing = [name for name in names if not (image_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing}")
    sizes = []
    for name in names:
        with Image.open(image_root / name) as image:
            sizes.append(image.size)
    return names, [size[0] for size in sizes], [size[1] for size in sizes]


def _base_examples(
    rows: Iterable[tuple[int, Dict[str, str]]], image_root: Path, skip_missing: bool
) -> tuple[list[Dict], list[Dict]]:
    examples: list[Dict] = []
    skipped: list[Dict] = []
    for row_index, row in rows:
        try:
            images, widths, heights = _validate_images(row, image_root)
        except FileNotFoundError as error:
            if not skip_missing:
                raise
            skipped.append({"source_row": row_index, "reason": str(error)})
            continue

        for key, name in FINDINGS:
            progression = row[f"{key}_progression"].strip().lower()
            if not progression:
                continue
            examples.append(
                {
                    "id": f"row{row_index}::{row['study_id']}::{name}",
                    "subject_id": str(row["subject_id"]),
                    "study_id": row["study_id"],
                    "previous_study_id": row["previous_study_id"],
                    "source_row": row_index,
                    "finding": name,
                    "progression": progression,
                    "label_quality": row.get(f"{key}_label_quality", "").strip()
                    or None,
                    "images": images,
                    "widths": widths,
                    "heights": heights,
                }
            )
    return examples, skipped


def _patient_split(examples: list[Dict], seed: int) -> tuple[set[str], set[str]]:
    """Create an exact 1:1 example split without separating a patient."""
    groups: dict[str, int] = Counter(example["subject_id"] for example in examples)
    total = len(examples)
    if total % 2:
        raise ValueError(
            f"An exact 1:1 split requires an even number of examples; found {total}."
        )
    target = total // 2
    patients = list(groups)
    random.Random(seed).shuffle(patients)

    reachable: dict[int, tuple[int, str] | None] = {0: None}
    for patient in patients:
        size = groups[patient]
        for subtotal in sorted(tuple(reachable), reverse=True):
            candidate = subtotal + size
            if candidate <= target and candidate not in reachable:
                reachable[candidate] = (subtotal, patient)
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError("No patient-level partition produces equal example counts.")

    train_patients: set[str] = set()
    subtotal = target
    while subtotal:
        previous, patient = reachable[subtotal]  # type: ignore[misc]
        train_patients.add(patient)
        subtotal = previous
    test_patients = set(groups) - train_patients
    return train_patients, test_patients


def _progression_record(example: Dict) -> Dict:
    return {
        "id": f"{example['id']}::progression",
        "image": example["images"],
        "width_list": example["widths"],
        "height_list": example["heights"],
        "meta": {
            "task": "temporal-progression",
            "disease": example["finding"],
            "subject_id": example["subject_id"],
            "study_id": example["study_id"],
            "previous_study_id": example["previous_study_id"],
            "source_row": example["source_row"],
            "label_quality": example["label_quality"],
        },
        "conversations": [
            {
                "from": "human",
                "value": PROGRESSION_PROMPT.format(finding=example["finding"]),
            }
        ],
        "answer": example["progression"],
    }


def _target_finding_records(example: Dict) -> list[Dict]:
    records = []
    for _, queried_finding in FINDINGS:
        records.append(
            {
                "id": f"{example['id']}::target::{queried_finding}",
                "image": example["images"],
                "width_list": example["widths"],
                "height_list": example["heights"],
                "meta": {
                    "task": "target-finding-identification",
                    "queried_finding": queried_finding,
                    "annotated_target": example["finding"],
                    "subject_id": example["subject_id"],
                    "study_id": example["study_id"],
                    "previous_study_id": example["previous_study_id"],
                    "source_row": example["source_row"],
                },
                "conversations": [
                    {
                        "from": "human",
                        "value": TARGET_FINDING_PROMPT.format(finding=queried_finding),
                    }
                ],
                "answer": "Yes" if queried_finding == example["finding"] else "No",
            }
        )
    return records


def _write_task(
    output_dir: Path, name: str, train: list[Dict], test: list[Dict]
) -> Dict:
    task_dir = output_dir / name
    write_jsonl(task_dir / "train.jsonl", train)
    write_jsonl(task_dir / "test.jsonl", test)
    write_jsonl(task_dir / "all.jsonl", train + test)
    return {"train": len(train), "test": len(test), "all": len(train) + len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_missing", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(enumerate(csv.DictReader(handle), start=1))
    examples, skipped = _base_examples(rows, image_root, args.skip_missing)
    train_patients, test_patients = _patient_split(examples, seed=args.seed)

    train_examples = [item for item in examples if item["subject_id"] in train_patients]
    test_examples = [item for item in examples if item["subject_id"] in test_patients]
    progression_train = [_progression_record(item) for item in train_examples]
    progression_test = [_progression_record(item) for item in test_examples]
    target_train = [
        record for item in train_examples for record in _target_finding_records(item)
    ]
    target_test = [
        record for item in test_examples for record in _target_finding_records(item)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "split": "patient-level 1:1",
        "source_rows": len(rows),
        "temporal_image_label_examples": len(examples),
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "skipped": skipped,
        "progression_classification": _write_task(
            output_dir,
            "progression_classification",
            progression_train,
            progression_test,
        ),
        "target_finding_identification": _write_task(
            output_dir, "target_finding_identification", target_train, target_test
        ),
        "progression_class_distribution": {
            "train": dict(Counter(item["answer"] for item in progression_train)),
            "test": dict(Counter(item["answer"] for item in progression_test)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
