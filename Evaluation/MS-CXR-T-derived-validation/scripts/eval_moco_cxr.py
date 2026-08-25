#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.biovil import LABEL_SPACES, task_from_data_path
from ms_cxr_benchmark.dataset import read_jsonl, write_jsonl
from ms_cxr_benchmark.metrics import evaluate_records, save_metrics
from ms_cxr_benchmark.moco_cxr import (
    MoCoCXRPairTransform,
    MoCoCXRTemporalDataset,
    MoCoCXRLateFusionClassifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one jointly trained disease-conditioned MoCo-CXR model."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Joint MoCo-CXR model.pt checkpoint.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    task = task_from_data_path(args.data_path)
    label_space = LABEL_SPACES[task]
    source_records = read_jsonl(args.data_path)
    records_by_id = {record["id"]: record for record in source_records}
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing joint checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != task:
        raise ValueError(
            f"{checkpoint_path} has task={checkpoint.get('task')!r}, expected {task!r}."
        )
    if checkpoint.get("training_scope") != "joint_disease_conditioned":
        raise ValueError(
            f"{checkpoint_path} is not a joint disease-conditioned checkpoint."
        )
    config = checkpoint["model_config"]
    if config.get("fusion") != "ordered_concat":
        raise ValueError(f"Unsupported fusion: {config.get('fusion')}")
    idx_to_label = {
        int(index): label for index, label in checkpoint["idx_to_label"].items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (
        args.precision == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    model = MoCoCXRLateFusionClassifier(
        num_classes=int(config["num_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        num_findings=int(config["num_findings"]),
        disease_embedding_dim=int(config["disease_embedding_dim"]),
        initialize_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = MoCoCXRTemporalDataset(
        args.data_path,
        args.image_root,
        label_space,
        transform=MoCoCXRPairTransform(training=False),
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    predictions = []
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="MoCo-CXR joint evaluation",
            unit="batch",
            dynamic_ncols=True,
        ):
            previous = batch["previous_image"].to(device, non_blocking=True)
            current = batch["current_image"].to(device, non_blocking=True)
            disease_index = batch["disease_index"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(previous, current, disease_index)
            prediction_ids = logits.argmax(dim=1).cpu().tolist()
            for sample_id, prediction_id, answer in zip(
                batch["id"], prediction_ids, batch["answer"], strict=False
            ):
                predictions.append(
                    {
                        "id": sample_id,
                        "answer": answer,
                        "response": idx_to_label[int(prediction_id)],
                        "meta": records_by_id[sample_id]["meta"],
                    }
                )

    if len(predictions) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions, produced {len(predictions)}."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "results.jsonl", predictions)
    metrics, evaluated = evaluate_records(predictions)
    write_jsonl(output_dir / "results_evaluated.jsonl", evaluated)
    save_metrics(output_dir / "metrics.json", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "MoCo-CXR ResNet-18",
                "adaptation": (
                    "joint disease-conditioned shared-encoder ordered two-image late fusion"
                ),
                "task": task,
                "num_samples": len(predictions),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "best_validation_finding_macro_balanced_accuracy": checkpoint.get(
                    "best_metric"
                ),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
