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

from ms_cxr_benchmark.biovil import (
    FINDINGS,
    BioViLTTemporalClassifier,
    LABEL_SPACES,
    MSCXRTemporalDataset,
    finding_slug,
    task_from_data_path,
)
from ms_cxr_benchmark.dataset import read_jsonl, write_jsonl
from ms_cxr_benchmark.metrics import evaluate_records, save_metrics


def create_transform(crop_size: int = 448):
    try:
        from health_multimodal.image.data.transforms import (
            create_chest_xray_transform_for_inference,
        )
    except ImportError as exc:
        raise ImportError(
            "Install hi-ml-multimodal to run BioViL-T evaluation."
        ) from exc
    return create_chest_xray_transform_for_inference(
        resize=512, center_crop_size=crop_size
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned BioViL-T on MS-CXR-T JSONL."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--image_root", required=True, help="MIMIC-CXR-JPG-sorted root."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing model.pt and metadata.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--per_finding",
        action="store_true",
        help="Load one pathology-specific checkpoint from <checkpoint_dir>/<finding_slug>.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    task = task_from_data_path(args.data_path)
    label_space = LABEL_SPACES[task]

    transform = create_transform(crop_size=args.crop_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_records = read_jsonl(args.data_path)
    records_by_id = {record["id"]: record for record in source_records}

    predictions = []
    findings = FINDINGS if args.per_finding else (None,)
    for finding in findings:
        model_dir = (
            checkpoint_dir / finding_slug(finding) if finding else checkpoint_dir
        )
        checkpoint_path = model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing BioViL-T checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_task = checkpoint.get("task", task)
        if checkpoint_task != task:
            raise ValueError(
                f"Checkpoint task {checkpoint_task!r} does not match evaluation task {task!r}."
            )
        idx_to_label = checkpoint.get("idx_to_label", label_space.to_label())
        idx_to_label = {int(k): v for k, v in idx_to_label.items()}

        dataset = MSCXRTemporalDataset(
            data_path=args.data_path,
            image_root=args.image_root,
            transform=transform,
            label_space=label_space,
            max_samples=args.max_samples,
            finding=finding,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        model = BioViLTTemporalClassifier(
            num_classes=len(label_space.labels),
            freeze_encoder=bool(checkpoint.get("freeze_encoder", False)),
            dropout=0.0,
            hidden_dim=int(checkpoint.get("classifier_hidden_dim", 128)),
            initialize_pretrained=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc=f"Evaluating BioViL-T ({finding or 'all'})",
                unit="batch",
                dynamic_ncols=True,
            ):
                previous_image = batch["previous_image"].to(device, non_blocking=True)
                current_image = batch["current_image"].to(device, non_blocking=True)
                logits = model(
                    previous_image=previous_image, current_image=current_image
                )
                pred_ids = torch.argmax(logits, dim=1).detach().cpu().tolist()
                for sample_id, pred_id, answer in zip(
                    batch["id"], pred_ids, batch["answer"], strict=False
                ):
                    predictions.append(
                        {
                            "id": sample_id,
                            "answer": answer,
                            "response": idx_to_label[int(pred_id)],
                            "meta": records_by_id[sample_id]["meta"],
                        }
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_order = {record["id"]: idx for idx, record in enumerate(source_records)}
    predictions.sort(key=lambda record: source_order[record["id"]])

    write_jsonl(output_dir / "results.jsonl", predictions)
    metrics, evaluated = evaluate_records(
        predictions,
    )
    write_jsonl(output_dir / "results_evaluated.jsonl", evaluated)
    save_metrics(output_dir / "metrics.json", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "num_samples": len(predictions),
                "checkpoint_dir": str(checkpoint_dir),
                "per_finding": args.per_finding,
            },
            f,
            indent=2,
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
