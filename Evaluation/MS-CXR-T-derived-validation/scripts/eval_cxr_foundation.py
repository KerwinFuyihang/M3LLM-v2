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

from ms_cxr_benchmark.biovil import LABEL_SPACES
from ms_cxr_benchmark.cxr_foundation import (
    CXRFoundationTemporalClassifier,
    CXRFoundationTemporalDataset,
)
from ms_cxr_benchmark.dataset import read_jsonl, write_jsonl
from ms_cxr_benchmark.metrics import evaluate_records, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a temporal classifier trained on CXR Foundation embeddings."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint = torch.load(
        checkpoint_dir / "model.pt", map_location="cpu", weights_only=False
    )
    task = checkpoint["task"]
    label_space = LABEL_SPACES[task]
    idx_to_label = {int(k): v for k, v in checkpoint["idx_to_label"].items()}
    disease_to_idx = checkpoint["disease_to_idx"]

    dataset = CXRFoundationTemporalDataset(
        args.data_path,
        args.embeddings,
        label_space,
        disease_to_idx,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CXRFoundationTemporalClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    records_by_id = {record["id"]: record for record in read_jsonl(args.data_path)}
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating CXR Foundation", unit="batch"):
            logits = model(
                batch["previous_embedding"].to(device),
                batch["current_embedding"].to(device),
                batch["disease_idx"].to(device),
            )
            pred_ids = logits.argmax(dim=1).cpu().tolist()
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "results.jsonl", predictions)
    metrics, evaluated = evaluate_records(predictions)
    write_jsonl(output_dir / "results_evaluated.jsonl", evaluated)
    save_metrics(output_dir / "metrics.json", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "num_samples": len(predictions),
                "checkpoint_dir": str(checkpoint_dir),
                "embeddings": args.embeddings,
            },
            f,
            indent=2,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
