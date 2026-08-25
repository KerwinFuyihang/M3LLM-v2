#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.biovil import LABEL_SPACES, task_from_data_path
from ms_cxr_benchmark.cxr_foundation import (
    CXRFoundationTemporalClassifier,
    CXRFoundationTemporalDataset,
    disease_vocabulary,
    infer_embedding_dim,
    save_checkpoint,
)
from ms_cxr_benchmark.dataset import read_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(label_to_idx: dict[str, int], counts: dict[str, int]) -> torch.Tensor:
    total = sum(counts.values())
    weights = torch.zeros(len(label_to_idx), dtype=torch.float32)
    for label, idx in label_to_idx.items():
        weights[idx] = total / max(counts.get(label, 1) * len(label_to_idx), 1)
    return weights


def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", unit="batch", dynamic_ncols=True):
            previous = batch["previous_embedding"].to(device)
            current = batch["current_embedding"].to(device)
            disease = batch["disease_idx"].to(device)
            labels = batch["label"].to(device)
            logits = model(previous, current, disease)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a disease-conditioned temporal head on CXR Foundation embeddings."
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--train_embeddings", required=True)
    parser.add_argument("--val_data_path", default=None)
    parser.add_argument("--val_embeddings", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=["binary", "progression"], default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--disease_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    task = args.task or task_from_data_path(args.train_data_path)
    label_space = LABEL_SPACES[task]
    train_records = read_jsonl(args.train_data_path)
    vocab_records = list(train_records)
    if args.val_data_path:
        vocab_records.extend(read_jsonl(args.val_data_path))
    disease_to_idx = {
        disease: idx for idx, disease in enumerate(disease_vocabulary(vocab_records))
    }
    embedding_dim = infer_embedding_dim(args.train_embeddings)

    train_dataset = CXRFoundationTemporalDataset(
        args.train_data_path,
        args.train_embeddings,
        label_space,
        disease_to_idx,
        max_samples=args.max_samples,
    )
    val_dataset = None
    dataset_for_weights = train_dataset
    if not args.val_data_path and args.val_ratio > 0:
        val_size = max(1, int(round(len(train_dataset) * args.val_ratio)))
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if args.val_data_path:
        if not args.val_embeddings:
            raise ValueError("--val_embeddings is required with --val_data_path.")
        val_dataset = CXRFoundationTemporalDataset(
            args.val_data_path,
            args.val_embeddings,
            label_space,
            disease_to_idx,
        )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "embedding_dim": embedding_dim,
        "num_diseases": len(disease_to_idx),
        "num_classes": len(label_space.labels),
        "disease_embedding_dim": args.disease_embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    model = CXRFoundationTemporalClassifier(**model_config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            label_space.to_index(), dataset_for_weights.label_counts()
        ).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_correct = total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in pbar:
            previous = batch["previous_embedding"].to(device)
            current = batch["current_embedding"].to(device)
            disease = batch["disease_idx"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(previous, current, disease)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(
                loss=f"{total_loss / max(total, 1):.4f}",
                acc=f"{total_correct / max(total, 1):.4f}",
            )
        scheduler.step()

        train_metrics = {
            "loss": total_loss / max(total, 1),
            "accuracy": total_correct / max(total, 1),
        }
        metrics = {"epoch": epoch, "train": train_metrics}
        if val_loader is not None:
            metrics["val"] = evaluate(model, val_loader, criterion, device)
            monitor = metrics["val"]["accuracy"]
        else:
            monitor = train_metrics["accuracy"]
        history.append(metrics)
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if monitor > best_metric:
            best_metric = monitor
            save_checkpoint(
                output_dir,
                model,
                task,
                disease_to_idx,
                embedding_dim,
                model_config,
                vars(args),
                epoch,
                best_metric,
            )
            print(f"Saved best checkpoint (accuracy={best_metric:.4f}).")


if __name__ == "__main__":
    main()
