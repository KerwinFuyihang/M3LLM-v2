#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.biovil import LABEL_SPACES, task_from_data_path
from ms_cxr_benchmark.dataset import read_jsonl
from ms_cxr_benchmark.moco_cxr import (
    FINDINGS,
    MoCoCXRPairTransform,
    MoCoCXRTemporalDataset,
    MoCoCXRLateFusionClassifier,
    balanced_accuracy,
    grouped_stratified_split,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model, loader, criterion, device, use_bf16: bool, num_classes: int
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    diseases_all: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", unit="batch", dynamic_ncols=True):
            previous = batch["previous_image"].to(device, non_blocking=True)
            current = batch["current_image"].to(device, non_blocking=True)
            disease_index = batch["disease_index"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(previous, current, disease_index)
                loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            diseases_all.extend(batch["disease"])
    correct = sum(
        label == prediction
        for label, prediction in zip(labels_all, predictions_all, strict=False)
    )
    by_finding: dict[str, tuple[list[int], list[int]]] = {}
    grouped_labels: dict[str, list[int]] = defaultdict(list)
    grouped_predictions: dict[str, list[int]] = defaultdict(list)
    for disease, label, prediction in zip(
        diseases_all, labels_all, predictions_all, strict=False
    ):
        grouped_labels[disease].append(label)
        grouped_predictions[disease].append(prediction)
    by_finding = {
        disease: (grouped_labels[disease], grouped_predictions[disease])
        for disease in grouped_labels
    }
    finding_scores = {
        disease: balanced_accuracy(labels, predictions, num_classes)
        for disease, (labels, predictions) in by_finding.items()
    }
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": balanced_accuracy(
            labels_all, predictions_all, num_classes=num_classes
        ),
        "finding_macro_balanced_accuracy": sum(finding_scores.values())
        / max(len(finding_scores), 1),
        "per_finding_balanced_accuracy": finding_scores,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint disease-conditioned MoCo-CXR two-image late-fusion training."
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=["binary", "progression"], default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--disease_embedding_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    task = args.task or task_from_data_path(args.train_data_path)
    label_space = LABEL_SPACES[task]
    label_to_idx = label_space.to_index()
    idx_to_label = label_space.to_label()

    all_records = read_jsonl(args.train_data_path)
    if args.max_samples > 0:
        all_records = all_records[: args.max_samples]
    if not all_records:
        raise ValueError(f"No training records found in {args.train_data_path}.")
    train_records, val_records = grouped_stratified_split(
        all_records, val_ratio=args.val_ratio, seed=args.seed
    )

    train_dataset = MoCoCXRTemporalDataset(
        args.train_data_path,
        args.image_root,
        label_space,
        transform=MoCoCXRPairTransform(training=True),
        records=train_records,
    )
    val_dataset = MoCoCXRTemporalDataset(
        args.train_data_path,
        args.image_root,
        label_space,
        transform=MoCoCXRPairTransform(training=False),
        records=val_records,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoCoCXRLateFusionClassifier(
        num_classes=len(label_space.labels),
        checkpoint_path=args.checkpoint_path,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        disease_embedding_dim=args.disease_embedding_dim,
        initialize_pretrained=True,
    ).to(device)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total_parameters:
        raise RuntimeError(
            f"Expected full fine-tuning, but only {trainable}/{total_parameters} parameters train."
        )

    stratum_counts = Counter(
        (
            str(record["meta"]["disease"]).strip().lower(),
            str(record["answer"]),
        )
        for record in train_records
    )
    num_strata = len(stratum_counts)
    stratum_weights = {
        stratum: len(train_records) / (num_strata * count)
        for stratum, count in stratum_counts.items()
    }
    train_criterion = nn.CrossEntropyLoss(reduction="none")
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(
        math.ceil(len(train_loader) / args.gradient_accumulation_steps), 1
    )
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    use_bf16 = (
        args.precision == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_metric = -1.0
    history: list[dict] = []
    global_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_metric = float(resume.get("best_metric", -1.0))
        global_step = int(resume.get("global_step", 0))
        history = list(resume.get("history", []))

    split_summary = {
        "training_scope": "joint_disease_conditioned",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_subjects": len({str(r["meta"]["subject_id"]) for r in train_records}),
        "val_subjects": len({str(r["meta"]["subject_id"]) for r in val_records}),
        "train_labels": dict(Counter(r["answer"] for r in train_records)),
        "val_labels": dict(Counter(r["answer"] for r in val_records)),
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total = 0
        train_labels: list[int] = []
        train_predictions: list[int] = []
        pbar = tqdm(
            train_loader,
            desc=f"{task} joint epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(pbar, start=1):
            previous = batch["previous_image"].to(device, non_blocking=True)
            current = batch["current_image"].to(device, non_blocking=True)
            disease_index = batch["disease_index"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(previous, current, disease_index)
                per_sample_loss = train_criterion(logits, labels)
                sample_weights = torch.tensor(
                    [
                        stratum_weights[(disease, answer)]
                        for disease, answer in zip(
                            batch["disease"], batch["answer"], strict=False
                        )
                    ],
                    dtype=per_sample_loss.dtype,
                    device=device,
                )
                loss = (per_sample_loss * sample_weights).mean()
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            should_step = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            train_labels.extend(labels.detach().cpu().tolist())
            train_predictions.extend(predictions.detach().cpu().tolist())
            train_acc = sum(
                label == prediction
                for label, prediction in zip(
                    train_labels, train_predictions, strict=False
                )
            ) / max(total, 1)
            pbar.set_postfix(
                loss=f"{total_loss / max(total, 1):.4f}",
                acc=f"{train_acc:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_correct = sum(
            label == prediction
            for label, prediction in zip(train_labels, train_predictions, strict=False)
        )
        train_metrics = {
            "loss": total_loss / max(total, 1),
            "accuracy": train_correct / max(total, 1),
            "balanced_accuracy": balanced_accuracy(
                train_labels, train_predictions, len(label_space.labels)
            ),
            "total": total,
        }
        val_metrics = evaluate(
            model,
            val_loader,
            eval_criterion,
            device,
            use_bf16,
            len(label_space.labels),
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=True)

        monitor = val_metrics["finding_macro_balanced_accuracy"]
        is_best = monitor > best_metric
        if is_best:
            best_metric = monitor
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "task": task,
            "training_scope": "joint_disease_conditioned",
            "loss_weighting": "inverse_disease_label_frequency",
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "model_config": {
                "num_classes": len(label_space.labels),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "num_findings": len(FINDINGS),
                "disease_embedding_dim": args.disease_embedding_dim,
                "fusion": "ordered_concat",
                "backbone": "resnet18",
            },
            "pretrained_metadata": model.pretrained_metadata,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "history": history,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "model.pt")
            with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": task,
                        "training_scope": "joint_disease_conditioned",
                        "loss_weighting": "inverse_disease_label_frequency",
                        "best_finding_macro_balanced_accuracy": best_metric,
                        "best_epoch": epoch,
                        "model_config": checkpoint["model_config"],
                        "pretrained_metadata": model.pretrained_metadata,
                        "split": split_summary,
                        "args": vars(args),
                    },
                    f,
                    indent=2,
                    ensure_ascii=True,
                )
            print(
                "Saved best joint checkpoint: "
                f"finding_macro_balanced_accuracy={best_metric:.4f}"
            )


if __name__ == "__main__":
    main()
