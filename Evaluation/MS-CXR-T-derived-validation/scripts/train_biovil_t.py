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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.transforms import InterpolationMode, RandomAffine, RandomCrop
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.biovil import (
    BioViLTTemporalClassifier,
    LABEL_SPACES,
    MSCXRTemporalDataset,
    save_training_artifacts,
    task_from_data_path,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_eval_transform(crop_size: int = 448):
    try:
        from health_multimodal.image.data.transforms import (
            create_chest_xray_transform_for_inference,
        )
    except ImportError as exc:
        raise ImportError("Install hi-ml-multimodal to use BioViL-T training.") from exc
    return create_chest_xray_transform_for_inference(
        resize=512, center_crop_size=crop_size
    )


class SynchronizedPairTransform:
    """Paper-style augmentation with identical parameters for prior/current images."""

    def __init__(self, crop_size: int = 448, resize: int = 512) -> None:
        self.crop_size = crop_size
        self.resize = resize

    def __call__(self, previous, current):
        previous = TF.resize(
            previous, self.resize, interpolation=InterpolationMode.BILINEAR
        )
        current = TF.resize(
            current, self.resize, interpolation=InterpolationMode.BILINEAR
        )

        if random.random() < 0.5:
            previous = TF.hflip(previous)
            current = TF.hflip(current)

        angle, translations, scale, shear = RandomAffine.get_params(
            degrees=(-30.0, 30.0),
            translate=None,
            scale_ranges=None,
            shears=(-15.0, 15.0),
            img_size=list(previous.size),
        )
        previous = TF.affine(
            previous,
            angle,
            translations,
            scale,
            shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        current = TF.affine(
            current,
            angle,
            translations,
            scale,
            shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        i, j, h, w = RandomCrop.get_params(
            previous, output_size=(self.crop_size, self.crop_size)
        )
        previous = TF.crop(previous, i, j, h, w)
        current = TF.crop(current, i, j, h, w)

        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        previous = TF.adjust_brightness(previous, brightness)
        current = TF.adjust_brightness(current, brightness)
        previous = TF.adjust_contrast(previous, contrast)
        current = TF.adjust_contrast(current, contrast)

        previous = TF.to_tensor(previous).expand(3, -1, -1)
        current = TF.to_tensor(current).expand(3, -1, -1)
        noise = torch.randn_like(previous) * 0.01
        return (previous + noise).clamp_(0, 1), (current + noise).clamp_(0, 1)


def make_loaders(args) -> tuple[DataLoader, DataLoader | None, str, dict]:
    task = args.task or task_from_data_path(args.train_data_path)
    if task not in LABEL_SPACES:
        raise ValueError(
            f"Unsupported task {task}. Choose one of {list(LABEL_SPACES.keys())}."
        )

    eval_transform = create_eval_transform(crop_size=args.crop_size)
    dataset = MSCXRTemporalDataset(
        data_path=args.train_data_path,
        image_root=args.image_root,
        transform=None,
        label_space=LABEL_SPACES[task],
        max_samples=args.max_samples,
        finding=args.finding,
        pair_transform=SynchronizedPairTransform(crop_size=args.crop_size),
    )
    if not dataset.records:
        raise ValueError(f"No training samples found for finding={args.finding!r}.")

    val_loader = None
    if args.val_data_path:
        val_dataset = MSCXRTemporalDataset(
            data_path=args.val_data_path,
            image_root=args.image_root,
            transform=eval_transform,
            label_space=LABEL_SPACES[task],
            max_samples=0,
            finding=args.finding,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    elif args.val_ratio > 0:
        val_size = int(len(dataset) * args.val_ratio)
        train_size = len(dataset) - val_size
        indices = np.random.permutation(len(dataset))
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()
        dataset = Subset(dataset, train_indices)
        val_subset = Subset(
            MSCXRTemporalDataset(
                data_path=args.train_data_path,
                image_root=args.image_root,
                transform=eval_transform,
                label_space=LABEL_SPACES[task],
                max_samples=args.max_samples,
                finding=args.finding,
            ),
            val_indices,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, task, LABEL_SPACES[task].to_index()


def compute_class_weights(label_to_idx: dict, counts: dict) -> torch.Tensor:
    total = sum(counts.values())
    num_classes = len(label_to_idx)
    weights = torch.zeros(num_classes, dtype=torch.float)
    for label, idx in label_to_idx.items():
        count = counts.get(label, 1)
        weights[idx] = total / max(count * num_classes, 1)
    return weights


def balanced_accuracy(
    labels: list[int], predictions: list[int], num_classes: int
) -> float:
    recalls = []
    for class_idx in range(num_classes):
        positives = sum(label == class_idx for label in labels)
        correct = sum(
            label == class_idx and pred == class_idx
            for label, pred in zip(labels, predictions, strict=False)
        )
        recalls.append(correct / positives if positives else 0.0)
    return sum(recalls) / max(len(recalls), 1)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    pbar = tqdm(
        loader, desc="Train" if train else "Eval", unit="batch", dynamic_ncols=True
    )
    for batch in pbar:
        previous_image = batch["previous_image"].to(device, non_blocking=True)
        current_image = batch["current_image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(previous_image=previous_image, current_image=current_image)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(preds.detach().cpu().tolist())
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        pbar.set_postfix(
            loss=f"{(total_loss / max(total, 1)):.4f}",
            acc=f"{(total_correct / max(total, 1)):.4f}",
        )

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
        "balanced_acc": balanced_accuracy(
            all_labels, all_predictions, model.classifier[-1].out_features
        ),
    }


def optimizer_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Exclude positional encodings and missing-image embeddings as in the paper."""
    no_decay_terms = ("pos", "position", "missing", "p_missing")
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(term in name.lower() for term in no_decay_terms):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def linear_warmup_decay_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return remaining / decay_steps

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BioViL-T for MS-CXR-T tasks."
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--val_data_path", default=None)
    parser.add_argument(
        "--image_root", required=True, help="MIMIC-CXR-JPG-sorted root."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=["binary", "progression"], default=None)
    parser.add_argument(
        "--finding",
        required=True,
        help="Train one pathology-specific model, as in the paper.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr_scheduler_type", choices=["linear", "none"], default="linear"
    )
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--pretrained_encoder_checkpoint",
        default=None,
        help=(
            "Existing BioViL-T classifier checkpoint used only to initialize the "
            "released image encoder. Avoids re-downloading weights."
        ),
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epoch", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, task, label_to_idx = make_loaders(args)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    model = BioViLTTemporalClassifier(
        num_classes=len(label_to_idx),
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
        hidden_dim=args.classifier_hidden_dim,
        pretrained_encoder_checkpoint=args.pretrained_encoder_checkpoint,
    ).to(device)

    base_dataset = (
        train_loader.dataset.dataset
        if isinstance(train_loader.dataset, Subset)
        else train_loader.dataset
    )
    class_weights = compute_class_weights(label_to_idx, base_dataset.label_counts()).to(
        device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
    )
    train_steps_per_epoch = max(
        (len(train_loader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
        1,
    )
    total_train_steps = train_steps_per_epoch * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    if args.lr_scheduler_type == "linear":
        scheduler = linear_warmup_decay_scheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_train_steps
        )
    else:
        scheduler = None

    use_autocast = (
        args.precision == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    best_metric = -1.0
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train(True)
        train_total_loss = 0.0
        train_total_correct = 0
        train_total = 0
        train_labels: list[int] = []
        train_predictions: list[int] = []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc="Train", unit="batch", dynamic_ncols=True)
        for step_idx, batch in enumerate(pbar, start=1):
            previous_image = batch["previous_image"].to(device, non_blocking=True)
            current_image = batch["current_image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                logits = model(
                    previous_image=previous_image, current_image=current_image
                )
                loss = criterion(logits, labels)
                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            should_step = (
                step_idx % args.gradient_accumulation_steps == 0
                or step_idx == len(train_loader)
            )
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None:
                    scheduler.step()

            preds = torch.argmax(logits, dim=1)
            train_labels.extend(labels.detach().cpu().tolist())
            train_predictions.extend(preds.detach().cpu().tolist())
            train_total_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_total_loss += loss.item() * labels.size(0)
            lr_val = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{(train_total_loss / max(train_total, 1)):.4f}",
                acc=f"{(train_total_correct / max(train_total, 1)):.4f}",
                lr=f"{lr_val:.2e}",
            )
        train_metrics = {
            "loss": train_total_loss / max(train_total, 1),
            "acc": train_total_correct / max(train_total, 1),
            "balanced_acc": balanced_accuracy(
                train_labels, train_predictions, len(label_to_idx)
            ),
        }
        metrics = {"epoch": epoch, "train": train_metrics}

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, val_loader, criterion, optimizer, device, train=False
                )
            metrics["val"] = val_metrics
            monitor = val_metrics["balanced_acc"]
        else:
            monitor = train_metrics["balanced_acc"]

        history.append(metrics)
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=True)

        state_dict = {
            "model_state_dict": model.state_dict(),
            "task": task,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "epoch": epoch,
            "finding": args.finding,
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "freeze_encoder": args.freeze_encoder,
        }
        if args.save_every_epoch:
            torch.save(state_dict, output_dir / f"checkpoint_epoch_{epoch}.pt")
        if monitor > best_metric:
            best_metric = monitor
            save_training_artifacts(
                output_dir=output_dir,
                state_dict=state_dict,
                args_dict=vars(args),
                label_space=LABEL_SPACES[task],
                extra={
                    "best_metric": best_metric,
                    "monitor": "balanced_accuracy",
                    "effective_batch_size": (
                        args.batch_size * args.gradient_accumulation_steps
                    ),
                },
            )
            print(f"New best checkpoint saved (monitor={best_metric:.4f}).")

    print(f"Training complete. Best monitor metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
