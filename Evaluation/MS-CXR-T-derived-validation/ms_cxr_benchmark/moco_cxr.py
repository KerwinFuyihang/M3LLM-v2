from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .biovil import FINDINGS, LABEL_SPACES, LabelSpace, finding_slug
from .dataset import read_jsonl


CXR_MEAN = (0.5020, 0.5020, 0.5020)
CXR_STD = (0.085585, 0.085585, 0.085585)
FINDING_TO_INDEX = {finding: index for index, finding in enumerate(FINDINGS)}


class MoCoCXRPairTransform:
    """MoCo-CXR preprocessing with shared augmentation parameters for both images."""

    def __init__(
        self,
        training: bool,
        resize: int = 320,
        crop_size: int = 320,
        rotation_degrees: float = 10.0,
    ) -> None:
        self.training = training
        self.resize = resize
        self.crop_size = crop_size
        self.rotation_degrees = rotation_degrees

    def _relative_crop(
        self, image: Image.Image, vertical: float, horizontal: float
    ) -> Image.Image:
        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            image = TF.resize(
                image,
                [max(height, self.crop_size), max(width, self.crop_size)],
                interpolation=InterpolationMode.BILINEAR,
            )
            width, height = image.size
        top = round(vertical * max(height - self.crop_size, 0))
        left = round(horizontal * max(width - self.crop_size, 0))
        return TF.crop(image, top, left, self.crop_size, self.crop_size)

    def __call__(
        self, previous: Image.Image, current: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = TF.resize(
            previous, self.resize, interpolation=InterpolationMode.BILINEAR
        )
        current = TF.resize(
            current, self.resize, interpolation=InterpolationMode.BILINEAR
        )

        if self.training:
            if random.random() < 0.5:
                previous = TF.hflip(previous)
                current = TF.hflip(current)
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            previous = TF.rotate(
                previous, angle, interpolation=InterpolationMode.BILINEAR, fill=0
            )
            current = TF.rotate(
                current, angle, interpolation=InterpolationMode.BILINEAR, fill=0
            )
            vertical, horizontal = random.random(), random.random()
            previous = self._relative_crop(previous, vertical, horizontal)
            current = self._relative_crop(current, vertical, horizontal)
        else:
            previous = TF.center_crop(previous, [self.crop_size, self.crop_size])
            current = TF.center_crop(current, [self.crop_size, self.crop_size])

        previous_tensor = TF.normalize(TF.to_tensor(previous), CXR_MEAN, CXR_STD)
        current_tensor = TF.normalize(TF.to_tensor(current), CXR_MEAN, CXR_STD)
        return previous_tensor, current_tensor


class MoCoCXRTemporalDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        image_root: str | Path,
        label_space: LabelSpace,
        transform: MoCoCXRPairTransform,
        finding: str | None = None,
        records: Sequence[Dict] | None = None,
        max_samples: int = 0,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.label_to_idx = label_space.to_index()
        self.transform = transform
        normalized_finding = finding.strip().lower() if finding else None
        source_records = (
            list(records) if records is not None else read_jsonl(self.data_path)
        )
        self.records = [
            record
            for record in source_records
            if normalized_finding is None
            or str(record["meta"]["disease"]).strip().lower() == normalized_finding
        ]
        if max_samples > 0:
            self.records = self.records[:max_samples]
        if not self.records:
            raise ValueError(f"No records found for finding={finding!r} in {data_path}")
        unknown_findings = sorted(
            {
                str(record["meta"]["disease"]).strip().lower()
                for record in self.records
                if str(record["meta"]["disease"]).strip().lower()
                not in FINDING_TO_INDEX
            }
        )
        if unknown_findings:
            raise ValueError(f"Unsupported findings in dataset: {unknown_findings}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, image_name: str) -> Image.Image:
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        previous = self._load_rgb(record["image"][0])
        current = self._load_rgb(record["image"][1])
        previous, current = self.transform(previous, current)
        disease = str(record["meta"]["disease"]).strip().lower()
        return {
            "id": record["id"],
            "previous_image": previous,
            "current_image": current,
            "label": torch.tensor(
                self.label_to_idx[record["answer"]], dtype=torch.long
            ),
            "answer": record["answer"],
            "disease": disease,
            "disease_index": torch.tensor(FINDING_TO_INDEX[disease], dtype=torch.long),
        }

    def label_counts(self) -> Dict[str, int]:
        return dict(Counter(record["answer"] for record in self.records))


def load_moco_cxr_resnet18(
    checkpoint_path: str | Path,
) -> tuple[nn.Module, Dict[str, object]]:
    """Load the released MoCo query encoder and discard its projection head."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MoCo-CXR checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise KeyError(f"{checkpoint_path} does not contain a state_dict.")

    prefix = "module.encoder_q."
    converted = {
        key[len(prefix) :]: value
        for key, value in raw_state.items()
        if key.startswith(prefix) and not key.startswith(f"{prefix}fc.")
    }
    if not converted:
        raise KeyError(f"No {prefix}* backbone weights found in {checkpoint_path}.")

    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    incompatible = backbone.load_state_dict(converted, strict=False)
    allowed_missing = {"fc.weight", "fc.bias"}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "MoCo-CXR checkpoint conversion failed: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    metadata: Dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_arch": checkpoint.get("arch"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loaded_tensors": len(converted),
        "feature_dim": 512,
    }
    return backbone, metadata


class MoCoCXRLateFusionClassifier(nn.Module):
    """Shared single-image MoCo-CXR encoder with ordered two-image late fusion."""

    def __init__(
        self,
        num_classes: int,
        checkpoint_path: str | Path | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        num_findings: int = len(FINDINGS),
        disease_embedding_dim: int = 64,
        initialize_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if initialize_pretrained:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required when initialize_pretrained=True."
                )
            self.encoder, self.pretrained_metadata = load_moco_cxr_resnet18(
                checkpoint_path
            )
        else:
            self.encoder = resnet18(weights=None)
            self.encoder.fc = nn.Identity()
            self.pretrained_metadata = {}

        self.disease_embedding = nn.Embedding(num_findings, disease_embedding_dim)
        # Ordered concatenation retains previous/current identity without adding
        # an explicit temporal-difference inductive bias. Disease conditioning
        # lets one jointly trained model answer all five finding-specific queries.
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024 + disease_embedding_dim),
            nn.Linear(1024 + disease_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        previous_image: torch.Tensor,
        current_image: torch.Tensor,
        disease_index: torch.Tensor,
    ) -> torch.Tensor:
        previous_features = self.encoder(previous_image)
        current_features = self.encoder(current_image)
        disease_features = self.disease_embedding(disease_index)
        fused = torch.cat(
            [previous_features, current_features, disease_features], dim=-1
        )
        return self.classifier(fused)


def compute_class_weights(
    label_to_idx: Dict[str, int], counts: Dict[str, int]
) -> torch.Tensor:
    total = sum(counts.values())
    num_classes = len(label_to_idx)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for label, index in label_to_idx.items():
        weights[index] = total / max(counts.get(label, 1) * num_classes, 1)
    return weights


def balanced_accuracy(
    labels: Sequence[int], predictions: Sequence[int], num_classes: int
) -> float:
    recalls = []
    for class_idx in range(num_classes):
        support = sum(label == class_idx for label in labels)
        correct = sum(
            label == class_idx and prediction == class_idx
            for label, prediction in zip(labels, predictions, strict=False)
        )
        recalls.append(correct / support if support else 0.0)
    return sum(recalls) / max(len(recalls), 1)


def grouped_stratified_split(
    records: Sequence[Dict], val_ratio: float, seed: int
) -> tuple[list[Dict], list[Dict]]:
    """Approximate disease-label stratification while keeping subjects disjoint."""

    if not 0 < val_ratio < 1:
        return list(records), []
    groups: Dict[str, list[Dict]] = {}
    for record in records:
        subject = str(record["meta"]["subject_id"])
        groups.setdefault(subject, []).append(record)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    def stratum(record: Dict) -> tuple[str, str]:
        return (
            str(record["meta"]["disease"]).strip().lower(),
            str(record["answer"]),
        )

    total_counts = Counter(stratum(record) for record in records)
    target_counts = {
        label: 0 if count <= 1 else min(count - 1, max(1, round(count * val_ratio)))
        for label, count in total_counts.items()
    }
    target_size = max(1, round(len(records) * val_ratio))
    val_subjects: set[str] = set()
    val_counts: Counter = Counter()
    val_size = 0
    remaining = list(group_items)

    while remaining and val_size < target_size:
        best_index = 0
        best_score = -math.inf
        for index, (_, group_records) in enumerate(remaining):
            group_counts = Counter(stratum(record) for record in group_records)
            gain = sum(
                min(
                    group_counts[label],
                    max(target_counts[label] - val_counts[label], 0),
                )
                / max(target_counts[label], 1)
                for label in target_counts
            )
            overflow = max(val_size + len(group_records) - target_size, 0) / target_size
            score = gain - overflow
            if score > best_score:
                best_score = score
                best_index = index
        subject, selected_records = remaining.pop(best_index)
        val_subjects.add(subject)
        val_counts.update(stratum(record) for record in selected_records)
        val_size += len(selected_records)

    train_records = [
        record
        for record in records
        if str(record["meta"]["subject_id"]) not in val_subjects
    ]
    val_records = [
        record
        for record in records
        if str(record["meta"]["subject_id"]) in val_subjects
    ]
    if not train_records or not val_records:
        raise ValueError("Grouped train/validation split produced an empty partition.")
    return train_records, val_records


__all__ = [
    "CXR_MEAN",
    "CXR_STD",
    "FINDINGS",
    "FINDING_TO_INDEX",
    "LABEL_SPACES",
    "MoCoCXRPairTransform",
    "MoCoCXRTemporalDataset",
    "MoCoCXRLateFusionClassifier",
    "balanced_accuracy",
    "compute_class_weights",
    "finding_slug",
    "grouped_stratified_split",
    "load_moco_cxr_resnet18",
]
