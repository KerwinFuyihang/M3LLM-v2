from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from .dataset import read_jsonl


@dataclass(frozen=True)
class LabelSpace:
    labels: Sequence[str]

    def to_index(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    def to_label(self) -> Dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}


LABEL_SPACES = {
    "binary": LabelSpace(labels=("No", "Yes")),
    "progression": LabelSpace(labels=("improving", "worsening", "stable")),
}

FINDINGS = (
    "consolidation",
    "edema",
    "pleural effusion",
    "pneumonia",
    "pneumothorax",
)


def finding_slug(finding: str) -> str:
    return finding.strip().lower().replace(" ", "_")


def task_from_data_path(data_path: str | Path) -> str:
    data_path = str(data_path).lower()
    if "binary" in data_path:
        return "binary"
    if "progression" in data_path:
        return "progression"
    raise ValueError(
        "Cannot infer task from data path. Use a path containing either 'binary' or 'progression'."
    )


class MSCXRTemporalDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        image_root: str | Path,
        transform: Callable | None,
        label_space: LabelSpace,
        max_samples: int = 0,
        finding: str | None = None,
        pair_transform: Callable | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.pair_transform = pair_transform
        self.label_to_idx = label_space.to_index()
        self.records = read_jsonl(self.data_path)
        if finding is not None:
            finding_normalized = finding.strip().lower()
            self.records = [
                record
                for record in self.records
                if str(record["meta"]["disease"]).strip().lower() == finding_normalized
            ]
        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        previous_image = self._load_gray_image(record["image"][0])
        current_image = self._load_gray_image(record["image"][1])
        if self.pair_transform is not None:
            previous_image, current_image = self.pair_transform(
                previous_image, current_image
            )
        elif self.transform is not None:
            previous_image = self.transform(previous_image)
            current_image = self.transform(current_image)
        else:
            raise ValueError("Either transform or pair_transform must be provided.")
        label = self.label_to_idx[record["answer"]]
        sample = {
            "id": record["id"],
            "previous_image": previous_image,
            "current_image": current_image,
            "label": torch.tensor(label, dtype=torch.long),
            "answer": record["answer"],
            "disease": str(record["meta"]["disease"]),
        }
        return sample

    def _load_gray_image(self, image_name: str) -> Image.Image:
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = self.image_root / image_name
        with Image.open(image_path) as image:
            return image.convert("L")

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            label = record["answer"]
            counts[label] = counts.get(label, 0) + 1
        return counts


class BioViLTTemporalClassifier(nn.Module):
    """BioViL-T multi-image encoder with the paper's multilayer task head."""

    def __init__(
        self,
        num_classes: int,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        hidden_dim: int = 128,
        pretrained_encoder_checkpoint: str | Path | None = None,
        initialize_pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            from health_multimodal.image.model.pretrained import (
                get_biovil_t_image_encoder,
            )
            from health_multimodal.image.model.model import ImageModel
            from health_multimodal.image.model.types import ImageEncoderType
        except ImportError as exc:
            raise ImportError(
                "health_multimodal is required. Install with `pip install hi-ml-multimodal`."
            ) from exc

        if initialize_pretrained and pretrained_encoder_checkpoint is None:
            self.image_model = get_biovil_t_image_encoder(freeze_encoder=freeze_encoder)
        else:
            self.image_model = ImageModel(
                img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                joint_feature_size=128,
                freeze_encoder=freeze_encoder,
                pretrained_model_path=None,
            )
            if pretrained_encoder_checkpoint is not None:
                checkpoint = torch.load(
                    pretrained_encoder_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                image_state = {
                    key.removeprefix("image_model."): value
                    for key, value in state_dict.items()
                    if key.startswith("image_model.")
                }
                if not image_state:
                    raise KeyError(
                        f"No image_model weights found in {pretrained_encoder_checkpoint}"
                    )
                self.image_model.load_state_dict(image_state, strict=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, previous_image: torch.Tensor, current_image: torch.Tensor
    ) -> torch.Tensor:
        # get_biovil_t_image_encoder() returns ImageModel with a MultiImageEncoder trunk.
        # ImageModel.forward does not accept previous_image/current_image kwargs, so we call
        # the multi-image trunk directly and then reuse forward_post_encoder.
        with torch.set_grad_enabled(not self.image_model.freeze_encoder):
            patch_x, pooled_x = self.image_model.encoder(
                current_image=current_image,
                previous_image=previous_image,
                return_patch_embeddings=True,
            )
        outputs = self.image_model.forward_post_encoder(patch_x, pooled_x)
        features = outputs.projected_global_embedding
        return self.classifier(features)


def save_training_artifacts(
    output_dir: str | Path,
    state_dict: Dict,
    args_dict: Dict,
    label_space: LabelSpace,
    extra: Dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_dir / "model.pt")

    metadata = {
        "args": args_dict,
        "labels": list(label_space.labels),
        "extra": extra or {},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)
