from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .biovil import LABEL_SPACES, LabelSpace
from .dataset import read_jsonl


def image_key(image_name: str) -> str:
    return Path(image_name).name


def load_embedding_cache(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}


def disease_vocabulary(records: Iterable[Dict]) -> list[str]:
    return sorted({str(record["meta"]["disease"]) for record in records})


class CXRFoundationTemporalDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        embedding_path: str | Path,
        label_space: LabelSpace,
        disease_to_idx: Dict[str, int],
        max_samples: int = 0,
    ) -> None:
        self.data_path = Path(data_path)
        self.records = read_jsonl(self.data_path)
        if max_samples > 0:
            self.records = self.records[:max_samples]
        self.embeddings = load_embedding_cache(embedding_path)
        self.label_to_idx = label_space.to_index()
        self.disease_to_idx = disease_to_idx

        missing = sorted(
            {
                image_key(image_name)
                for record in self.records
                for image_name in record["image"]
                if image_key(image_name) not in self.embeddings
            }
        )
        if missing:
            raise KeyError(
                f"{len(missing)} images are missing from {embedding_path}; "
                f"first missing keys: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        previous = torch.from_numpy(self.embeddings[image_key(record["image"][0])])
        current = torch.from_numpy(self.embeddings[image_key(record["image"][1])])
        disease = str(record["meta"]["disease"])
        if disease not in self.disease_to_idx:
            raise KeyError(
                f"Unknown disease {disease!r}; rebuild the disease vocabulary."
            )
        return {
            "id": record["id"],
            "previous_embedding": previous,
            "current_embedding": current,
            "disease_idx": torch.tensor(self.disease_to_idx[disease], dtype=torch.long),
            "label": torch.tensor(
                self.label_to_idx[record["answer"]], dtype=torch.long
            ),
            "answer": record["answer"],
        }

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.records:
            label = record["answer"]
            counts[label] = counts.get(label, 0) + 1
        return counts


class CXRFoundationTemporalClassifier(nn.Module):
    """Disease-conditioned temporal head over frozen CXR Foundation embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_diseases: int,
        num_classes: int,
        disease_embedding_dim: int = 32,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.disease_embedding = nn.Embedding(num_diseases, disease_embedding_dim)
        input_dim = embedding_dim * 4 + disease_embedding_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        previous_embedding: torch.Tensor,
        current_embedding: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        delta = current_embedding - previous_embedding
        disease = self.disease_embedding(disease_idx)
        features = torch.cat(
            [previous_embedding, current_embedding, delta, delta.abs(), disease], dim=-1
        )
        return self.classifier(features)


def infer_embedding_dim(embedding_path: str | Path) -> int:
    with np.load(embedding_path, allow_pickle=False) as data:
        if not data.files:
            raise ValueError(f"No embeddings found in {embedding_path}")
        array = np.asarray(data[data.files[0]])
    if array.ndim != 1:
        raise ValueError(f"Expected pooled 1-D embeddings, got shape {array.shape}")
    return int(array.shape[0])


def save_checkpoint(
    output_dir: str | Path,
    model: nn.Module,
    task: str,
    disease_to_idx: Dict[str, int],
    embedding_dim: int,
    model_config: Dict,
    args: Dict,
    epoch: int,
    best_metric: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_space = LABEL_SPACES[task]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task": task,
            "label_to_idx": label_space.to_index(),
            "idx_to_label": label_space.to_label(),
            "disease_to_idx": disease_to_idx,
            "embedding_dim": embedding_dim,
            "model_config": model_config,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task,
                "labels": list(label_space.labels),
                "disease_to_idx": disease_to_idx,
                "embedding_dim": embedding_dim,
                "model_config": model_config,
                "args": args,
                "best_metric": best_metric,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
