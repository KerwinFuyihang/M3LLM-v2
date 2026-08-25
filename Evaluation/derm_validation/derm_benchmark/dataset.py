from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


class DermBinaryDataset:
    """JSONL dataset adapter for MedEvalKit wrappers."""

    def __init__(
        self,
        data_path: str | Path,
        image_root: str | Path,
        max_samples: int = 0,
        num_chunks: int = 1,
        chunk_idx: int = 0,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.max_samples = max_samples
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self.samples: List[Dict] = []

    def load_data(self) -> List[Dict]:
        records = read_jsonl(self.data_path)
        if self.max_samples > 0:
            records = records[: self.max_samples]
        self.samples = []
        for idx, record in enumerate(records):
            if idx % self.num_chunks != self.chunk_idx:
                continue
            self.samples.append(self.construct_messages(record))
        return self.samples

    def construct_messages(self, record: Dict) -> Dict:
        prompt = record["conversations"][0]["value"]
        image_names = record.get("image") or []
        if not image_names:
            raise ValueError(
                f"Sample {record.get('id', '<unknown>')} has an empty image list."
            )
        image_paths = []
        for image_name in image_names:
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = self.image_root / image_name
            image_paths.append(str(image_path))

        sample = dict(record)
        sample["messages"] = {"prompt": prompt, "image_paths": image_paths}
        return sample

    @staticmethod
    def load_images_from_paths(image_paths: List[str], max_image_side: int = 0):
        images = []
        for image_path in image_paths:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path).convert("RGB")
            if max_image_side > 0 and max(image.size) > max_image_side:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((max_image_side, max_image_side), resampling)
            images.append(image)
        return images
