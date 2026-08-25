from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm


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


class MS_CXR_T_Dataset:
    """JSONL dataset adapter for MedEvalKit-style VLM wrappers."""

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
        for idx, record in tqdm(
            enumerate(records),
            total=len(records),
            desc="Loading dataset",
            unit="sample",
            dynamic_ncols=True,
        ):
            if idx % self.num_chunks != self.chunk_idx:
                continue
            self.samples.append(self.construct_messages(record))
        return self.samples

    def construct_messages(self, record: Dict) -> Dict:
        prompt = record["conversations"][0]["value"]
        image_paths = []
        for image_name in record["image"]:
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = self.image_root / image_path
            image_paths.append(str(image_path))

        sample = dict(record)
        sample["messages"] = {"prompt": prompt, "image_paths": image_paths}
        return sample
