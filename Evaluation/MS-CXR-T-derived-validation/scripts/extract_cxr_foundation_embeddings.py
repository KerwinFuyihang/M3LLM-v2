#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark.dataset import read_jsonl


def resolve_model_dirs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not (model_dir / "saved_model.pb").exists():
            raise FileNotFoundError(f"No saved_model.pb under {model_dir}")
        qformer_dir = Path(args.qformer_model_dir) if args.qformer_model_dir else None
        if args.embedding_mode == "contrastive":
            if qformer_dir is None or not (qformer_dir / "saved_model.pb").exists():
                raise FileNotFoundError(
                    "--qformer_model_dir with saved_model.pb is required in contrastive mode."
                )
        return model_dir, qformer_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to download google/cxr-foundation."
        ) from exc

    patterns = [f"{args.subfolder}/*"]
    if args.embedding_mode == "contrastive":
        patterns.append(f"{args.qformer_subfolder}/*")
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            allow_patterns=patterns,
            token=args.hf_token or os.environ.get("HF_TOKEN"),
        )
    )
    qformer_dir = (
        snapshot / args.qformer_subfolder
        if args.embedding_mode == "contrastive"
        else None
    )
    return snapshot / args.subfolder, qformer_dir


def collect_images(data_paths: list[str], image_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for data_path in data_paths:
        for record in read_jsonl(data_path):
            for image_name in record["image"]:
                image_path = Path(image_name)
                if not image_path.is_absolute():
                    image_path = image_root / image_path
                key = image_path.name
                if key in images and images[key] != image_path:
                    raise ValueError(
                        f"Duplicate image basename {key} maps to multiple paths."
                    )
                images[key] = image_path
    missing = [str(path) for path in images.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[:5]}")
    return images


def serialize_image(tf, image_path: Path) -> bytes:
    with Image.open(image_path) as image:
        image = image.convert("L")
        output = io.BytesIO()
        image.save(output, format="PNG")
    example = tf.train.Example()
    features = example.features.feature
    features["image/encoded"].bytes_list.value.append(output.getvalue())
    features["image/format"].bytes_list.value.append(b"png")
    return example.SerializeToString()


def pool_embedding(array: np.ndarray, pooling: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    if pooling == "flatten":
        return array.reshape(-1)
    if array.ndim == 1:
        return array
    axes = tuple(range(array.ndim - 1))
    if pooling == "mean":
        return array.mean(axis=axes)
    if pooling == "max":
        return array.max(axis=axes)
    raise ValueError(f"Unsupported pooling: {pooling}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen google/cxr-foundation ELIXR-C embeddings."
    )
    parser.add_argument("--data_path", action="append", required=True)
    parser.add_argument(
        "--image_root",
        required=True,
        help="MIMIC-CXR-JPG-sorted root.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--repo_id", default="google/cxr-foundation")
    parser.add_argument("--subfolder", default="elixr-c-v2-pooled")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--qformer_subfolder", default="pax-elixr-b-text")
    parser.add_argument("--qformer_model_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--signature", default="serving_default")
    parser.add_argument("--input_key", default="input_example")
    parser.add_argument("--output_key", default="feature_maps_0")
    parser.add_argument(
        "--embedding_mode",
        choices=["contrastive", "elixr_c"],
        default="contrastive",
        help="Official contrastive Q-Former embedding (default) or intermediate ELIXR-C map.",
    )
    parser.add_argument("--qformer_output_key", default="all_contrastive_img_emb")
    parser.add_argument("--pooling", choices=["mean", "max", "flatten"], default="mean")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for google/cxr-foundation. "
            "Install tensorflow-cpu (or tensorflow) in a separate extraction environment."
        ) from exc
    try:
        # Registers SentencepieceOp and other custom ops referenced by the SavedModel.
        import tensorflow_text

        _ = tensorflow_text
    except ImportError as exc:
        raise ImportError(
            "tensorflow-text is required to register SentencepieceOp. Install matching "
            "versions, for example: pip install 'tensorflow==2.20.0' "
            "'tensorflow-text==2.20.1' 'protobuf<6'."
        ) from exc

    model_dir, qformer_dir = resolve_model_dirs(args)
    print(f"Loading CXR Foundation SavedModel from {model_dir}")
    model = tf.saved_model.load(str(model_dir))
    if args.signature not in model.signatures:
        raise KeyError(
            f"Signature {args.signature!r} unavailable. Found {list(model.signatures)}. "
            "The Hugging Face export must include its official serving signature."
        )
    infer = model.signatures[args.signature]
    qformer_infer = None
    if qformer_dir is not None:
        print(f"Loading CXR Foundation Q-Former from {qformer_dir}")
        qformer_model = tf.saved_model.load(str(qformer_dir))
        if args.signature not in qformer_model.signatures:
            raise KeyError(
                f"Signature {args.signature!r} unavailable in Q-Former. "
                f"Found {list(qformer_model.signatures)}."
            )
        qformer_infer = qformer_model.signatures[args.signature]

    images = collect_images(args.data_path, Path(args.image_root))
    embeddings: dict[str, np.ndarray] = {}
    for key, image_path in tqdm(
        images.items(), desc="Extracting CXR embeddings", unit="image"
    ):
        serialized = serialize_image(tf, image_path)
        outputs = infer(**{args.input_key: tf.constant([serialized])})
        if args.output_key not in outputs:
            raise KeyError(
                f"Output {args.output_key!r} unavailable. Found {list(outputs.keys())}."
            )
        image_features = outputs[args.output_key]
        if qformer_infer is not None:
            qformer_outputs = qformer_infer(
                image_feature=image_features,
                ids=tf.zeros((1, 1, 128), dtype=tf.int32),
                paddings=tf.zeros((1, 1, 128), dtype=tf.float32),
            )
            if args.qformer_output_key not in qformer_outputs:
                raise KeyError(
                    f"Q-Former output {args.qformer_output_key!r} unavailable. "
                    f"Found {list(qformer_outputs.keys())}."
                )
            raw_embedding = qformer_outputs[args.qformer_output_key].numpy()
        else:
            raw_embedding = image_features.numpy()
        embeddings[key] = pool_embedding(raw_embedding, args.pooling)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **embeddings)
    metadata = {
        "repo_id": args.repo_id,
        "model_dir": str(model_dir),
        "qformer_model_dir": str(qformer_dir) if qformer_dir else None,
        "subfolder": args.subfolder,
        "qformer_subfolder": args.qformer_subfolder,
        "embedding_mode": args.embedding_mode,
        "signature": args.signature,
        "input_key": args.input_key,
        "output_key": args.output_key,
        "pooling": args.pooling,
        "num_images": len(embeddings),
        "embedding_dim": (
            int(next(iter(embeddings.values())).shape[0]) if embeddings else 0
        ),
        "data_paths": args.data_path,
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
