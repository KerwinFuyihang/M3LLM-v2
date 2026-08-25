#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_cxr_benchmark import MS_CXR_T_Dataset, write_jsonl
from ms_cxr_benchmark.metrics import evaluate_records, save_metrics


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def init_medevalkit_model(args: argparse.Namespace):
    medevalkit_path = Path(args.medevalkit_path).resolve()
    sys.path.insert(0, str(medevalkit_path))

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["REASONING"] = str(args.reasoning)
    os.environ["use_vllm"] = str(args.use_vllm)
    os.environ["max_image_num"] = str(args.max_image_num)
    os.environ["tensor_parallel_size"] = str(args.tensor_parallel_size)
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if str_to_bool(args.use_vllm) and int(args.tensor_parallel_size) > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from LLMs import init_llm

    model_args = SimpleNamespace(
        model_name=args.model_name,
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        max_image_num=args.max_image_num,
    )
    return init_llm(model_args)


def load_image(image_path: str, max_image_side: int):
    image = Image.open(image_path).convert("RGB")
    if max_image_side > 0 and max(image.size) > max_image_side:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_image_side, max_image_side), resampling)
    return image


def run_batches(samples, model, batch_size: int, max_image_side: int):
    out_records = []
    total = len(samples)
    pbar = tqdm(
        total=total,
        desc="Inference",
        unit="sample",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for start in range(0, total, batch_size):
        batch = samples[start : start + batch_size]
        opened_images = []
        for sample in batch:
            image_paths = sample["messages"].pop("image_paths")
            images = [
                load_image(image_path, max_image_side) for image_path in image_paths
            ]
            sample["messages"]["images"] = images
            opened_images.extend(images)
        outputs = model.generate_outputs([sample["messages"] for sample in batch])
        for sample, response in zip(batch, outputs):
            record = {key: value for key, value in sample.items() if key != "messages"}
            record["response"] = response
            out_records.append(record)
        for image in opened_images:
            image.close()
        gc.collect()
        pbar.update(len(batch))
        pbar.set_postfix(
            batch=f"{start // batch_size + 1}/{(total + batch_size - 1) // batch_size}",
            done=len(out_records),
        )
    pbar.close()
    return out_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MS-CXR-T benchmark with MedEvalKit model wrappers."
    )
    parser.add_argument(
        "--data_path", required=True, help="Path to train/test/all JSONL."
    )
    parser.add_argument(
        "--image_root",
        default=os.environ.get("MSCXR_IMAGE_ROOT"),
        help="MIMIC-CXR-JPG-sorted root (or set MSCXR_IMAGE_ROOT).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--medevalkit_path",
        default=os.environ.get("MEDEVALKIT_PATH", "../MedEvalKit"),
    )
    parser.add_argument("--model_name", default="TestModel")
    parser.add_argument("--model_path", default="unused")
    parser.add_argument("--use_vllm", default="True")
    parser.add_argument("--reasoning", default="False")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--tensor_parallel_size", default="1")
    parser.add_argument("--max_image_num", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.0001)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_image_side",
        type=int,
        default=0,
        help="Resize each image so its longest side is at most this value. Use 0 for original resolution.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    if not args.image_root:
        parser.error("--image_root is required (or set MSCXR_IMAGE_ROOT).")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MS_CXR_T_Dataset(
        data_path=args.data_path,
        image_root=args.image_root,
        max_samples=args.max_samples,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
    )
    print(f"Loading samples from {args.data_path} ...")
    samples = dataset.load_data()
    print(f"Loaded {len(samples)} samples.")

    print(f"Loading model {args.model_name} from {args.model_path} ...")
    model = init_medevalkit_model(args)
    print("Model ready. Starting inference.")

    out_records = run_batches(
        samples,
        model,
        batch_size=args.batch_size,
        max_image_side=args.max_image_side,
    )
    write_jsonl(output_dir / "results.jsonl", out_records)

    print("Evaluating predictions ...")
    metrics, evaluated = evaluate_records(
        out_records,
    )
    write_jsonl(output_dir / "results_evaluated.jsonl", evaluated)
    save_metrics(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
