#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derm_benchmark import DermBinaryDataset, write_jsonl
from derm_benchmark.metrics import evaluate_records, save_metrics


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


_IMAGE_LINE_RE = re.compile(r"^Image-\d+:\s*(?:<image>)?\s*$")


def prompt_for_loaded_images(prompt: str) -> str:
    lines = []
    for line in prompt.splitlines():
        if _IMAGE_LINE_RE.match(line.strip()):
            continue
        lines.append(line.replace("<image>", ""))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_done_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id"):
                done.add(record["id"])
    return done


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


def make_record(
    sample: dict,
    response: str,
    response_source: str = "model",
    num_images_used: int | None = None,
) -> dict:
    record = {key: value for key, value in sample.items() if key != "messages"}
    record["response"] = response
    record["response_source"] = response_source
    if num_images_used is not None:
        record["num_images_used"] = num_images_used
        record["num_images_total"] = len(sample.get("image", []))
    if sample.get("images_used"):
        record["images_used"] = sample["images_used"]
    return record


def run_single_sample(
    sample: dict,
    model,
    max_image_side: int,
    max_image_num: int,
) -> dict:
    full_image_paths = list(sample["messages"]["image_paths"])
    if len(full_image_paths) > max_image_num:
        raise ValueError(
            f"Sample {sample.get('id', 'unknown')} exceeds the {max_image_num}-image eligibility limit."
        )
    image_paths = full_image_paths
    if not image_paths:
        raise ValueError(f"Sample {sample.get('id', 'unknown')} has no images to send.")
    original_prompt = sample["messages"]["prompt"]
    aligned_prompt = prompt_for_loaded_images(original_prompt)
    sample["messages"]["prompt"] = aligned_prompt
    sample["images_used"] = [Path(path).name for path in image_paths]
    opened_images = []
    try:
        images = DermBinaryDataset.load_images_from_paths(
            image_paths, max_image_side=max_image_side
        )
        if len(images) != len(image_paths):
            raise RuntimeError(
                f"Loaded {len(images)} images but expected {len(image_paths)} for {sample.get('id')}"
            )
        opened_images.extend(images)
        sample["messages"]["images"] = images
        outputs = model.generate_outputs([sample["messages"]])
        return make_record(
            sample,
            outputs[0],
            response_source="model",
            num_images_used=len(image_paths),
        )
    finally:
        sample["messages"].pop("images", None)
        sample["messages"]["image_paths"] = full_image_paths
        sample["messages"]["prompt"] = original_prompt
        sample.pop("images_used", None)
        for image in opened_images:
            image.close()
        gc.collect()


def run_batches(
    samples,
    model,
    batch_size: int,
    max_image_side: int,
    max_image_num: int,
    results_path: Path | None = None,
):
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
        if batch_size == 1:
            record = run_single_sample(
                batch[0],
                model,
                max_image_side=max_image_side,
                max_image_num=max_image_num,
            )
            batch_records = [record]
        else:
            batch_records = []
            for sample in batch:
                batch_records.append(
                    run_single_sample(
                        sample,
                        model,
                        max_image_side=max_image_side,
                        max_image_num=max_image_num,
                    )
                )
        out_records.extend(batch_records)
        if results_path is not None:
            with results_path.open("a", encoding="utf-8") as f:
                for record in batch_records:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
        pbar.update(len(batch))
        pbar.set_postfix(
            batch=f"{start // batch_size + 1}/{(total + batch_size - 1) // batch_size}",
            done=len(out_records),
        )
    pbar.close()
    return out_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dermatology benchmark with MedEvalKit model wrappers."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--image_root",
        default=os.environ.get("DERM_IMAGE_ROOT"),
        help="Resized dermatology image root (or set DERM_IMAGE_ROOT).",
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
    parser.add_argument("--max_image_num", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.0001)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_image_side", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample ids already present in output_dir/results.jsonl and append new rows.",
    )
    args = parser.parse_args()
    if not args.image_root:
        parser.error("--image_root is required (or set DERM_IMAGE_ROOT).")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DermBinaryDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        max_samples=args.max_samples,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
    )
    print(f"Loading samples from {args.data_path} ...")
    samples = dataset.load_data()
    results_path = output_dir / "results.jsonl"
    done_ids: set[str] = set()
    if args.resume:
        done_ids = load_done_ids(results_path)
        if done_ids:
            before = len(samples)
            samples = [s for s in samples if s.get("id") not in done_ids]
            print(
                f"Resume: skipping {before - len(samples)} done ids, {len(samples)} remaining."
            )
    print(f"Loaded {len(samples)} samples to run (max_image_num={args.max_image_num}).")
    if samples:
        n_ineligible = sum(
            1
            for s in samples
            if args.max_image_num > 0
            and len(s["messages"]["image_paths"]) > args.max_image_num
        )
        print(
            f"Images: IMAGE_ROOT={args.image_root}; "
            f"{n_ineligible}/{len(samples)} samples exceed the eligibility limit."
        )
        first = samples[0]
        print(
            f"Example {first.get('id')}: "
            f"{len(first['messages']['image_paths'])} files, "
            f"prompt after image-align has no <image> tokens "
            f"(wrappers attach PIL via messages['images'])."
        )
    if not samples:
        print("Nothing to run.")
    else:
        print(f"Loading model {args.model_name} from {args.model_path} ...")
        model = init_medevalkit_model(args)
        print("Model ready. Starting inference.")
        if not args.resume and results_path.exists():
            results_path.unlink()
        run_batches(
            samples,
            model,
            batch_size=args.batch_size,
            max_image_side=args.max_image_side,
            max_image_num=args.max_image_num,
            results_path=results_path,
        )

    out_records = (
        [
            json.loads(line)
            for line in results_path.open(encoding="utf-8")
            if line.strip()
        ]
        if results_path.exists()
        else []
    )
    write_jsonl(results_path, out_records)

    print("Evaluating predictions ...")
    metrics, evaluated = evaluate_records(
        out_records,
    )
    write_jsonl(output_dir / "results_evaluated.jsonl", evaluated)
    save_metrics(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
