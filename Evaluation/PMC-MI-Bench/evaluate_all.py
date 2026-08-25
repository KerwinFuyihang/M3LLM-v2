#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import re
from typing import Any, Dict, List

from PIL import Image
from tqdm import tqdm

from eval_utils import (
    calculate_bertscore,
    calculate_bertsimilarity,
    calculate_bleu_scores,
    calculate_rouge,
    judge_multi_choice,
    parse_options_from_prompt,
)
from model_wrapper.vlm_manager import get_model
from task_config import TASKS

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sanitize_model_name(name: str) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", (name or "").strip()).strip("_")
    return slug or "unknown_model"


def model_path_slug(model_path: str) -> str:
    path = (model_path or "").strip().rstrip("/\\")
    if not path:
        return ""
    base = os.path.basename(path)
    if re.match(r"checkpoint-\d+$", base, re.I):
        parent = os.path.basename(os.path.dirname(path))
        slug = f"{parent}_{base}" if parent else base
    else:
        slug = base
    return sanitize_model_name(slug)


def get_model_results_dir(model_name: str, model_path: str) -> str:
    slug = model_path_slug(model_path) or sanitize_model_name(model_name)
    return os.path.join(RESULTS_ROOT, slug)


class Args:
    temperature = 0
    top_p = 1.0
    repetition_penalty = 1.0
    max_new_tokens = 2048
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")


def load_samples(path: str) -> List[dict]:
    is_jsonl = path.endswith(".jsonl") or path.endswith(".jsonl.gz")
    opener = gzip.open if path.endswith(".gz") else open
    samples: List[dict] = []
    with opener(path, "rt", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                samples.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                samples = data.get("data") or data.get("samples") or [data]
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
    return samples


def evaluate(model_name: str, model_path: str, task_name: str) -> None:
    model_results_dir = get_model_results_dir(model_name, model_path)
    model_id = os.path.basename(model_results_dir)
    os.makedirs(model_results_dir, exist_ok=True)

    if task_name not in TASKS:
        raise ValueError(f"Task '{task_name}' not found in task_config")

    task_cfg = TASKS[task_name]
    samples = load_samples(task_cfg["json_path"])
    model = get_model(model_name, model_path, Args())
    results: List[Dict[str, Any]] = []

    is_multi_choice = task_name.strip().lower() in {
        "multi-choice",
        "multiple-choice",
        "multichoice",
    }

    mc_total = 0
    mc_correct_total = 0
    metric_keys = [
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "rouge-1",
        "rouge-2",
        "rouge-l",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "sts",
    ]
    agg_sums = {k: 0.0 for k in metric_keys}
    agg_count = 0
    for idx, sample in enumerate(tqdm(samples, desc=f"{task_name}-{model_name}")):
        question = sample.get("question", "")
        reference_answer = sample.get("gt", "")
        image_index = sample.get("image_indices", "")
        sid = str(sample.get("id", idx))

        image_filenames = sample.get("images", []) or []
        if isinstance(image_filenames, str):
            image_filenames = [image_filenames]
        loaded_images = []
        for fn in image_filenames:
            full_path = os.path.join(task_cfg["image_root"], fn)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing image file: {full_path}")
            loaded_images.append(Image.open(full_path).convert("RGB"))

        messages = {
            "system": task_cfg.get("system_prompt", ""),
            "prompt": question,
            "images": loaded_images,
            "image_indices": image_index,
        }
        try:
            prediction = model.generate_output(messages)
        finally:
            for image in loaded_images:
                image.close()

        if is_multi_choice:
            choices = parse_options_from_prompt(question)
            if not choices:
                print(f"[Warn] No options parsed for id={sid}.")
                mc_correct = 0
            else:
                mc_correct = int(
                    judge_multi_choice(choices, reference_answer, prediction)
                )
            mc_total += 1
            mc_correct_total += mc_correct
            metrics = {"mc_correct": mc_correct}
        else:
            bleu = calculate_bleu_scores(reference_answer, prediction)
            rouge = calculate_rouge(reference_answer, prediction)
            bert = calculate_bertscore(prediction, reference_answer)
            sts = calculate_bertsimilarity(reference_answer, prediction)
            metrics = {
                **bleu,
                **rouge,
                **bert,
                "sts": float(sts),
            }
            for k in metric_keys:
                if k in metrics and isinstance(metrics[k], (int, float)):
                    agg_sums[k] += float(metrics[k])
            agg_count += 1

        results.append(
            {
                "id": sid,
                "task": task_name,
                "model": model_name,
                "model_id": model_id,
                "model_path": model_path,
                "question": question,
                "reference_answer": reference_answer,
                "prediction": prediction,
                "metrics": metrics,
            }
        )

    output_path = os.path.join(model_results_dir, f"{task_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_path}")

    if is_multi_choice:
        mc_acc = (mc_correct_total / mc_total) if mc_total else 0.0
        summary = {
            "task": task_name,
            "model": model_name,
            "model_id": model_id,
            "model_path": model_path,
            "num_samples": mc_total,
            "num_correct": mc_correct_total,
            "accuracy": mc_acc,
            "accuracy_pct": 100.0 * mc_acc,
        }
        print(
            f"[Multi-choice] Accuracy: {mc_correct_total}/{mc_total} = {100.0 * mc_acc:.2f}%"
        )
    else:
        averages = {
            k: (agg_sums[k] / agg_count if agg_count else 0.0)
            for k in metric_keys
            if k in agg_sums
        }
        summary = {
            "task": task_name,
            "model": model_name,
            "model_id": model_id,
            "model_path": model_path,
            "num_samples": agg_count,
            "macro_avg": averages,
        }

    summary_path = os.path.join(model_results_dir, f"{task_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PMC-MI-Bench evaluation for a model and task."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS.keys()))
    args = parser.parse_args()
    evaluate(args.model_name, args.model_path, args.task)
