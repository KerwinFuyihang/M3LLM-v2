from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm


BINARY_LABELS = ("Yes", "No")
PROGRESSION_LABELS = ("improving", "worsening", "stable")


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def normalize_for_exact(text: str) -> str:
    text = strip_thinking(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`\"']", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_label_match(response: str, labels: Sequence[str]) -> Optional[str]:
    normalized = normalize_for_exact(response)
    canonical = {label.lower(): label for label in labels}
    if normalized in canonical:
        return canonical[normalized]
    return None


def heuristic_label_match(response: str, labels: Sequence[str]) -> Optional[str]:
    normalized = normalize_for_exact(response)
    hits = []
    for label in labels:
        label_norm = label.lower()
        if re.search(rf"\b{re.escape(label_norm)}\b", normalized):
            hits.append(label)
    if len(hits) == 1:
        return hits[0]
    return None


def labels_for_task(task: str) -> Sequence[str]:
    if task in {"binary-disease-presence", "target-finding-identification"}:
        return BINARY_LABELS
    if task == "temporal-progression":
        return PROGRESSION_LABELS
    raise ValueError(f"Unsupported task: {task}")


def resolve_prediction(
    response: str,
    labels: Sequence[str],
) -> Dict:
    exact = exact_label_match(response, labels)
    if exact is not None:
        return {"prediction": exact, "parse_method": "exact"}

    heuristic = heuristic_label_match(response, labels)
    if heuristic is not None:
        return {"prediction": heuristic, "parse_method": "heuristic"}

    return {"prediction": None, "parse_method": "invalid"}


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_classification_metrics(
    evaluated: Iterable[Dict],
    labels: Sequence[str] | None = None,
    include_finding_breakdown: bool = True,
) -> Dict:
    """Overall (micro) accuracy plus per-class and macro precision/recall/F1.

    When records contain ``meta.disease``, also report per-finding metrics and
    finding-level macros (unweighted mean over findings).
    """
    records = list(evaluated)
    if labels is None:
        if records and "meta" in records[0] and "task" in records[0]["meta"]:
            labels = labels_for_task(str(records[0]["meta"]["task"]))
        else:
            labels = sorted({str(record["answer"]) for record in records})
    labels = list(labels)

    parse_methods: Counter = Counter()
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    # rows = true label, cols = predicted label
    confusion = [[0 for _ in labels] for _ in labels]
    invalid_by_label: Counter = Counter()
    invalid = 0
    correct = 0

    for record in records:
        parse_methods[record.get("parse_method", "unknown")] += 1
        answer = record["answer"]
        pred = record.get("prediction")
        if pred is None and "response" in record:
            pred = exact_label_match(
                record.get("response", ""), labels
            ) or heuristic_label_match(record.get("response", ""), labels)
        if answer not in label_to_idx:
            raise ValueError(f"Unknown reference label: {answer}")
        if pred not in label_to_idx:
            invalid += 1
            invalid_by_label[answer] += 1
            continue
        confusion[label_to_idx[answer]][label_to_idx[pred]] += 1
        if pred == answer:
            correct += 1

    valid = sum(sum(row) for row in confusion)
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(labels):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(len(labels)) if row != idx)
        fn = (
            sum(confusion[idx][col] for col in range(len(labels)) if col != idx)
            + invalid_by_label[label]
        )
        support = sum(confusion[idx]) + invalid_by_label[label]
        class_metrics = _prf1(tp, fp, fn)
        per_class[label] = {
            "support": support,
            **class_metrics,
            "accuracy": _safe_div(tp, support),
        }

    n_labels = max(len(labels), 1)
    macro_precision = sum(item["precision"] for item in per_class.values()) / n_labels
    macro_recall = sum(item["recall"] for item in per_class.values()) / n_labels
    macro_f1 = sum(item["f1"] for item in per_class.values()) / n_labels
    # Micro-averaged P/R/F1 for multiclass equal overall accuracy when every
    # sample has exactly one label; keep them for completeness.
    micro_tp = correct
    micro_fp = valid - correct
    micro_fn = len(records) - correct
    micro = _prf1(micro_tp, micro_fp, micro_fn)

    metrics: Dict = {
        "accuracy": _safe_div(correct, len(records)),
        "balanced_accuracy": macro_recall,  # macro over label classes
        "correct": correct,
        "total": len(records),
        "valid": valid,
        "invalid": invalid,
        "labels": labels,
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1": micro["f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion,
            "invalid_by_reference_label": dict(invalid_by_label),
        },
        "parse_methods": dict(parse_methods),
    }

    if include_finding_breakdown:
        finding_metrics = _compute_finding_macros(records, labels)
        if finding_metrics is not None:
            metrics.update(finding_metrics)

    return metrics


def _compute_finding_macros(
    records: Sequence[Dict],
    labels: Sequence[str],
) -> Dict | None:
    """Per-finding metrics and unweighted macros across findings/diseases."""
    by_finding: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        meta = record.get("meta") or {}
        disease = meta.get("disease") or meta.get("queried_finding")
        if disease is None or disease == "":
            continue
        by_finding[str(disease)].append(record)

    if not by_finding:
        return None

    per_finding: Dict[str, Dict] = {}
    for disease in sorted(by_finding):
        # Avoid infinite recursion: no nested finding breakdown.
        finding_stats = compute_classification_metrics(
            by_finding[disease],
            labels=labels,
            include_finding_breakdown=False,
        )
        per_finding[disease] = {
            "total": finding_stats["total"],
            "valid": finding_stats["valid"],
            "correct": finding_stats["correct"],
            "accuracy": finding_stats["accuracy"],
            # Macro over label classes within this finding
            # (binary: Yes/No; progression: improving/worsening/stable).
            "balanced_accuracy": finding_stats["balanced_accuracy"],
            "macro_precision": finding_stats["macro_precision"],
            "macro_recall": finding_stats["macro_recall"],
            "macro_f1": finding_stats["macro_f1"],
            "per_class": finding_stats["per_class"],
        }

    n_findings = max(len(per_finding), 1)
    finding_macro_accuracy = (
        sum(item["accuracy"] for item in per_finding.values()) / n_findings
    )
    finding_macro_balanced_accuracy = (
        sum(item["balanced_accuracy"] for item in per_finding.values()) / n_findings
    )
    finding_macro_precision = (
        sum(item["macro_precision"] for item in per_finding.values()) / n_findings
    )
    finding_macro_recall = (
        sum(item["macro_recall"] for item in per_finding.values()) / n_findings
    )
    finding_macro_f1 = (
        sum(item["macro_f1"] for item in per_finding.values()) / n_findings
    )

    return {
        "num_findings": len(per_finding),
        # Unweighted mean of per-finding overall accuracy.
        "finding_macro_accuracy": finding_macro_accuracy,
        # Unweighted mean of per-finding class-balanced accuracy
        # (macro over label classes within each finding, then macro over findings).
        "finding_macro_balanced_accuracy": finding_macro_balanced_accuracy,
        "finding_macro_precision": finding_macro_precision,
        "finding_macro_recall": finding_macro_recall,
        "finding_macro_f1": finding_macro_f1,
        "per_finding": per_finding,
    }


def evaluate_records(
    records: Iterable[Dict],
) -> tuple[Dict, List[Dict]]:
    records = list(records)
    evaluated: List[Dict] = []
    task_counts: Dict[str, Counter] = defaultdict(Counter)
    task_records: Dict[str, List[Dict]] = defaultdict(list)

    for record in tqdm(
        records,
        total=len(records),
        desc="Evaluating",
        unit="sample",
        dynamic_ncols=True,
    ):
        task = record["meta"]["task"]
        answer = record["answer"]
        response = record.get("response", "")
        labels = labels_for_task(task)
        parsed = resolve_prediction(
            response=response,
            labels=labels,
        )
        prediction = parsed["prediction"]
        correct = prediction == answer

        enriched = dict(record)
        enriched["prediction"] = prediction
        enriched["parse_method"] = parsed["parse_method"]
        enriched["correct"] = bool(correct)
        evaluated.append(enriched)

        task_counts[task][parsed["parse_method"]] += 1
        task_records[task].append(enriched)

    by_task = {}
    for task, task_eval in task_records.items():
        task_metrics = compute_classification_metrics(
            task_eval, labels=labels_for_task(task)
        )
        task_metrics["parse_methods"] = dict(task_counts[task])
        by_task[task] = task_metrics

    # Single-task runs keep top-level keys for backward compatibility.
    if len(by_task) == 1:
        only_task = next(iter(by_task.values()))
        metrics = dict(only_task)
        metrics["by_task"] = by_task
        return metrics, evaluated

    overall = compute_classification_metrics(evaluated)
    overall["by_task"] = by_task
    return overall, evaluated


def save_metrics(path: str | Path, metrics: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)
