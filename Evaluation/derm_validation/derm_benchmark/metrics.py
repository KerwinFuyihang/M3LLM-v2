from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LABELS = ("Yes", "No")


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`\"']", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_label(response: str) -> Optional[str]:
    normalized = normalize_text(response)
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
        return "No"
    yes_hit = re.search(r"\byes\b", normalized) is not None
    no_hit = re.search(r"\bno\b", normalized) is not None
    if yes_hit ^ no_hit:
        return "Yes" if yes_hit else "No"
    return None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_classification_metrics(
    evaluated: Iterable[Dict],
    positive_label: str = "Yes",
    negative_label: str = "No",
) -> Dict:
    """Binary metrics with positive_label treated as the positive class."""
    tp = fp = tn = fn = 0
    invalid_positive = 0
    invalid_negative = 0
    invalid = 0
    total = 0
    parse_methods: Counter = Counter()

    for record in evaluated:
        total += 1
        parse_methods[record.get("parse_method", "unknown")] += 1
        pred = record.get("prediction")
        if pred is None and "response" in record:
            pred = parse_label(record.get("response", ""))
        answer = record["answer"]
        if pred not in (positive_label, negative_label):
            invalid += 1
            if answer == positive_label:
                invalid_positive += 1
            elif answer == negative_label:
                invalid_negative += 1
            else:
                raise ValueError(f"Unknown reference label: {answer}")
            continue

        if pred == positive_label and answer == positive_label:
            tp += 1
        elif pred == positive_label and answer == negative_label:
            fp += 1
        elif pred == negative_label and answer == positive_label:
            fn += 1
        else:
            tn += 1

    valid = tp + fp + tn + fn
    yes_metrics = _prf1(tp, fp, fn + invalid_positive)
    no_metrics = _prf1(tn, fn, fp + invalid_negative)

    accuracy = _safe_div(tp + tn, total)
    balanced_accuracy = (yes_metrics["recall"] + no_metrics["recall"]) / 2.0
    macro_precision = (yes_metrics["precision"] + no_metrics["precision"]) / 2.0
    macro_recall = (yes_metrics["recall"] + no_metrics["recall"]) / 2.0
    macro_f1 = (yes_metrics["f1"] + no_metrics["f1"]) / 2.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "correct": tp + tn,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "invalid_positive": invalid_positive,
            "invalid_negative": invalid_negative,
        },
        "positive_class": positive_label,
        "precision": yes_metrics["precision"],
        "recall": yes_metrics["recall"],
        "f1": yes_metrics["f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            positive_label: {
                "support": tp + fn + invalid_positive,
                **yes_metrics,
                "accuracy": _safe_div(tp, tp + fn + invalid_positive),
            },
            negative_label: {
                "support": tn + fp + invalid_negative,
                **no_metrics,
                "accuracy": _safe_div(tn, tn + fp + invalid_negative),
            },
        },
        "parse_methods": dict(parse_methods),
    }


def evaluate_records(
    records: Iterable[Dict],
) -> tuple[Dict, List[Dict]]:
    evaluated: List[Dict] = []
    parse_counts = Counter()

    for record in records:
        answer = record["answer"]
        response = record.get("response", "")
        pred = record.get("prediction")
        parse_method = record.get("parse_method")

        if pred is None:
            pred = parse_label(response)
            parse_method = "heuristic" if pred is not None else "invalid"

        parse_counts[parse_method or "unknown"] += 1

        enriched = dict(record)
        enriched["prediction"] = pred
        enriched["parse_method"] = parse_method
        enriched["correct"] = pred == answer if pred is not None else False
        evaluated.append(enriched)

    metrics = compute_classification_metrics(evaluated)
    metrics["parse_methods"] = dict(parse_counts)
    return metrics, evaluated


def save_metrics(path: str | Path, metrics: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)
