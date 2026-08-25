"""Reward function used for Stage II policy refinement."""

from __future__ import annotations

import re
from typing import Any, Iterable


REWARD_NAME = "m3llm_policy_refinement"
REWARD_TYPE = "batch"

ANSWER_WEIGHT = 0.6
SELECTION_WEIGHT = 0.3
FORMAT_WEIGHT = 0.1
DEFAULT_BERTSCORE_MODEL = "roberta-large"

SELECTION_BLOCK_RE = re.compile(
    r"<selection>\s*(.*?)\s*</selection>", re.IGNORECASE | re.DOTALL
)
ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FIG_TOKEN_RE = re.compile(r"\bFig-\d+\b", re.IGNORECASE)

_BERT_SCORER = None


def _normalize_figures(figures: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for figure in figures:
        if not isinstance(figure, str):
            continue
        match = re.search(r"(\d+)", figure)
        if match is None:
            continue
        identifier = f"Fig-{int(match.group(1))}"
        if identifier not in seen:
            normalized.append(identifier)
            seen.add(identifier)
    return normalized


def _extract_selection(response: str) -> list[str] | None:
    match = SELECTION_BLOCK_RE.search(response)
    if match is None:
        return None
    return _normalize_figures(FIG_TOKEN_RE.findall(match.group(1)))


def _extract_answer(response: str) -> str:
    match = ANSWER_BLOCK_RE.search(response)
    return match.group(1).strip() if match else response.strip()


def _format_reward(response: str) -> float:
    """Score the presence and order of Thought, Selection and Answer blocks."""
    lower = response.lower()

    def block(tag: str) -> tuple[int, int] | None:
        start = lower.find(f"<{tag}>")
        end = lower.find(f"</{tag}>")
        return (start, end) if 0 <= start < end else None

    thought = block("thought")
    selection = block("selection")
    answer = block("answer")
    valid = sum(item is not None for item in (thought, selection, answer))
    if valid == 0:
        return 0.0
    if valid == 1:
        return 0.3
    if valid == 2:
        return 0.6

    assert thought is not None and selection is not None and answer is not None
    ordered = (
        thought[0] < thought[1] < selection[0] < selection[1] < answer[0] < answer[1]
    )
    return 1.0 if ordered else 0.6


def _selection_f1(predicted: list[str] | None, reference: list[str]) -> float:
    if predicted is None:
        return 0.0
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not predicted_set and not reference_set:
        return 1.0
    if not predicted_set or not reference_set:
        return 0.0
    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(reference_set)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def _answer_reward(predicted: str, reference: str, model_type: str) -> float:
    if not predicted or not reference:
        return 0.0

    from bert_score import BERTScorer

    global _BERT_SCORER
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(lang="en", model_type=model_type)
    _, _, f1 = _BERT_SCORER.score([predicted], [reference])
    return max(0.0, min(1.0, float(f1.item())))


def compute_score(
    reward_inputs: list[dict[str, Any]],
    answer_weight: float = ANSWER_WEIGHT,
    selection_weight: float = SELECTION_WEIGHT,
    format_weight: float = FORMAT_WEIGHT,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
) -> list[dict[str, float]]:
    """Return the weighted answer, selection and format rewards.

    The default operator is 0.6 answer + 0.3 selection + 0.1 format.
    Component-ablation experiments set the corresponding weight to zero.
    """
    scores: list[dict[str, float]] = []
    for item in reward_inputs:
        response = str(item.get("response", ""))
        raw_reference_selection = item.get("ground_truth", [])
        reference_selection = _normalize_figures(
            raw_reference_selection if isinstance(raw_reference_selection, list) else []
        )
        reference_answer = str(item.get("ground_truth_answer_text", "")).strip()

        selection = _selection_f1(_extract_selection(response), reference_selection)
        output_format = _format_reward(response)
        answer = _answer_reward(
            _extract_answer(response), reference_answer, model_type=bertscore_model
        )
        overall = (
            answer_weight * answer
            + selection_weight * selection
            + format_weight * output_format
        )
        scores.append(
            {
                "overall": float(overall),
                "answer": float(answer),
                "selection": float(selection),
                "format": float(output_format),
            }
        )
    return scores
