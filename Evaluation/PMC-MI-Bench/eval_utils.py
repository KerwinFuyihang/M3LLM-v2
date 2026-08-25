import re

import evaluate
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer

_bertscore = None
_sbertmodel = None
rouge_scorer = Rouge()


def _get_sbertmodel():
    global _sbertmodel
    if _sbertmodel is None:
        _sbertmodel = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sbertmodel


def tokenize(text):
    return text.lower().replace(".", " .").split()


def calculate_bleu_scores(reference, prediction):
    reference_tokens = tokenize(reference)
    prediction_tokens = tokenize(prediction)
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n) + (0.0,) * (4 - n)
        scores[f"bleu{n}"] = sentence_bleu(
            [reference_tokens], prediction_tokens, weights=weights
        )
    return scores


def calculate_bertscore(
    pred, ref, lang="en", model_type="microsoft/deberta-xlarge-mnli"
):
    global _bertscore
    if _bertscore is None:
        _bertscore = evaluate.load("bertscore")

    preds = [pred] if isinstance(pred, str) else list(pred)
    refs = [ref] if isinstance(ref, str) else list(ref)
    out = _bertscore.compute(
        predictions=preds,
        references=refs,
        lang=lang,
        model_type=model_type,
    )

    p = float(sum(out["precision"]) / len(out["precision"]))
    r = float(sum(out["recall"]) / len(out["recall"]))
    f = float(sum(out["f1"]) / len(out["f1"]))
    return {"bertscore_p": p, "bertscore_r": r, "bertscore_f1": f}


def calculate_bertsimilarity(reference, prediction):
    model = _get_sbertmodel()
    embeddings_reference = model.encode(reference)
    embeddings_prediction = model.encode(prediction)
    similarities = model.similarity(embeddings_reference, embeddings_prediction)
    return float(similarities.squeeze().item())


def calculate_rouge(reference, prediction):
    if not reference.strip() or not prediction.strip():
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    scores = rouge_scorer.get_scores(prediction.lower(), reference.lower())[0]
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


def parse_options_from_prompt(prompt: str):
    text = prompt or ""
    lines = re.split(r"\n+", text)
    opts, buf, current_letter = [], "", None

    def flush():
        nonlocal buf, current_letter, opts
        if current_letter is not None:
            cleaned = re.sub(r"^[A-F]\s*[\.\):\-]\s*", "", buf.strip(), flags=re.I)
            if cleaned:
                opts.append(cleaned.strip().lower())
        buf, current_letter = "", None

    for line in lines:
        m = re.match(r"^\s*([A-F])\s*[\.\):\-]\s*(.*)", line, flags=re.I)
        if m:
            flush()
            current_letter = m.group(1).upper()
            buf = m.group(2)
        elif current_letter is not None:
            buf = (buf + " " + line.strip()).strip()
    flush()
    if opts:
        return opts

    parts = re.split(r"(?<![A-Za-z])([A-F])\s*[\.\):\-]\s*", text)
    merged = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            merged.append(parts[i + 1].strip().lower())
    return merged


def judge_multi_choice(choices, answer, response):
    allowed = {chr(ord("a") + i) for i in range(len(choices))}
    prediction = str(response).strip().lower()
    reference = str(answer).strip().lower()
    return prediction in allowed and prediction == reference
