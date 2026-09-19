"""Record transformations for the five PMC-MI construction steps."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    Generator,
    SkipRecord,
    as_text,
    extract_sections,
    first_value,
    read_prompt,
    strip_label,
)


TEXT_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
VISION_MODEL = "FreedomIntelligence/HuatuoGPT-Vision-34B"


def caption(record: Mapping[str, Any]) -> str:
    return as_text(first_value(record, "caption", "compound_caption", "figure_caption"))


def inline_summary(record: Mapping[str, Any]) -> str:
    return as_text(first_value(record, "Inline Summary", "inline_summary", "references"))


def medical_knowledge(record: Mapping[str, Any]) -> str:
    return as_text(first_value(record, "medical knowledge", "Medical Knowledge", "medical_knowledge"))


def subimages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = first_value(record, "subcaptions", "subimages", "sub_images", default=[])
    if not isinstance(values, list):
        raise SkipRecord("subimage collection is missing or is not a list")
    return [dict(item) for item in values if isinstance(item, Mapping)]


def fig_id(subimage: Mapping[str, Any], order: int) -> str:
    explicit = first_value(subimage, "fig_id", "figure_id", "image_index")
    if explicit:
        value = str(explicit)
        return value if value.lower().startswith("fig-") else f"Fig-{value}"
    # The source scripts stored zero-based idx values; the manuscript convention
    # is Fig-1 through Fig-N in stored record order.
    return f"Fig-{order + 1}"


def subimage_caption(subimage: Mapping[str, Any]) -> str:
    return as_text(first_value(subimage, "text", "caption", "subimage_caption"))


def visual_description(subimage: Mapping[str, Any]) -> str:
    return as_text(
        first_value(
            subimage,
            "visual perception description",
            "visual_description",
            "Visual description",
        )
    )


def image_name(subimage: Mapping[str, Any]) -> str:
    return as_text(first_value(subimage, "image", "image_path", "filename"))


def load_image_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("image manifest JSON must map image identifiers to paths")
        return {str(key): str(item) for key, item in value.items()}
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("image") or row.get("filename") or row.get("id")
            value = row.get("path") or row.get("image_path")
            if key and value:
                result[str(key)] = str(value)
    return result


def resolve_image_path(name: str, args: argparse.Namespace) -> str:
    image_path = Path(name)
    if image_path.is_file():
        return str(image_path)
    manifest = getattr(args, "_image_manifest", {})
    if name in manifest and Path(manifest[name]).is_file():
        return str(Path(manifest[name]))
    image_root = getattr(args, "image_root", None)
    if image_root:
        candidate = image_root / name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"could not resolve subimage '{name}'; use --image-root or --image-manifest"
    )


def transform_step1(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    passages = as_text(first_value(record, "references", "inline_text", "article_passages"))
    if not passages:
        raise SkipRecord("no figure-associated inline text")
    raw = generator.generate(
        read_prompt("step1_inline_summary.txt"), f"Inline text:\n{passages}"
    )
    summary = strip_label(raw, ("Inline summary", "Output inline summary"))
    if not summary:
        raise ValueError("empty inline summary")
    record["Inline Summary"] = summary
    if args.include_raw:
        record.setdefault("construction_raw_outputs", {})["step1"] = raw
    return record


def transform_step2(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    user = (
        f"Compound-figure caption: {caption(record)}\n"
        f"Inline summary: {inline_summary(record)}"
    )
    if not caption(record) and not inline_summary(record):
        raise SkipRecord("caption and inline summary are both missing")
    raw = generator.generate(read_prompt("step2_medical_knowledge.txt"), user)
    parsed = extract_sections(raw, ("Keywords", "Medical knowledge"))
    if not parsed.get("Medical knowledge"):
        raise ValueError("output did not contain a Medical knowledge field")
    keyword_text = parsed.get("Keywords", "")
    record["keywords"] = [
        item.strip().strip("[]\"'") for item in keyword_text.split(",") if item.strip()
    ]
    record["medical knowledge"] = parsed["Medical knowledge"]
    if args.include_raw:
        record.setdefault("construction_raw_outputs", {})["step2"] = raw
    return record


def transform_step3(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    items = subimages(record)
    if not items:
        raise SkipRecord("record contains no constituent subimages")
    prompt = read_prompt("step3_visual_description.txt")
    updated: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    for order, item in enumerate(items):
        name = image_name(item)
        if not name:
            raise SkipRecord(f"subimage {order + 1} has no image identifier")
        user = "\n".join(
            [
                f"Subimage: {fig_id(item, order)}",
                f"Subimage caption: {subimage_caption(item)}",
                f"Compound-figure caption: {caption(record)}",
                f"Inline summary: {inline_summary(record)}",
                f"Medical knowledge: {medical_knowledge(record)}",
            ]
        )
        raw = generator.generate(prompt, user, [resolve_image_path(name, args)])
        description = strip_label(raw, ("Visual description", "Output visual description"))
        if not description:
            raise ValueError(f"empty visual description for {name}")
        item["visual perception description"] = description
        item["fig_id"] = fig_id(item, order)
        updated.append(item)
        raw_outputs.append(raw)
    record["subcaptions"] = updated
    if args.include_raw:
        record.setdefault("construction_raw_outputs", {})["step3"] = raw_outputs
    return record


def load_classification_scores(
    path: Path | None, filename_column: str = "filename", score_column: str = "prob"
) -> dict[str, float]:
    if path is None:
        return {}
    result: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get(filename_column) and row.get(score_column) not in (None, ""):
                result[str(row[filename_column])] = float(row[score_column])
    return result


def eligible_subimages(record: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = subimages(record)
    scores: dict[str, float] = getattr(args, "_classification_scores", {})
    if not scores:
        return [item for item in candidates if visual_description(item)]
    threshold = args.classification_threshold
    return [
        item
        for item in candidates
        if visual_description(item) and scores.get(image_name(item), 0.0) > threshold
    ]


def formatted_subimages(items: Sequence[Mapping[str, Any]], all_items: Sequence[Mapping[str, Any]]) -> str:
    positions = {image_name(item): index for index, item in enumerate(all_items)}
    chunks: list[str] = []
    for fallback_order, item in enumerate(items):
        order = positions.get(image_name(item), fallback_order)
        identifier = fig_id(item, order)
        chunks.extend(
            [
                f"Identifier: {identifier}",
                f"Caption: {subimage_caption(item)}",
                f"Visual description: {visual_description(item)}",
            ]
        )
    return "\n".join(chunks)


def parse_cqa(raw: str, *, context_required: bool = True) -> dict[str, str]:
    parsed = extract_sections(raw, ("Context", "Question", "Answer"))
    required = ["Question", "Answer"] + (["Context"] if context_required else [])
    missing = [name for name in required if not parsed.get(name)]
    if missing:
        raise ValueError(f"output missing labelled field(s): {', '.join(missing)}")
    return {
        "context": parsed.get("Context", ""),
        "question": parsed["Question"],
        "answer": parsed["Answer"],
    }


def parse_options(value: str) -> dict[str, str]:
    normalized = value.replace("\n", " ").strip()
    positions: list[tuple[int, int, str]] = []
    for label in ("A", "B", "C", "D"):
        match = re.search(rf"(?:^|;|\s){label}[.)]\s*", normalized)
        if match:
            positions.append((match.start(), match.end(), label))
    positions.sort()
    result: dict[str, str] = {}
    for index, (_, start, label) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(normalized)
        result[label] = normalized[start:end].strip().rstrip(";")
    if set(result) != {"A", "B", "C", "D"}:
        raise ValueError("Options must contain exactly one A, B, C, and D entry")
    return result


def add_raw(record: dict[str, Any], task: str, raw: str, args: argparse.Namespace) -> None:
    if args.include_raw:
        record.setdefault("construction_raw_outputs", {})[f"step4_{task}"] = raw


def transform_step4_single(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    candidates = eligible_subimages(record, args)
    if not candidates:
        raise SkipRecord("no eligible subimage with a visual description")
    selected = random.Random(args.seed + row_index).choice(candidates)
    all_items = subimages(record)
    order = next((i for i, item in enumerate(all_items) if image_name(item) == image_name(selected)), 0)
    identifier = fig_id(selected, order)
    user = "\n".join(
        [
            f"Compound-figure caption: {caption(record)}",
            f"Inline summary: {inline_summary(record)}",
            f"Medical knowledge: {medical_knowledge(record)}",
            "Target visual-input type: compound figure plus constituent subimages",
            f"Selected subimage identifier: {identifier}",
            f"Selected subimage caption: {subimage_caption(selected)}",
            f"Selected subimage visual description: {visual_description(selected)}",
        ]
    )
    raw = generator.generate(read_prompt("step4_single_subimage_vqa.txt"), user)
    record["task_type"] = "single_subimage_vqa"
    record["qa_pairs"] = [{**parse_cqa(raw), "selected_image_paths": [image_name(selected)], "image_index": [identifier]}]
    add_raw(record, "single_subimage_vqa", raw, args)
    return record


def transform_step4_multi(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    candidates = eligible_subimages(record, args)
    rng = random.Random(args.seed + row_index)
    requested = rng.choice(list(range(args.min_selected, args.max_selected + 1)))
    if len(candidates) < requested:
        raise SkipRecord(f"requires {requested} eligible subimages; found {len(candidates)}")
    selected = rng.sample(candidates, requested)
    all_items = subimages(record)
    positions = {image_name(item): i for i, item in enumerate(all_items)}
    identifiers = [fig_id(item, positions.get(image_name(item), i)) for i, item in enumerate(selected)]
    user = "\n".join(
        [
            f"Compound-figure caption: {caption(record)}",
            f"Inline summary: {inline_summary(record)}",
            f"Medical knowledge: {medical_knowledge(record)}",
            "Target visual-input type: compound figure plus constituent subimages",
            f"Selected subimages: {', '.join(identifiers)}",
            formatted_subimages(selected, all_items),
        ]
    )
    raw = generator.generate(read_prompt("step4_multi_subimage_vqa.txt"), user)
    record["task_type"] = "multi_subimage_vqa"
    record["qa_pairs"] = [
        {
            **parse_cqa(raw),
            "selected_image_paths": [image_name(item) for item in selected],
            "image_index": identifiers,
        }
    ]
    add_raw(record, "multi_subimage_vqa", raw, args)
    return record


def centre(item: Mapping[str, Any]) -> tuple[float, float] | None:
    value = first_value(item, "bbox_center", "bounding_box_centre", "centre", "center", default=None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    bbox = first_value(item, "bbox", "bounding_box", default=None)
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) >= 4:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        return (x1 + x2) / 2, (y1 + y2) / 2
    return None


def transform_step4_relative(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    items = subimages(record)
    pair = first_value(record, "selected_subimages", "relative_position_pair", default=None)
    if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) >= 2:
        by_name = {image_name(item): item for item in items}
        selected = [by_name.get(str(pair[0])), by_name.get(str(pair[1]))]
        if any(item is None for item in selected):
            raise SkipRecord("selected relative-position subimages were not found")
        first, second = selected  # type: ignore[misc]
    elif len(items) >= 2:
        first, second = items[0], items[1]
    else:
        raise SkipRecord("relative-position construction requires two subimages")
    first_centre, second_centre = centre(first), centre(second)
    relation = as_text(first_value(record, "relative_position", "spatial_relation", "position"))
    if first_centre is None or second_centre is None or not relation:
        raise SkipRecord("bounding-box centres or precomputed relative-position label are missing")
    first_order, second_order = items.index(first), items.index(second)
    first_id, second_id = fig_id(first, first_order), fig_id(second, second_order)
    user = "\n".join(
        [
            f"First subimage: {first_id}",
            f"First bounding-box centre: {first_centre}",
            f"Second subimage: {second_id}",
            f"Second bounding-box centre: {second_centre}",
            f"Precomputed relative position: {relation}",
            "Coordinate convention: image origin at the upper left; x increases to the right and y increases downward",
        ]
    )
    raw = generator.generate(read_prompt("step4_relative_position_vqa.txt"), user)
    parsed = parse_cqa(raw, context_required=False)
    record["task_type"] = "relative_position_vqa"
    record["qa_pairs"] = [
        {
            "question": parsed["question"],
            "answer": parsed["answer"],
            "selected_image_paths": [image_name(first), image_name(second)],
            "image_index": [first_id, second_id],
            "relative_position": relation,
        }
    ]
    add_raw(record, "relative_position_vqa", raw, args)
    return record


def transform_step4_compound(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    items = subimages(record)
    if not any(visual_description(item) for item in items):
        raise SkipRecord("constituent-subimage visual descriptions are missing")
    descriptions = formatted_subimages(items, items)
    user = "\n".join(
        [
            "Target visual-input type: compound figure (Fig-0)",
            f"Compound-figure caption: {caption(record)}",
            f"Inline summary: {inline_summary(record)}",
            f"Medical knowledge: {medical_knowledge(record)}",
            f"Constituent-subimage descriptions:\n{descriptions}",
        ]
    )
    raw = generator.generate(read_prompt("step4_compound_figure_vqa.txt"), user)
    record["task_type"] = "compound_figure_vqa"
    record["qa_pairs"] = [{**parse_cqa(raw), "image_index": ["Fig-0"]}]
    add_raw(record, "compound_figure_vqa", raw, args)
    return record


def transform_step4_text(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    user = "\n".join(
        [
            f"Compound-figure caption: {caption(record)}",
            f"Inline summary: {inline_summary(record)}",
            f"Medical knowledge: {medical_knowledge(record)}",
        ]
    )
    raw = generator.generate(read_prompt("step4_text_only_qa.txt"), user)
    record["task_type"] = "text_only_qa"
    record["qa_pairs"] = [parse_cqa(raw)]
    add_raw(record, "text_only_qa", raw, args)
    return record


def transform_step4_mcq(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    items = eligible_subimages(record, args)
    if args.target_mode == "compound":
        identifiers = ["Fig-0"]
        target_type = "compound figure"
        relevant = subimages(record)
    else:
        if not items:
            raise SkipRecord("no eligible subimage for multiple-choice VQA")
        selected = random.Random(args.seed + row_index).choice(items)
        all_items = subimages(record)
        order = next((i for i, item in enumerate(all_items) if image_name(item) == image_name(selected)), 0)
        identifiers = [fig_id(selected, order)]
        target_type = "multi-image input"
        relevant = [selected]
    correct_position = as_text(
        first_value(record, "required_correct_option_position", "correct_option_position")
    ).upper()
    if correct_position not in {"A", "B", "C", "D"}:
        correct_position = ("A", "B", "C", "D")[(row_index + args.seed) % 4]
    user = "\n".join(
        [
            f"Target visual-input type: {target_type}",
            f"Target visual identifiers: {', '.join(identifiers)}",
            f"Compound-figure caption: {caption(record)}",
            f"Inline summary: {inline_summary(record)}",
            f"Relevant visual descriptions:\n{formatted_subimages(relevant, subimages(record))}",
            f"Medical knowledge: {medical_knowledge(record)}",
            f"Required correct-option position: {correct_position}",
        ]
    )
    raw = generator.generate(read_prompt("step4_multiple_choice_vqa.txt"), user)
    parsed = extract_sections(raw, ("Question", "Options", "Correct answer"))
    if any(not parsed.get(field) for field in ("Question", "Options", "Correct answer")):
        raise ValueError("output must contain Question, Options, and Correct answer")
    answer = parsed["Correct answer"].strip().upper().rstrip(".")
    if answer != correct_position:
        raise ValueError(
            f"model returned correct option {answer!r}; required position was {correct_position}"
        )
    record["task_type"] = "multiple_choice_vqa"
    record["qa_pairs"] = [
        {
            "question": parsed["Question"],
            "options": parse_options(parsed["Options"]),
            "correct_answer": answer,
            "image_index": identifiers,
        }
    ]
    add_raw(record, "multiple_choice_vqa", raw, args)
    return record


def transform_step5(
    record: dict[str, Any], row_index: int, generator: Generator, args: argparse.Namespace
) -> dict[str, Any]:
    task_type = as_text(first_value(record, "task_type", "instruction_type"))
    if task_type in {"relative_position_vqa", "multiple_choice_vqa"}:
        raise SkipRecord(f"{task_type} has no auxiliary context to refine")
    pairs = record.get("qa_pairs")
    if not isinstance(pairs, list) or not pairs:
        if first_value(record, "question", "Question") and first_value(record, "answer", "Answer"):
            pairs = [
                {
                    "context": first_value(record, "context", "Context"),
                    "question": first_value(record, "question", "Question"),
                    "answer": first_value(record, "answer", "Answer"),
                }
            ]
        else:
            raise SkipRecord("record contains no context-question-answer pair")
    descriptions = formatted_subimages(subimages(record), subimages(record)) if subimages(record) else ""
    refined_pairs: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("qa_pairs must contain JSON objects")
        draft_context = as_text(first_value(pair, "context", "Context"))
        question = as_text(first_value(pair, "question", "Question"))
        answer = as_text(first_value(pair, "answer", "Answer", "reference_answer"))
        if not question or not answer:
            raise SkipRecord("question or reference answer is missing")
        user = "\n".join(
            [
                f"Compound-figure caption: {caption(record)}",
                f"Inline summary: {inline_summary(record)}",
                f"Medical knowledge: {medical_knowledge(record)}",
                f"Visual descriptions:\n{descriptions}",
                f"Question: {question}",
                f"Reference answer: {answer}",
                f"Draft context: {draft_context}",
            ]
        )
        raw = generator.generate(read_prompt("step5_context_refinement.txt"), user)
        refined = strip_label(raw, ("Refined context", "Output refined context"))
        if not refined:
            raise ValueError("empty refined context")
        updated = dict(pair)
        updated["draft_context"] = draft_context
        updated["context"] = refined
        refined_pairs.append(updated)
        raw_outputs.append(raw)
    record["qa_pairs"] = refined_pairs
    if args.include_raw:
        record.setdefault("construction_raw_outputs", {})["step5"] = raw_outputs
    return record
