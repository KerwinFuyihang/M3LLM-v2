#!/usr/bin/env python3
"""Run the response-quality rubric through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = ("id", "question", "reference_answer", "candidate_response")
DIMENSIONS = ("completeness", "clarity", "correctness")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing required fields: {', '.join(missing)}"
                )
            rows.append(row)
    return rows


def render_item_prompt(template: str, row: dict[str, Any]) -> str:
    return (
        template.replace("{{QUESTION}}", str(row["question"]))
        .replace("{{REFERENCE_ANSWER}}", str(row["reference_answer"]))
        .replace("{{CANDIDATE_RESPONSE}}", str(row["candidate_response"]))
    )


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object.")
    return value


def validate_scores(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        entry = value.get(dimension)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing object for {dimension}.")
        score = entry.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{dimension}.score must be an integer from 1 to 5.")
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str):
            raise ValueError(f"{dimension}.rationale must be a string.")
        validated[dimension] = {"score": score, "rationale": rationale.strip()}
    return validated


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["id"]))
    return completed


def request_judgement(
    client: Any,
    config: dict[str, Any],
    system_prompt: str,
    item_prompt: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    request: dict[str, Any] = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item_prompt},
        ],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "seed": config["seed"],
        "response_format": config["response_format"],
    }
    if config.get("extra_body"):
        request["extra_body"] = config["extra_body"]

    last_error: Exception | None = None
    for attempt in range(1, int(config["max_retries"]) + 1):
        try:
            response = client.chat.completions.create(**request)
            raw = response.choices[0].message.content or ""
            return validate_scores(parse_json_object(raw)), raw
        except Exception as error:  # provider errors and malformed outputs are retried
            last_error = error
            if attempt == int(config["max_retries"]):
                break
            time.sleep(min(2 ** (attempt - 1), 16))
    raise RuntimeError(f"Judge request failed after {config['max_retries']} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "open_weight_deepseek_v4_flash.json",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=ROOT / "prompts" / "system_prompt.txt",
    )
    parser.add_argument(
        "--user-prompt",
        type=Path,
        default=ROOT / "prompts" / "user_prompt.txt",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    user_template = args.user_prompt.read_text(encoding="utf-8").strip()
    rows = read_jsonl(args.input)
    completed = load_completed_ids(args.output) if args.resume else set()

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from error

    api_key = os.environ.get(config["api_key_env"])
    if not api_key and str(config["base_url"]).startswith(
        ("http://localhost", "http://127.0.0.1")
    ):
        api_key = "EMPTY"
    if not api_key:
        raise SystemExit(f"Set {config['api_key_env']} for endpoint {config['base_url']}.")

    client = OpenAI(
        api_key=api_key,
        base_url=config["base_url"],
        timeout=float(config["timeout_seconds"]),
        max_retries=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            item_id = str(row["id"])
            if item_id in completed:
                continue
            item_prompt = render_item_prompt(user_template, row)
            scores, raw = request_judgement(client, config, system_prompt, item_prompt)
            prompt_digest = hashlib.sha256(
                (system_prompt + "\n\n" + item_prompt).encode("utf-8")
            ).hexdigest()
            output = {
                "id": item_id,
                "candidate_model": row.get("candidate_model"),
                "judge_model": config["model"],
                "scores": scores,
                "parameters": {
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_tokens": config["max_tokens"],
                    "seed": config["seed"],
                },
                "prompt_sha256": prompt_digest,
                "raw_response": raw,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {item_id}")


if __name__ == "__main__":
    main()
