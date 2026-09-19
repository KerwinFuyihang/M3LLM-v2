"""Shared I/O, generation backends, parsing, and resume utilities.

The historical construction scripts used local Hugging Face checkpoints.  This
module keeps that route and adds an OpenAI-compatible route so the same scripts
can target a locally served checkpoint (for example, through vLLM) without
embedding cluster paths or credentials.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"


class SkipRecord(RuntimeError):
    """Raised when a record does not meet the documented task requirements."""


def read_prompt(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSON array or JSONL file while preserving record order."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "records", "items"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array or use JSONL format")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: array item {index} is not a JSON object")
        yield record


def record_key(record: Mapping[str, Any], row_index: int, id_field: str | None) -> str:
    candidates = ([id_field] if id_field else []) + [
        "sample_id",
        "sampleID",
        "id",
        "compound_figure_id",
        "image",
    ]
    for field in candidates:
        if field and record.get(field) not in (None, ""):
            return str(record[field])
    stable = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"row-{row_index}-{hashlib.sha1(stable).hexdigest()[:12]}"


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("_construction_key")
            if key is not None:
                keys.add(str(key))
    return keys


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def first_value(record: Mapping[str, Any], *fields: str, default: Any = "") -> Any:
    for field in fields:
        value = record.get(field)
        if value not in (None, ""):
            return value
    return default


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def strip_label(text: str, labels: Sequence[str]) -> str:
    result = text.strip()
    result = re.sub(r"^```(?:text|markdown)?\s*", "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s*```$", "", result).strip()
    for label in labels:
        pattern = (
            rf"^\s*(?:[-*]\s*)?(?:\*\*)?(?:Output\s*:\s*)?"
            rf"{re.escape(label)}(?:\*\*)?\s*:\s*"
        )
        result = re.sub(pattern, "", result, count=1, flags=re.IGNORECASE)
    return result.strip()


def extract_sections(text: str, names: Sequence[str]) -> dict[str, str]:
    """Parse labelled sections such as Context/Question/Answer."""
    clean = text.strip()
    positions: list[tuple[int, int, str]] = []
    for name in names:
        pattern = re.compile(
            rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:Output\s+)?"
            rf"{re.escape(name)}(?:\*\*)?\s*:\s*"
        )
        match = pattern.search(clean)
        if match:
            positions.append((match.start(), match.end(), name))
    positions.sort()
    values: dict[str, str] = {}
    for index, (_, content_start, name) in enumerate(positions):
        content_end = positions[index + 1][0] if index + 1 < len(positions) else len(clean)
        values[name] = clean[content_start:content_end].strip()
    return values


def image_as_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str] = (),
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleGenerator(Generator):
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package for this backend") from exc

        api_key = os.environ.get(args.api_key_env, "EMPTY")
        self.client = OpenAI(base_url=args.base_url, api_key=api_key)
        self.model = args.model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_tokens = args.max_new_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str] = (),
    ) -> str:
        if image_paths:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for path in image_paths:
                content.append(
                    {"type": "image_url", "image_url": {"url": image_as_data_url(path)}}
                )
            user_content: Any = content
        else:
            user_content = user_prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class HFTextGenerator(Generator):
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers for the hf-text backend") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map=args.device_map,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str] = (),
    ) -> str:
        if image_paths:
            raise ValueError("hf-text does not accept images; use openai-compatible or huatuo-local")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        sampling = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": sampling,
        }
        if sampling:
            generation_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class HuatuoLocalGenerator(Generator):
    """Adapter for the HuatuoGPT-Vision implementation already bundled here."""

    def __init__(self, args: argparse.Namespace) -> None:
        import sys

        repository_root = ROOT.parent
        sys.path.insert(0, str(repository_root))
        try:
            from Evaluation.MedEvalKit.models.HuatuoGPT.cli import HuatuoChatbot  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the bundled HuatuoGPT-Vision adapter; install its requirements"
            ) from exc
        self.bot = HuatuoChatbot(args.model)
        sampling = args.temperature > 0
        # Update rather than replace so the adapter's model-specific EOS and
        # padding token identifiers remain intact.
        self.bot.gen_kwargs.update(
            {
                "do_sample": sampling,
                "max_new_tokens": args.max_new_tokens,
                "min_new_tokens": 1,
                "repetition_penalty": args.repetition_penalty,
            }
        )
        if sampling:
            self.bot.gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        else:
            self.bot.gen_kwargs.pop("temperature", None)
            self.bot.gen_kwargs.pop("top_p", None)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str] = (),
    ) -> str:
        if not image_paths:
            raise ValueError("huatuo-local requires at least one image")
        outputs = self.bot.inference(
            f"{system_prompt}\n\n{user_prompt}", [str(path) for path in image_paths]
        )
        return str(outputs[0]).strip()


def make_generator(args: argparse.Namespace, *, images_required: bool = False) -> Generator:
    if args.backend == "openai-compatible":
        return OpenAICompatibleGenerator(args)
    if args.backend == "hf-text":
        if images_required:
            raise ValueError("Step 3 requires openai-compatible or huatuo-local")
        return HFTextGenerator(args)
    if args.backend == "huatuo-local":
        return HuatuoLocalGenerator(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_backend: str = "hf-text",
) -> None:
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL")
    parser.add_argument(
        "--backend",
        choices=("hf-text", "openai-compatible", "huatuo-local"),
        default=default_backend,
    )
    parser.add_argument("--model", default=default_model, help="Checkpoint or served model name")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Default: <output>.errors.jsonl",
    )


Transform = Callable[[dict[str, Any], int, Generator, argparse.Namespace], dict[str, Any]]


def run_pipeline(args: argparse.Namespace, transform: Transform, *, images_required: bool = False) -> None:
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    failure_log = args.failure_log or args.output.with_suffix(args.output.suffix + ".errors.jsonl")
    if args.overwrite:
        failure_log.unlink(missing_ok=True)

    done = completed_keys(args.output)
    generator = make_generator(args, images_required=images_required)
    processed = skipped = failed = 0
    for row_index, record in enumerate(read_records(args.input)):
        key = record_key(record, row_index, args.id_field)
        if key in done:
            skipped += 1
            continue
        try:
            result = transform(dict(record), row_index, generator, args)
            result["_construction_key"] = key
            append_jsonl(args.output, result)
            processed += 1
        except SkipRecord as exc:
            append_jsonl(
                failure_log,
                {"_construction_key": key, "status": "skipped", "reason": str(exc)},
            )
            skipped += 1
        except Exception as exc:  # records the item and allows long jobs to continue
            append_jsonl(
                failure_log,
                {
                    "_construction_key": key,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
            )
            failed += 1
    print(f"processed={processed} skipped={skipped} failed={failed} output={args.output}")
