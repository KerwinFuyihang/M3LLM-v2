"""Wrapper for UFNLP/MedGPT-oss (GPT-oss-20B backbone + CLIP ViT-L/14@336).

Notes (from the official model card):
- Load with AutoModel + trust_remote_code; do NOT use AutoProcessor
  (the repo does not register an image processor).
- Inference goes through model.chat(tokenizer, pixel_values, question,
  generation_config); it handles the Harmony system message, the
  <|return|> stop token, and strips the channel envelope.
- Images: InternVL-style dynamic tiling at 336px (see image_utils.py),
  one "<image>" placeholder per image in the question string.
- Requires transformers>=4.55 and an approved (gated) HF access token.
"""

import os

import torch
from transformers import AutoModel, AutoTokenizer

from .image_utils import load_image
from model_wrapper.vlm_base import VLMBase
from model_wrapper.hf_cache import get_hf_cache_dir


def _norm_one_label(x, fallback_i):
    if x is None:
        return f"Fig-{fallback_i}"
    s = str(x).strip()
    if s.lower().startswith("fig-"):
        s = s.split("-", 1)[1]
    try:
        return f"Fig-{int(s)}"
    except Exception:
        return s if s.lower().startswith("fig-") else f"Fig-{s}"


def _norm_fig_labels(labels, n):
    if not labels:
        labels = []
    out = []
    for i in range(n):
        lab = labels[i] if i < len(labels) else None
        out.append(_norm_one_label(lab, i))
    return out


class MedGPTOSS(VLMBase):
    def __init__(self, model_path, args):
        super().__init__()
        cache_dir = getattr(args, "cache_dir", None) or get_hf_cache_dir()
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=cache_dir,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False, cache_dir=cache_dir
        )

        self.generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature != 0,
            "repetition_penalty": args.repetition_penalty,
        }
        if args.temperature != 0:
            self.generation_config["temperature"] = args.temperature
            self.generation_config["top_p"] = args.top_p

        self.max_tiles = int(os.environ.get("medgpt_max_tiles_per_image", "12"))

    def process_messages(self, messages):
        system = messages.get("system", "") or ""
        text = messages.get("prompt", "") or ""

        raw_imgs = []
        if "image" in messages and messages["image"] is not None:
            raw_imgs = [messages["image"]]
        elif "images" in messages and messages["images"]:
            raw_imgs = list(messages["images"])

        loaded = [load_image(img, max_num=self.max_tiles) for img in raw_imgs]

        if not loaded:
            question = (system + "\n" if system and text else system) + text
            return {"question": question, "pixel_values": None}

        labels = _norm_fig_labels(messages.get("image_indices"), len(loaded))
        lines = [f"{lab}: <image>" for lab in labels]
        question = (
            (system + "\n" if system else "")
            + "\n".join(lines)
            + (("\n" + text) if text else "")
        )
        pixel_values = torch.cat(loaded, dim=0).to(torch.bfloat16).to(self.model.device)
        return {"question": question, "pixel_values": pixel_values}

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        response = self.model.chat(
            self.tokenizer,
            pixel_values=llm_inputs["pixel_values"],
            question=llm_inputs["question"],
            generation_config=dict(self.generation_config),
        )
        return response

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]
