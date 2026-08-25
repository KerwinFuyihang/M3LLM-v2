"""Wrapper for ZJU-AI4H/Hulu-Med-7B (Qwen2.5 backbone + SigLIP encoder).

Notes (from the official model card / remote code):
- Custom architecture (HulumedQwen2...) loaded via AutoModelForCausalLM
  with trust_remote_code=True; processor is a custom HulumedProcessor
  loaded via AutoProcessor(trust_remote_code=True).
- The processor takes a `conversation=[...]` kwarg (not text/images):
  content items look like {"type": "image", "image": {"image_path": p}}
  or {"type": "image", "image": [PIL.Image]}, plus {"type": "text", ...}.
- Decode with processor.batch_decode(..., use_think=False) which strips
  <think>...</think> segments.
- The remote processing code imports cv2/ffmpeg/imageio/decord, so:
  pip install opencv-python ffmpeg-python imageio decord
  Recommended: transformers==4.51.2 (per model card), flash-attn optional.
"""

import importlib.util

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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


def _image_content(img):
    """Build a HulumedProcessor image content item from a path or PIL image."""
    if isinstance(img, str):
        return {"type": "image", "image": {"image_path": img}}
    if isinstance(img, Image.Image):
        return {"type": "image", "image": [img.convert("RGB")]}
    raise TypeError(f"Unsupported image type: {type(img)}")


class HuluMed(VLMBase):
    def __init__(self, model_path, args):
        super().__init__()
        cache_dir = getattr(args, "cache_dir", None) or get_hf_cache_dir()
        attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            cache_dir=cache_dir,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, cache_dir=cache_dir
        )

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self, messages):
        system = messages.get("system", "") or ""
        text = messages.get("prompt", "") or ""

        raw_imgs = []
        if "image" in messages and messages["image"] is not None:
            raw_imgs = [messages["image"]]
        elif "images" in messages and messages["images"]:
            raw_imgs = list(messages["images"])

        content = []
        if system:
            content.append({"type": "text", "text": system + "\n"})

        labels = _norm_fig_labels(messages.get("image_indices"), len(raw_imgs))
        for lab, img in zip(labels, raw_imgs):
            item = _image_content(img)
            if len(raw_imgs) > 1:
                content.append({"type": "text", "text": f"{lab}: "})
            content.append(item)

        content.append({"type": "text", "text": text})
        conversation = [{"role": "user", "content": content}]

        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs, bool(raw_imgs)

    def generate_output(self, messages):
        inputs, has_image = self.process_messages(messages)
        do_sample = self.temperature != 0
        gen_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        if not has_image:
            # "modals" is included by the processor when images are present.
            gen_kwargs["modals"] = ["text"]

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        output = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, use_think=False
        )[0].strip()
        return output

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]
