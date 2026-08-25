"""MedGPT-oss wrapper (HF only, trust_remote_code).

Uses AutoModel + AutoTokenizer (no AutoProcessor).
Requires transformers>=4.55; gated HF repo needs HF_TOKEN.
"""

import os

import torch
from transformers import AutoModel, AutoTokenizer

from .image_utils import load_image


class MedGPT_OSS:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
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

    def _safe_load(self, img):
        try:
            return load_image(img, max_num=self.max_tiles)
        except Exception as e:
            print(f"[Warning] failed to load image: {e}")
            return None

    def process_messages(self, messages):
        if "messages" in messages:
            parts = []
            for message in messages["messages"]:
                parts.append(f'{message["role"]}: {message["content"]}')
            return {"question": "\n".join(parts), "pixel_values": None}

        system = messages.get("system", "") or ""
        text = messages.get("prompt", "") or ""

        raw_imgs = []
        if "image" in messages and messages["image"] is not None:
            raw_imgs = [messages["image"]]
        elif "images" in messages and messages["images"]:
            raw_imgs = list(messages["images"])
            max_images = int(os.environ.get("max_image_num", "0"))
            if max_images > 0 and len(raw_imgs) > max_images:
                raw_imgs = raw_imgs[:max_images]

        loaded = []
        for img in raw_imgs:
            t = self._safe_load(img)
            if t is not None:
                loaded.append(t)

        if not loaded:
            question = (system + "\n" if system and text else system) + text
            return {"question": question, "pixel_values": None}

        if len(loaded) == 1:
            header = "<image>"
        else:
            header = "\n".join(
                f"<image_{i + 1}>: <image>" for i in range(len(loaded))
            )
        question = (system + "\n" if system else "") + header + (
            "\n" + text if text else ""
        )
        pixel_values = torch.cat(loaded, dim=0).to(torch.bfloat16).to(self.llm.device)
        return {"question": question, "pixel_values": pixel_values}

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        response = self.llm.chat(
            self.tokenizer,
            pixel_values=llm_inputs["pixel_values"],
            question=llm_inputs["question"],
            generation_config=dict(self.generation_config),
        )
        return response

    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            res.append(self.generate_output(messages))
        return res
