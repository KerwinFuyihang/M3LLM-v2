"""Hulu-Med wrapper (HF only, trust_remote_code).

Requires: opencv-python, ffmpeg-python, imageio, decord.
Recommended: transformers==4.51.2.
"""

import importlib.util
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def _image_content(img):
    if isinstance(img, str):
        return {"type": "image", "image": {"image_path": img}}
    if isinstance(img, Image.Image):
        return {"type": "image", "image": [img.convert("RGB")]}
    raise TypeError(f"Unsupported image type: {type(img)}")


class HuluMed:
    def __init__(self, model_path, args):
        super().__init__()
        attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def process_messages(self, messages):
        conversation = []
        has_image = False

        if "messages" in messages:
            for message in messages["messages"]:
                conversation.append({
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                })
        else:
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

            content = []
            if system:
                content.append({"type": "text", "text": system + "\n"})
            for i, img in enumerate(raw_imgs):
                try:
                    item = _image_content(img)
                except Exception as e:
                    print(f"[Warning] failed to prepare image: {e}")
                    continue
                if len(raw_imgs) > 1:
                    content.append({"type": "text", "text": f"<image_{i + 1}>: "})
                content.append(item)
                has_image = True
            content.append({"type": "text", "text": text})
            conversation.append({"role": "user", "content": content})

        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.llm.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs, has_image

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
            gen_kwargs["modals"] = ["text"]

        with torch.inference_mode():
            output_ids = self.llm.generate(**inputs, **gen_kwargs)
        output = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, use_think=False
        )[0].strip()
        return output

    def generate_outputs(self, messages_list):
        res = []
        for messages in messages_list:
            res.append(self.generate_output(messages))
        return res
