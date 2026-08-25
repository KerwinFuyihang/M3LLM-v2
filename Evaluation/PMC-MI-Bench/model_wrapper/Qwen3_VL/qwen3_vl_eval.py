import io

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from model_wrapper.vlm_base import VLMBase


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
    labels = [] if labels is None else list(labels)
    out = []
    for i in range(n):
        lab = labels[i] if i < len(labels) else None
        out.append(_norm_one_label(lab, i))
    return out


def _to_rgb_image(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (bytes, bytearray, io.BytesIO)):
        value = img if isinstance(img, (bytes, bytearray)) else img.getvalue()
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)!r}")


class Qwen3_VL(VLMBase):
    def __init__(self, model_path, args):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens

    def generate_output(self, messages: dict) -> str:
        user_text = messages.get("prompt", "") or ""
        system_text = messages.get("system", "") or ""

        imgs_in = []
        if "images" in messages:
            imgs_in = (
                messages["images"]
                if isinstance(messages["images"], list)
                else [messages["images"]]
            )
        elif "image" in messages:
            imgs_in = [messages["image"]]

        pil_images = [_to_rgb_image(img) for img in imgs_in]

        content = []
        if pil_images:
            labels = _norm_fig_labels(messages.get("image_indices"), len(pil_images))
            for lab, pil in zip(labels, pil_images):
                content.append({"type": "text", "text": f"{lab}:"})
                content.append({"type": "image", "image": pil})
            if user_text:
                content.append({"type": "text", "text": user_text})
        else:
            content.append({"type": "text", "text": user_text})

        full_messages = []
        if system_text:
            full_messages.append({"role": "system", "content": system_text})
        full_messages.append({"role": "user", "content": content})

        prompt = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(full_messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
        )

        cut_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            cut_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
