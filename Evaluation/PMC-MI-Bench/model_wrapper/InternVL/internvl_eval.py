import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import load_image
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


def _load_to_bf16_cuda(image):
    tensor = load_image(image).to(dtype=torch.bfloat16, device="cuda")
    return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor


class InternVL(VLMBase):
    def __init__(self, model_path, args):
        super().__init__()
        cache_dir = getattr(args, "cache_dir", None) or get_hf_cache_dir()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda",
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            cache_dir=cache_dir,
        )
        self.generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.temperature > 0,
        }

    def process_messages(self, messages):
        prompt = messages.get("system", "") or ""
        text = messages.get("prompt", "") or ""

        if ("image" not in messages) and ("images" not in messages):
            return {
                "prompt": (prompt + ("\n" if prompt and text else "") + text),
                "pixel_values": None,
            }

        if "image" in messages and "images" not in messages:
            tensor = _load_to_bf16_cuda(messages["image"])
            labels = messages.get("image_indices") or ["Fig-0"]
            label = _norm_fig_labels(labels, 1)[0]
            full_prompt = f"{prompt}\n{label}: <image>" + ("\n" + text if text else "")
            return {"prompt": full_prompt, "pixel_values": tensor}

        raw_imgs = messages.get("images") or []
        if not isinstance(raw_imgs, (list, tuple)):
            raw_imgs = [raw_imgs]
        if not raw_imgs:
            raise ValueError("The images field is present but empty.")
        loaded = [_load_to_bf16_cuda(image) for image in raw_imgs]
        labels = _norm_fig_labels(messages.get("image_indices"), len(loaded))
        lines = [f"{lab}: <image>" for lab in labels]
        full_prompt = prompt + ("\n" if prompt else "") + "\n".join(lines)
        full_prompt += ("\n" + text) if text else ""
        pixel_values = torch.cat(loaded, dim=0)
        return {"prompt": full_prompt, "pixel_values": pixel_values}

    def generate_output(self, messages):
        llm_inputs = self.process_messages(messages)
        question = llm_inputs["prompt"]
        pixel_values = llm_inputs["pixel_values"]

        if pixel_values is None:
            try:
                resp, _ = self.model.chat(
                    self.tokenizer,
                    None,
                    question,
                    self.generation_config,
                    history=None,
                    return_history=True,
                )
                return resp
            except Exception:
                inputs = self.tokenizer(question, return_tensors="pt").to(
                    self.model.device
                )
                gen_ids = self.model.generate(**inputs, **self.generation_config)
                cut = inputs["input_ids"].shape[1]
                return self.tokenizer.decode(gen_ids[0][cut:], skip_special_tokens=True)

        response, _ = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            history=None,
            return_history=True,
        )
        return response

    def generate_outputs(self, messages_list):
        return [self.generate_output(messages) for messages in messages_list]
