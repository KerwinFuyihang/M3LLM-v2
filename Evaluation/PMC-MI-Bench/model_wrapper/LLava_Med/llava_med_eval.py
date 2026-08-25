import os
import torch

from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM


from .utils import (
    LlavaMistralConfig,
    LlavaMistralForCausalLM,
    load_pretrained_model,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from .conversation import conv_templates
from .mm_utils import tokenizer_image_token, process_images
from model_wrapper.vlm_base import VLMBase
from model_wrapper.hf_cache import get_hf_cache_dir

AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)


def _figure_labels(labels, count):
    labels = [] if labels is None else list(labels)
    return [str(labels[i]) if i < len(labels) else f"Fig-{i}" for i in range(count)]


class LLavaMed(VLMBase):
    def __init__(self, model_path, args):
        super().__init__()
        cache_dir = getattr(args, "cache_dir", None) or get_hf_cache_dir()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava-med-v1.5-mistral-7b",
            cache_dir=cache_dir,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens

    def process_messages(self, messages):
        conv = conv_templates["mistral_instruct"]
        conv.messages = []
        if "system" in messages:
            conv.system = messages["system"]

        imgs = []
        if "image" in messages:
            image = messages["image"]
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
            imgs.append(image)
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + messages["prompt"]
        elif "images" in messages:
            images = messages["images"]
            if not isinstance(images, (list, tuple)):
                images = [images]
            labels = _figure_labels(messages.get("image_indices"), len(images))
            prompt = ""
            for label, image in zip(labels, images):
                prompt += f"{label}: " + DEFAULT_IMAGE_TOKEN + "\n"
                if isinstance(image, str):
                    if not os.path.exists(image):
                        raise FileNotFoundError(image)
                    image = Image.open(image).convert("RGB")
                elif isinstance(image, Image.Image):
                    image = image.convert("RGB")
                else:
                    raise TypeError(f"Unsupported image type: {type(image)!r}")
                imgs.append(image)
            prompt += messages["prompt"]
        else:
            prompt = messages["prompt"]
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        imgs = None if len(imgs) == 0 else imgs
        return prompt, imgs

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        if len(sequence) >= max_length:
            return sequence
        return torch.cat(
            [
                sequence,
                torch.full(
                    (max_length - len(sequence),),
                    padding_value,
                    device=sequence.device,
                    dtype=sequence.dtype,
                ),
            ]
        )

    def generate_output(self, messages):
        prompt, imgs = self.process_messages(messages)
        if imgs:
            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            imgs = process_images(imgs, self.image_processor, self.model.config)
        else:
            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            imgs = None

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        return outputs

    def generate_outputs(self, messages_list):
        outputs = []
        for messages in tqdm(messages_list):
            output = self.generate_output(messages)
            outputs.append(output)
        return outputs
