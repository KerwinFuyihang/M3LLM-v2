import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _data_path(filename: str) -> str:
    return os.path.join(ROOT, "data", filename)


def _image_root() -> str:
    return os.environ.get("PMC_BENCH_IMAGE_ROOT") or os.path.join(ROOT, "images")


TASKS = {
    "puretext": {
        "json_path": _data_path("puretextQA_benchmark.json"),
        "image_root": _image_root(),
        "system_prompt": (
            "You are a medical expert. Answer the following text-only medical question "
            "accurately and concisely. Base your response on the information provided, "
            "use complete sentences, and do not overstate the available evidence."
        ),
    },
    "multi-choice": {
        "json_path": _data_path("multiplechoiceVQA_benchmark.json"),
        "image_root": _image_root(),
        "system_prompt": (
            "You are a medical expert. Select the best answer to the following medical "
            "multiple-choice question. Respond with the option letter only."
        ),
    },
    "single-subimageVQA": {
        "json_path": _data_path("single_subimageVQA_benchmark.json"),
        "image_root": _image_root(),
        "system_prompt": (
            "You are a medical expert. Answer the following question using the specified "
            "subimage within the provided multi-image input. Base your response on the "
            "relevant visual evidence and use concise, complete sentences."
        ),
    },
    "bboxVQA": {
        "json_path": _data_path("spatial_relation_benchmark.json"),
        "image_root": _image_root(),
        "system_prompt": (
            "You are a medical expert. Determine the relative spatial position of the two "
            "specified subimages from their locations within the provided compound figure. "
            "Respond with the requested spatial relation only."
        ),
    },
    "compoundVQA": {
        "json_path": _data_path("compound-imageVQA_benchmark.json"),
        "image_root": _image_root(),
        "system_prompt": (
            "You are a medical expert. Answer the following question using the provided "
            "compound medical figure. Base your response on the relevant visual evidence "
            "and use concise, complete sentences."
        ),
    },
    "multisubimageVQA": {
        "json_path": _data_path("multiplesubimageVQA_benchmark.json"),
        "image_root": _image_root(),
        "system_prompt": (
            "You are a medical expert. Answer the following question by integrating evidence "
            "across the specified subimages. Base your response on the relevant visual "
            "evidence and use concise, complete sentences."
        ),
    },
}
