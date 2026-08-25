"""Prompt template for differential-diagnosis validity evaluation."""


def differential_diagnosis_prompt(diagnosis: str, number_of_images: int) -> str:
    image_lines = "\n".join(
        f"Image-{index}: <image>" for index in range(number_of_images)
    )
    return (
        "You are a medical AI assistant interpreting clinical dermatology images.\n"
        f"{image_lines}\n"
        f"Question: Is {diagnosis} a valid differential for this image?\n"
        "Answer with a single word: Yes or No."
    )
