"""Image tiling for UFNLP/MedGPT-oss.

Taken from the official model card: InternVL-style dynamic tiling at
336x336 (CLIP ViT-L/14@336) with ImageNet normalization and an optional
thumbnail tile. Returns a (n_tiles, 3, 336, 336) float tensor.
"""

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IM_MEAN, IM_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _transform(sz=336):
    return T.Compose([
        T.Lambda(lambda i: i.convert("RGB") if i.mode != "RGB" else i),
        T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IM_MEAN, IM_STD),
    ])


def _closest_ratio(ar, ratios, w, h, sz):
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        d = abs(ar - r[0] / r[1])
        if d < best_diff:
            best, best_diff = r, d
        elif d == best_diff and w * h > 0.5 * sz * sz * r[0] * r[1]:
            best = r
    return best


def _tile(img, max_num=12, sz=336, thumb=True):
    w, h = img.size
    ratios = sorted(
        {(i, j) for n in range(1, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    r = _closest_ratio(w / h, ratios, w, h, sz)
    resized = img.resize((sz * r[0], sz * r[1]))
    tiles = [
        resized.crop(((i % r[0]) * sz, (i // r[0]) * sz,
                      (i % r[0] + 1) * sz, (i // r[0] + 1) * sz))
        for i in range(r[0] * r[1])
    ]
    if thumb and len(tiles) > 1:
        tiles.append(img.resize((sz, sz)))
    return tiles


def load_image(image_file, max_num=12, sz=336):
    """Accepts a file path or a PIL.Image; returns (n_tiles, 3, sz, sz)."""
    if isinstance(image_file, str):
        img = Image.open(image_file).convert("RGB")
    elif isinstance(image_file, Image.Image):
        img = image_file.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_file)}")
    tf = _transform(sz)
    return torch.stack([tf(t) for t in _tile(img, max_num, sz)])
