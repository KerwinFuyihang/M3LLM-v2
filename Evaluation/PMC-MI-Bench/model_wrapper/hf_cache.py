from __future__ import annotations

import glob
import os


def get_hf_home() -> str | None:
    return os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")


def get_hf_cache_dir() -> str | None:
    hf_home = get_hf_home()
    if not hf_home:
        return None
    hub = os.path.join(hf_home, "hub")
    return hub if os.path.isdir(hub) else hf_home


def find_hub_file(repo_dir_name: str, filename: str) -> str | None:
    root = get_hf_cache_dir()
    if not root:
        return None
    pattern = os.path.join(root, repo_dir_name, "snapshots", "*", filename)
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None
