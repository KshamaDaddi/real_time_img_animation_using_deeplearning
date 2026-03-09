import os
import random
import json
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image


def _project_root() -> str:
    """Return the project root (two levels above this file)."""
    return os.path.dirname(os.path.dirname(__file__))


def _datasets_root() -> str:
    """Resolve datasets directory, defaulting to `<project>/datasets`."""
    return os.environ.get("DATASETS_DIR", os.path.join(_project_root(), "datasets"))


def _image_paths_in(dir_path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    try:
        return [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(exts)
        ]
    except Exception:
        return []


STYLE_ID_TO_DATASET_PATH = {
    "90s_anime": ["90s Styled Anime Images"],
    "ghibli_anime": ["ghibli-illustration-generated"],
    "anime_illustration": ["Randomly Styled Anime Images"],
    "japanese_anime": ["Randomly Styled Anime Images", "dataset/trainA"],
    "cyberpunk_anime": ["CyberVerse(Cyberpunk_ImagesDataset)"]
}


def _style_stats_path() -> str:
    return os.path.join(_project_root(), "models", "style_stats.json")


_STYLE_STATS_CACHE = None


def _load_style_stats():
    global _STYLE_STATS_CACHE
    if _STYLE_STATS_CACHE is not None:
        return _STYLE_STATS_CACHE
    try:
        with open(_style_stats_path(), "r", encoding="utf-8") as f:
            _STYLE_STATS_CACHE = json.load(f)
    except Exception:
        _STYLE_STATS_CACHE = {}
    return _STYLE_STATS_CACHE


def get_reference_paths(style_id: str) -> List[str]:
    """Return available reference image paths for a style.

    Prefers `datasets/<style_id>/` if present; falls back to `static/images/<style_id>.jpg`.
    """
    # Check for a mapped path first
    if style_id in STYLE_ID_TO_DATASET_PATH:
        mapped = STYLE_ID_TO_DATASET_PATH[style_id]
        base = _datasets_root()
        paths: List[str] = []
        if isinstance(mapped, str):
            ds_dir = os.path.join(base, mapped)
            if os.path.isdir(ds_dir):
                paths.extend(_image_paths_in(ds_dir))
        else:
            for m in mapped:
                ds_dir = os.path.join(base, m)
                if os.path.isdir(ds_dir):
                    paths.extend(_image_paths_in(ds_dir))
        if paths:
            return paths

    # Fallback to the original logic
    ds_dir = os.path.join(_datasets_root(), style_id)
    if os.path.isdir(ds_dir):
        paths = _image_paths_in(ds_dir)
        if paths:
            return paths

    # Fallback to single style preview image under static
    fallback = os.path.join(_project_root(), "static", "images", f"{style_id}.jpg")
    if os.path.isfile(fallback):
        return [fallback]

    return []


def pick_reference_image(style_id: str) -> Optional[Image.Image]:
    paths = get_reference_paths(style_id)
    if not paths:
        return None
    path = random.choice(paths)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def reinhard_stats_transfer(src_img: Image.Image, stats: dict, alpha: float = 0.85) -> Image.Image:
    """Apply Reinhard-style transfer using precomputed LAB stats.

    stats must contain keys: 'lab_mean' and 'lab_std'.
    """
    import numpy as np
    import cv2
    src_lab = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    eps = 1e-6
    r_mean = stats.get("lab_mean", [0.0, 0.0, 0.0])
    r_std = stats.get("lab_std", [1.0, 1.0, 1.0])
    for c in range(3):
        s_mean = float(np.mean(src_lab[..., c]))
        s_std = float(np.std(src_lab[..., c])) + eps
        src_lab[..., c] = (src_lab[..., c] - s_mean) * (float(r_std[c]) / s_std) + float(r_mean[c])
    transferred = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out_img = Image.fromarray(transferred)
    alpha = max(0.0, min(1.0, alpha))
    if alpha < 1.0:
        return Image.blend(src_img, out_img, alpha)
    return out_img


def reinhard_color_transfer(src_img: Image.Image, ref_img: Image.Image, alpha: float = 0.85) -> Image.Image:
    """Apply Reinhard color transfer from `ref_img` to `src_img` and blend by `alpha`.

    - Converts images to LAB to match mean/std of channels.
    - `alpha` controls blending: 1.0 = full transfer, 0.0 = original.
    """
    # Resize reference to source size for stable statistics
    ref_img = ref_img.resize(src_img.size, Image.Resampling.LANCZOS)

    src_lab = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2LAB).astype(np.float32)

    eps = 1e-6
    for c in range(3):
        s_mean = float(np.mean(src_lab[..., c]))
        s_std = float(np.std(src_lab[..., c])) + eps
        r_mean = float(np.mean(ref_lab[..., c]))
        r_std = float(np.std(ref_lab[..., c])) + eps
        src_lab[..., c] = (src_lab[..., c] - s_mean) * (r_std / s_std) + r_mean

    transferred = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out_img = Image.fromarray(transferred)

    # Blend with original to avoid overfitting to reference colors
    alpha = max(0.0, min(1.0, alpha))
    if alpha < 1.0:
        return Image.blend(src_img, out_img, alpha)
    return out_img


# Default style-specific blending strengths (can be overridden via env)
STYLE_ALPHA = {
    "ghibli_anime": 1.00,
    "fantasy_anime": 0.70,
    "cyberpunk_anime": 0.65,
    "aesthetic_anime": 0.65,
    "japanese_anime": 0.65,
    "anime_illustration": 0.65,
    "90s_anime": 0.50,
    "manga_anime": 0.00,
    "anime_sketch": 0.00,
    "3d_anime": 0.55,
    "cartoon_anime": 0.55,
}


def apply_dataset_reference(img: Image.Image, style_id: str) -> Image.Image:
    """Apply dataset-driven color transfer if reference images are available.

    Looks in `DATASETS_DIR/<style_id>/` or falls back to `static/images/<style_id>.jpg`.
    Blend strength may be tuned with `DATASET_ALPHA_<STYLE_ID>` env var.
    """
    if os.environ.get("ENABLE_DATASET_REFERENCE", "1") != "1":
        return img

    # Prefer learned style statistics if available
    stats_all = _load_style_stats()
    stats = stats_all.get(style_id)
    if stats:
        try:
            env_key = f"DATASET_ALPHA_{style_id.upper()}"
            alpha = float(os.environ.get(env_key, STYLE_ALPHA.get(style_id, 0.60)))
            return reinhard_stats_transfer(img, stats, alpha=alpha)
        except Exception as e:
            print(f"Stats transfer failed for {style_id}: {e}")

    # Fallback to per-image reference paths
    paths = get_reference_paths(style_id)
    if not paths:
        return img

    try:
        ref_img = pick_reference_image(style_id)
        if ref_img is None:
            return img
        env_key = f"DATASET_ALPHA_{style_id.upper()}"
        alpha = float(os.environ.get(env_key, STYLE_ALPHA.get(style_id, 0.60)))
        return reinhard_color_transfer(img, ref_img, alpha=alpha)
    except Exception as e:
        print(f"Dataset reference failed for {style_id}: {e}")
        return img


