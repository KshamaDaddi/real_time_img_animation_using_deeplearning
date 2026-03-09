import io
import os
import base64
from typing import Optional

from PIL import Image
from .dataset_reference import apply_dataset_reference
from .nn_refiner import refine_with_trained_cnn


_MODEL_CACHE = {}
_FACE2PAINT_CACHE = None


def _get_generator(pretrained_key: str):
    """Load AnimeGANv2 generator via torch.hub and cache it."""
    import torch
    if pretrained_key in _MODEL_CACHE:
        return _MODEL_CACHE[pretrained_key]
    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained=pretrained_key,
        progress=True,
        device="cpu",
    ).eval()
    _MODEL_CACHE[pretrained_key] = model
    return model


def _get_face2paint(size: int = 512):
    """Load face2paint helper via torch.hub and cache it."""
    global _FACE2PAINT_CACHE
    if _FACE2PAINT_CACHE is not None:
        return _FACE2PAINT_CACHE
    _FACE2PAINT_CACHE = __import__("torch").hub.load(
        "bryandlee/animegan2-pytorch:main", "face2paint", size=size, device="cpu"
    )
    return _FACE2PAINT_CACHE


STYLE_TO_MODEL = {
    # SD img2img preferred for better accuracy
    "90s_anime": None,
    "aesthetic_anime": None,
    "japanese_anime": None,
    "anime_illustration": None,
    "ghibli_anime": None,
    "fantasy_anime": None,
    "cyberpunk_anime": None,
    # Manga/sketch handled by post-process (no GAN)
    # 3D/cartoon handled by stylization functions
}

# Per-style SD parameter overrides to improve fidelity
STYLE_SD_PARAMS = {
    "ghibli_anime": {"strength": 0.50, "steps": 36, "guidance": 8.5},
    "90s_anime": {"strength": 0.60, "steps": 28, "guidance": 7.5},
    "anime_illustration": {"strength": 0.55, "steps": 30, "guidance": 7.2},
    "japanese_anime": {"strength": 0.55, "steps": 28, "guidance": 7.0},
    "fantasy_anime": {"strength": 0.55, "steps": 32, "guidance": 7.6},
    "cyberpunk_anime": {"strength": 0.50, "steps": 34, "guidance": 8.0},
}

def _try_load_local_lora(pipe, style_id: str):
    """Load a local LoRA from `models/lora/<style_id>/` if present.

    Picks the first `*.safetensors` or `*.bin` weight file.
    """
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        lora_dir = os.path.join(base, "models", "lora", style_id)
        if not os.path.isdir(lora_dir):
            return
        candidates = [
            f for f in os.listdir(lora_dir)
            if f.lower().endswith((".safetensors", ".bin"))
        ]
        if not candidates:
            return
        weight_name = candidates[0]
        pipe.load_lora_weights(lora_dir, weight_name=weight_name)
        try:
            pipe.fuse_lora()
        except Exception:
            pass
    except Exception as e:
        print(f"Local LoRA load failed for {style_id}: {e}")


# --- Stable Diffusion (SD 1.5) img2img pipeline for low VRAM ---
def _get_sd_img2img_pipe():
    """Create or retrieve a low‑VRAM SD 1.5 Img2Img pipeline.

    Implements memory‑saving features analogous to --lowvram and --xformers.
    """
    # Allow disabling SD via env or if import fails
    if os.environ.get("DISABLE_SD", "0") == "1":
        _MODEL_CACHE["sd_disabled"] = True
        raise RuntimeError("SD pipeline disabled via DISABLE_SD=1")

    import torch
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except Exception as e:
        print(f"Diffusers import failed, disabling SD: {e}")
        _MODEL_CACHE["sd_disabled"] = True
        raise

    cache_key = "sd15_img2img"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    base_model = os.environ.get("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Safety checker off for style transfer use‑case; keep if you prefer
    try:
        pipe.safety_checker = None
    except Exception:
        pass

    # Enable VRAM‑saving features
    try:
        pipe.enable_attention_slicing()  # slices attention to reduce peak memory
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        # xformers may be unavailable; proceed without it
        pass
    try:
        # CPU offloading is the closest analogue to --lowvram
        pipe.enable_model_cpu_offload()
    except Exception:
        # Fallback to moving to device directly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)

    # Optional: load a LoRA for stronger 90s style if provided via env
    lora_repo = os.environ.get("SD_LORA_90S_REPO")
    lora_weight = os.environ.get("SD_LORA_90S_WEIGHT")  # e.g., "pytorch_lora_weights.safetensors"
    if lora_repo:
        try:
            pipe.load_lora_weights(lora_repo, weight_name=lora_weight)
            # Fuse LoRA for a small speed gain after loading
            try:
                pipe.fuse_lora()
            except Exception:
                pass
        except Exception as e:
            print(f"LoRA load failed ({lora_repo}): {e}")

    _MODEL_CACHE[cache_key] = pipe
    return pipe


def _sd_prompts_for_90s_anime():
    """Prompt set tailored for 90s anime cel/VHS aesthetic."""
    prompt = (
        "90s anime style, cel shading, thick outlines, saturated colors,"
        " hand‑drawn look, retro aesthetic, studio quality"
    )
    negative = (
        "photo, realistic, 3d render, lowres, blurry, jpeg artifacts,"
        " deformed, extra limbs, oversaturated lighting"
    )
    return prompt, negative


def _sd_prompts_for_style(style_id: str):
    """Prompt/negative templates for various styles.

    Tuned for stronger color, neon lighting, and smoothness.
    """
    presets = {
        "cyberpunk_anime": (
            "cyberpunk anime, neon magenta and cyan lights, rim lighting,"
            " rainy street, bokeh, glowing signage, dark city, dramatic shadows,"
            " saturated RGB, high contrast, cinematic frame, ultra smooth",
            "lowres, washed out, desaturated, jpeg artifacts, dull lighting,"
            " oversharpened, extra limbs, deformed, text watermark",
        ),
        "ghibli_anime": (
            "Studio Ghibli style, hand-painted background, watercolor, soft lines,"
            " warm earthy tones, gentle lighting, whimsical atmosphere",
            "photo-realistic, harsh shadows",
        ),
        "aesthetic_anime": (
            "modern aesthetic anime, pastel tones, soft focus, dreamy glow,"
            " clean outlines, magazine cover composition",
            "blurry, low quality, noisy, overexposed, text",
        ),
        "japanese_anime": (
            "classic anime portrait, clean cel shading, balanced colors,"
            " precise outlines, studio quality",
            "photo, realistic render, distorted face, lowres, artifacts",
        ),
        "anime_illustration": (
            "anime illustration, clean lineart, bold outlines, cel shading,"
            " flat colors, detailed expressive eyes with glints, vibrant palette,"
            " dynamic pose, studio quality",
            "photo, realistic render, soft gradients, muddy colors, messy lines,"
            " blur, jpeg artifacts, watermark, text",
        ),
        "fantasy_anime": (
            "fantasy anime scene, lush colors, magical glow, volumetric light,"
            " cinematic composition, ethereal atmosphere",
            "flat lighting, dull colors, lowres, noisy",
        ),
    }
    return presets.get(style_id, _sd_prompts_for_90s_anime())


def _painterly_texture(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Apply a subtle, painterly texture to emulate hand-drawn feel."""
    if intensity == 0:
        return img
    import cv2
    import numpy as np
    arr = np.array(img)
    # Use a bilateral filter to smooth while preserving edges, then add subtle noise
    smoothed = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    noise = np.random.normal(0, 3 * intensity, arr.shape).astype(np.uint8)
    textured = cv2.add(smoothed, noise)
    return Image.fromarray(textured)


def _boost_greens(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Boost green tones to create lush, vibrant nature scenes."""
    if intensity == 0:
        return img
    from PIL import ImageEnhance
    # Split channels, enhance green, then merge back
    r, g, b = img.split()
    enhancer = ImageEnhance.Brightness(g)
    g = enhancer.enhance(1 + 0.2 * intensity)
    return Image.merge("RGB", (r, g, b))


def _warm_balance(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Add a warm, golden-hour tint for a nostalgic feel."""
    if intensity == 0:
        return img
    import numpy as np
    arr = np.array(img).astype(np.float32)
    # Add a subtle yellow/orange tint
    arr[:, :, 0] += 10 * intensity  # Red
    arr[:, :, 1] += 5 * intensity   # Green
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _apply_neon_glow(img: Image.Image) -> Image.Image:
    """Add subtle bloom/glow to bright regions to enhance neon effect."""
    import cv2
    import numpy as np
    arr = np.array(img)
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=6, sigmaY=6)
    # Blend original with blurred bright areas for glow
    out = cv2.addWeighted(arr, 1.0, blur, 0.35, 0)
    return Image.fromarray(out)


def _postprocess_sketch(img: Image.Image) -> Image.Image:
    import cv2
    import numpy as np
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    dodge = cv2.divide(gray, 255 - blur, scale=256)
    return Image.fromarray(dodge)


def _postprocess_cartoon(img: Image.Image) -> Image.Image:
    import cv2
    import numpy as np
    img_np = np.array(img)
    color = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, edges)
    return Image.fromarray(cartoon)


def _postprocess_cyberpunk(img: Image.Image) -> Image.Image:
    """Neon color grading for cyberpunk vibe."""
    import numpy as np
    arr = np.array(img).astype(np.float32)
    # Boost magenta/cyan channels
    arr[..., 0] = np.clip(arr[..., 0] * 0.9 + 20, 0, 255)  # R
    arr[..., 1] = np.clip(arr[..., 1] * 0.8 + 10, 0, 255)  # G
    arr[..., 2] = np.clip(arr[..., 2] * 1.15 + 25, 0, 255)  # B
    return Image.fromarray(arr.astype(np.uint8))


def _postprocess_manga(img: Image.Image) -> Image.Image:
    import cv2
    import numpy as np
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.bitwise_not(edges)
    return Image.fromarray(edges)


   # --- Style fingerprint utilities to ensure distinct visual traits ---
# Get fingerprint intensity from environment variables with defaults
def _get_fingerprint_intensity(style: str, effect: str, default: float) -> float:
    """Get intensity value for a specific style's effect from environment variables."""
    env_var = f"FINGERPRINT_{style.upper()}_{effect.upper()}"
    try:
        value = float(os.environ.get(env_var, default))
        return max(0.0, min(1.0, value))  # Clamp between 0-1
    except (ValueError, TypeError):
        return default


def _add_grain(arr, amount: float = 0.05):
    import numpy as np
    noise = np.random.randn(*arr.shape).astype(np.float32) * (255 * amount)
    out = np.clip(arr.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def _add_scanlines(arr, intensity: float = 0.12, spacing: int = 2):
    import numpy as np
    h, w, c = arr.shape
    mask = np.ones((h, 1), dtype=np.float32)
    mask[::spacing] = 1.0 - intensity
    mask = np.repeat(mask, w, axis=1)
    mask = np.expand_dims(mask, axis=-1)
    out = np.clip(arr.astype(np.float32) * mask, 0, 255)
    return out.astype(np.uint8)


def _chromatic_aberration(img: Image.Image, shift: int = 2) -> Image.Image:
    import numpy as np
    import cv2
    arr = np.array(img)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    M_right = np.float32([[1, 0, shift], [0, 1, 0]])
    M_left = np.float32([[1, 0, -shift], [0, 1, 0]])
    r_shift = cv2.warpAffine(r, M_right, (arr.shape[1], arr.shape[0]))
    b_shift = cv2.warpAffine(b, M_left, (arr.shape[1], arr.shape[0]))
    out = np.stack([r_shift, g, b_shift], axis=-1)
    return Image.fromarray(out)


def _warm_earthy(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    import numpy as np
    arr = np.array(img).astype(np.float32)
    # Warm tone: boost R slightly, reduce B, mild desaturation
    r_boost = 1.06 * intensity
    b_reduce = 0.94 + (0.06 * (1.0 - intensity))
    r_add = 8 * intensity
    b_sub = 4 * intensity
    
    arr[..., 0] = np.clip(arr[..., 0] * r_boost + r_add, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * b_reduce - b_sub, 0, 255)
    # Mild contrast lift for softness
    contrast = 0.92 + (0.08 * (1.0 - intensity))
    arr = np.clip((arr - 127.5) * contrast + 127.5, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _pastel_grade(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    import numpy as np
    arr = np.array(img).astype(np.float32)
    # Reduce saturation by moving channels towards mean
    sat_factor = 0.75 + (0.25 * (1.0 - intensity))
    mean = arr.mean(axis=2, keepdims=True)
    arr = mean + (arr - mean) * sat_factor
    # Gentle pink/blue tint
    r_add = 6 * intensity
    b_add = 10 * intensity
    arr[..., 0] = np.clip(arr[..., 0] + r_add, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] + b_add, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _cel_edges(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    import cv2
    import numpy as np
    arr = np.array(img)
    # Smooth color then overlay edges
    color = cv2.bilateralFilter(arr, d=7, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Invert edges for line art overlay
    edges_inv = 255 - edges
    edge_weight = 0.35 * intensity
    out = cv2.addWeighted(color, 1.0, edges_inv, edge_weight, 0)
    return Image.fromarray(out)


def _unsharp(
    img: Image.Image,
    radius: int = 2,
    percent: int = 140,
    intensity: float = 1.0,
    amount: float | None = None,
) -> Image.Image:
    from PIL import ImageFilter
    if amount is not None:
        try:
            intensity = float(amount)
        except Exception:
            pass
    adjusted_percent = int(percent * max(0.0, float(intensity)))
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=adjusted_percent, threshold=3))


def _fantasy_glow(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    # Stronger glow with slight mystical tint
    import numpy as np
    from PIL import ImageFilter
    blur_radius = 6 * intensity
    glow = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(img).astype(np.float32)
    glow_arr = np.array(glow).astype(np.float32)
    
    base_weight = 0.85 + (0.15 * (1.0 - intensity))
    glow_weight = 0.35 * intensity
    
    out = np.clip(arr * base_weight + glow_arr * glow_weight, 0, 255)
    # Cyan/gold grade
    r_boost = 1.03 * intensity
    b_boost = 1.05 * intensity
    r_add = 6 * intensity
    b_add = 8 * intensity
    
    out[..., 0] = np.clip(out[..., 0] * r_boost + r_add, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * b_boost + b_add, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _boost_saturation(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Increase color saturation in a controlled way.
    intensity in [0,1] maps to factor ~[1.0, 1.6].
    """
    from PIL import ImageEnhance
    factor = 1.0 + 0.6 * max(0.0, min(1.0, intensity))
    return ImageEnhance.Color(img).enhance(factor)


def _bold_outline(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Emphasize outlines by darkening detected edges and slightly thickening them."""
    import numpy as np
    from PIL import ImageFilter, ImageOps
    edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
    edges = ImageOps.invert(edges)
    # Thicken lines a bit
    k = 3 if intensity < 0.6 else 5
    edges = edges.filter(ImageFilter.MaxFilter(k))
    edge_arr = np.array(edges).astype(np.float32) / 255.0
    edge_weight = 0.45 * max(0.0, min(1.0, intensity))
    base = np.array(img).astype(np.float32)
    out = np.clip(base * (1.0 - edge_weight * edge_arr[..., None]), 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _cel_shade(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Approximate cel shading via smoothing + posterization."""
    import numpy as np
    import cv2
    arr = np.array(img)
    # Edge-preserving smoothing
    smooth = cv2.bilateralFilter(arr, d=7, sigmaColor=80, sigmaSpace=80)
    pil_smooth = Image.fromarray(smooth)
    # Reduce color buckets (posterize)
    from PIL import ImageOps
    bits = max(3, int(8 - 4 * max(0.0, min(1.0, intensity))))
    cel = ImageOps.posterize(pil_smooth, bits)
    return cel


def _vignette(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Darken corners for cinematic focus using a radial mask."""
    import numpy as np
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = max(cy, 1.0), max(cx, 1.0)
    r = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    # Build mask: 1 at center, down to ~0.7 at edges depending on intensity
    vig_strength = max(0.0, min(1.0, intensity))
    mask = 1.0 - vig_strength * np.clip(r ** 1.2, 0.0, 1.0)
    mask = np.clip(mask, 0.7, 1.0)
    out = np.clip(arr * mask[..., None], 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _scanlines_image(img: Image.Image, intensity: float = 0.12, spacing: int = 2) -> Image.Image:
    """Apply horizontal scanlines to emulate display artifacts."""
    import numpy as np
    arr = np.array(img)
    out = _add_scanlines(arr, intensity=intensity, spacing=spacing)
    return Image.fromarray(out)


def _film_grain_image(img: Image.Image, intensity: float = 0.2) -> Image.Image:
    """Add film grain; intensity controls noise amount."""
    import numpy as np
    amount = max(0.0, min(0.4, 0.05 + 0.25 * intensity))
    arr = _add_grain(np.array(img), amount=amount)
    return Image.fromarray(arr)


def _neon_color_grade(img: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Strong magenta/cyan neon grading with intensity control."""
    import numpy as np
    arr = np.array(img).astype(np.float32)
    i = max(0.0, min(1.0, intensity))
    # Scale channels: push towards neon palette
    arr[..., 0] = np.clip(arr[..., 0] * (0.9 + 0.1 * i) + (10 + 25 * i), 0, 255)  # R
    arr[..., 1] = np.clip(arr[..., 1] * (0.8 + 0.1 * i) + (5 + 12 * i), 0, 255)   # G
    arr[..., 2] = np.clip(arr[..., 2] * (1.05 + 0.2 * i) + (10 + 25 * i), 0, 255) # B
    # Slight contrast for gritty feel
    arr = np.clip((arr - 127.5) * (1.02 + 0.12 * i) + 127.5, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _rain_streaks(img: Image.Image, intensity: float = 0.4) -> Image.Image:
    """Overlay diagonal rain streaks with motion blur."""
    from PIL import ImageDraw, ImageFilter
    import random
    i = max(0.0, min(1.0, intensity))
    base = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    count = max(10, int(80 * i))
    alpha = int(40 + 140 * i)
    for _ in range(count):
        x = random.randint(0, w)
        y = random.randint(0, h)
        length = random.randint(int(0.05 * h), int(0.18 * h))
        # Diagonal down-right
        draw.line((x, y, x + int(length * 0.5), y + length), fill=(220, 220, 255, alpha), width=1)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1 + 2 * i))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _hologram_overlay(img: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Overlay subtle neon HUD frames near the edges.
    Replaces filled blue boxes with thin, low‑alpha outlines and soft glow
    to avoid distracting rectangles while keeping a cyberpunk feel.
    """
    from PIL import ImageDraw, ImageFilter
    import random

    i = max(0.0, min(1.0, float(intensity)))
    if i < 0.05:
        return img

    base = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size

    # Fewer panels, positioned near frame edges
    boards = max(1, int(1 + 3 * i))
    for _ in range(boards):
        bw = random.randint(int(0.14 * w), int(0.26 * w))
        bh = random.randint(int(0.10 * h), int(0.20 * h))
        # Bias to left/right edges
        if random.random() < 0.5:
            bx = random.randint(0, int(0.18 * w))
        else:
            bx = random.randint(max(0, int(0.82 * w) - bw), max(0, w - bw))
        by = random.randint(int(0.08 * h), max(int(0.65 * h), int(0.08 * h)))

        # Pick between cyan and magenta neon with low fill alpha
        if random.random() < 0.5:
            border = (40, 220, 255, int(90 + 90 * i))
            fill = (40, 220, 255, int(20 + 50 * i))
        else:
            border = (255, 60, 180, int(90 + 90 * i))
            fill = (255, 60, 180, int(18 + 45 * i))

        # Thin outline and faint fill
        draw.rectangle((bx, by, bx + bw, by + bh), outline=border, width=max(1, int(2 * i)))
        draw.rectangle((bx + 1, by + 1, bx + bw - 1, by + bh - 1), outline=fill, width=1)

        # Sparse internal scanlines
        line_alpha = int(25 + 55 * i)
        for ly in range(by + 4, by + bh - 4, 6):
            draw.line((bx + 4, ly, bx + bw - 4, ly), fill=(255, 255, 255, line_alpha))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.8 + 1.2 * i))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _rim_light(img: Image.Image, intensity: float = 0.6) -> Image.Image:
    """Add cyan rim lighting along edges to enhance silhouettes."""
    from PIL import ImageFilter
    import numpy as np
    i = max(0.0, min(1.0, intensity))
    edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
    edge_arr = np.array(edges).astype(np.float32) / 255.0
    base = np.array(img).astype(np.float32)
    # Cyan tint based on edge strength
    tint = np.stack([
        40.0 * i * edge_arr,
        120.0 * i * edge_arr,
        180.0 * i * edge_arr,
    ], axis=-1)
    out = np.clip(base + tint, 0, 255)
    return Image.fromarray(out.astype(np.uint8))

def _chibi_kawaii_effect(img: Image.Image, intensity: float = 0.6) -> Image.Image:
    """Apply a 'kawaii' aesthetic: subtly increase eye size and add soft blush.
    Uses a face detector to locate features. If no face is found, returns original.
    Intensity controls the strength of the distortion and blush.
    """
    import cv2
    import numpy as np
    from PIL import ImageDraw

    # Use a simple Haar cascade for face detection (less overhead than dlib)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    except Exception:
        return img # Cascade files not found

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return img # No faces detected

    output_img = img.copy()
    draw = ImageDraw.Draw(output_img, 'RGBA')

    for (x, y, w, h) in faces:
        # --- 1. Exaggerate Eyes ---
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Simple eye enlargement by warping
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[:2]
            
            # Center of eyes
            center1 = (x + ex1 + ew1//2, y + ey1 + eh1//2)
            center2 = (x + ex2 + ew2//2, y + ey2 + eh2//2)
            
            # Warp factor based on intensity
            scale = 1.0 + 0.2 * intensity
            
            # Create a meshgrid and apply radial distortion around eye centers
            H, W, _ = arr.shape
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            
            for cx, cy in [center1, center2]:
                dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                # Create a localized mask for the warp
                mask = np.exp(-(dist**2) / (2 * (max(ew1, eh1) * 1.5)**2))
                
                dx = (xx - cx) * (scale - 1) * mask
                dy = (yy - cy) * (scale - 1) * mask

                map_x = (xx - dx).astype(np.float32)
                map_y = (yy - dy).astype(np.float32)

                # Apply warp to a copy to avoid compounding distortions
                arr = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            
            output_img = Image.fromarray(arr)
            draw = ImageDraw.Draw(output_img, 'RGBA')


        # --- 2. Add Blush ---
        blush_radius = int(w * 0.15 * intensity)
        if blush_radius > 0:
            # Position blush under the eyes, towards the cheeks
            blush_y = y + h // 2
            cheek_left_x = x + w // 4
            cheek_right_x = x + 3 * w // 4
            
            blush_alpha = int(80 * intensity)
            blush_color = (255, 105, 180, blush_alpha) # Pink

            # Create a transparent overlay for the blush
            blush_overlay = Image.new('RGBA', output_img.size, (255, 255, 255, 0))
            blush_draw = ImageDraw.Draw(blush_overlay)

            # Draw two oval blushes
            blush_draw.ellipse(
                (cheek_left_x - blush_radius, blush_y - blush_radius//2, cheek_left_x + blush_radius, blush_y + blush_radius//2),
                fill=blush_color
            )
            blush_draw.ellipse(
                (cheek_right_x - blush_radius, blush_y - blush_radius//2, cheek_right_x + blush_radius, blush_y + blush_radius//2),
                fill=blush_color
            )
            
            # Alpha composite the blush onto the main image
            output_img = Image.alpha_composite(output_img.convert("RGBA"), blush_overlay)

    return output_img.convert("RGB")


def _magic_runes_overlay(img: Image.Image, intensity: float = 0.4) -> Image.Image:
    """Overlay faint glowing arcane runes to suggest magical contracts/power.
    Draw simple glyphs with blur and additive blend. Intensity controls opacity.
    """
    from PIL import ImageDraw, ImageFilter
    import random

    base = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    glyph_count = max(2, int(6 * max(0.0, min(1.0, intensity))))
    alpha = int(60 + 140 * intensity)

    for _ in range(glyph_count):
        # Random positions and sizes
        gx = random.randint(int(0.1 * w), int(0.9 * w))
        gy = random.randint(int(0.1 * h), int(0.9 * h))
        r = random.randint(int(0.02 * min(w, h)), int(0.08 * min(w, h)))
        color = (120, 255, 240, alpha)  # teal glow
        # Circles + lines to mimic sigils
        draw.ellipse((gx - r, gy - r, gx + r, gy + r), outline=color, width=2)
        draw.line((gx - r, gy, gx + r, gy), fill=color, width=2)
        draw.line((gx, gy - r, gx, gy + r), fill=color, width=2)

    # Blur for glow
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3 + 3 * intensity))

    # Additive blend via alpha composite over a brightened copy
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def _energy_aura(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """Add a radial energy aura to suggest transformations/power-ups."""
    import numpy as np

    base = np.array(img).astype(np.float32)
    h, w = base.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((y - cy) / (h / 2.0)) ** 2 + ((x - cx) / (w / 2.0)) ** 2)
    aura = np.clip(1.0 - r, 0.0, 1.0)
    aura = (aura ** 2) * (0.45 * max(0.0, min(1.0, intensity)))
    # Cyan-magenta aura tint
    tint = np.stack([
        0.6 * aura,            # R
        0.2 * aura,            # G
        1.0 * aura             # B
    ], axis=-1) * 255.0
    out = np.clip(base + tint, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _atmospheric_depth(img: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Simulate atmospheric perspective (misty depth) with a vertical gradient haze."""
    import numpy as np

    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    y = np.linspace(0, 1, h).reshape(h, 1)
    haze = (y ** 1.4) * (0.35 * max(0.0, min(1.0, intensity)))
    # Cool haze tint
    haze_rgb = np.concatenate([
        (180.0 * haze),
        (210.0 * haze),
        (240.0 * haze)
    ], axis=1).reshape(h, 1, 3)
    out = np.clip(arr + haze_rgb, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _speed_lines_overlay(img: Image.Image, intensity: float = 0.3) -> Image.Image:
    """Add diagonal speed lines to suggest motion. Intensity controls density/opacity."""
    from PIL import ImageDraw
    intensity = max(0.0, min(1.0, float(intensity)))
    if intensity < 0.01:
        return img

    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    spacing = max(8, int(24 - 16 * intensity))
    alpha = int(40 + 140 * intensity)
    color = (255, 255, 255, alpha)

    # Diagonal lines left-bottom to right-top
    for i in range(-h, w, spacing):
        x1, y1 = i, h
        x2, y2 = i + h, 0
        draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, int(2 * intensity)))

    # Soft blur via scale trick
    small = overlay.resize((max(1, w // 2), max(1, h // 2)), resample=Image.Resampling.BILINEAR)
    blur = small.resize((w, h), resample=Image.Resampling.BILINEAR)
    base = img.convert("RGBA")
    return Image.alpha_composite(base, blur).convert("RGB")


def _eye_highlights(img: Image.Image, intensity: float = 0.4) -> Image.Image:
    """Add small reflective glints to detected eyes for expressiveness."""
    try:
        import cv2
        import numpy as np
        from PIL import ImageDraw
        i = max(0.0, min(1.0, float(intensity)))
        if i < 0.05:
            return img

        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for (ex, ey, ew, eh) in eyes[:4]:
            r1 = max(2, int(ew * 0.08 * (0.7 + i)))
            r2 = max(1, int(ew * 0.05 * (0.6 + i)))
            alpha = int(160 + 90 * i)
            cx1, cy1 = ex + int(ew * 0.35), ey + int(eh * 0.35)
            cx2, cy2 = ex + int(ew * 0.5), ey + int(eh * 0.45)
            draw.ellipse([(cx1 - r1, cy1 - r1), (cx1 + r1, cy1 + r1)], fill=(255, 255, 255, alpha))
            draw.ellipse([(cx2 - r2, cy2 - r2), (cx2 + r2, cy2 + r2)], fill=(255, 255, 255, int(alpha * 0.8)))

        base = img.convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")
    except Exception:
        return img


def _enforce_anime_look(img: Image.Image, strength: float = 0.6) -> Image.Image:
    """Universal pass to push outputs toward bold anime:
    - Stronger outlines
    - Cel shading
    - Mild color boost
    - Light unsharp for crisp edges
    """
    s = max(0.0, min(1.0, float(strength)))
    img = _bold_outline(img, intensity=0.65 + 0.25 * s)
    img = _cel_shade(img, intensity=0.55 + 0.35 * s)
    img = _boost_saturation(img, intensity=0.20 + 0.30 * s)
    img = _unsharp(img, amount=0.25 + 0.35 * s)
    return img


def _apply_style_fingerprint(img: Image.Image, style_id: str) -> Image.Image:
    """Apply a distinct visual trait based on style_id.

    This is where non-GAN/non-SD styles are created, and where we can
    add a final opinionated touch to other styles.
    """
    if style_id == "cyberpunk_anime":
         # High Tech, Low Life: neon grade + grime + display artifacts
         neon = _get_fingerprint_intensity("cyberpunk", "neon", 0.9)
         grain = _get_fingerprint_intensity("cyberpunk", "grain", 0.05)
         scan = _get_fingerprint_intensity("cyberpunk", "scanlines", 0.0)
         chroma = _get_fingerprint_intensity("cyberpunk", "chroma", 0.25)
         rain = _get_fingerprint_intensity("cyberpunk", "rain", 0.25)
         holo = _get_fingerprint_intensity("cyberpunk", "holograms", 0.2)
         vign = _get_fingerprint_intensity("cyberpunk", "vignette", 0.25)
         rim = _get_fingerprint_intensity("cyberpunk", "rimlight", 0.6)

         img = _neon_color_grade(img, intensity=neon)
         if grain > 0.01:
             img = _film_grain_image(img, intensity=grain)
         if scan > 0.01:
             img = _scanlines_image(img, intensity=scan)
         if chroma > 0.01:
             shift = max(1, int(2 + 3 * chroma))
             img = _chromatic_aberration(img, shift=shift)
         if rain > 0.01:
             img = _rain_streaks(img, intensity=rain)
         if holo > 0.01:
             img = _hologram_overlay(img, intensity=holo)
         img = _apply_neon_glow(img)
         img = _rim_light(img, intensity=rim)
         img = _vignette(img, intensity=vign)
         return img

    if style_id == "anime_sketch":
        return _postprocess_sketch(img)

    if style_id == "3d_anime":
        return _apply_3d_effect(img)

    if style_id == "cartoon_anime":
        return _apply_cartoon_effect(img)

    if style_id == "anime_illustration":
        # Clean line art, cel-shading, vibrant color; optional motion/background cues
        lines = float(os.environ.get("FINGERPRINT_ILLUSTRATION_LINES", 0.75))
        cel = float(os.environ.get("FINGERPRINT_ILLUSTRATION_CEL_SHADE", 0.70))
        sat = float(os.environ.get("FINGERPRINT_ILLUSTRATION_SATURATION", 0.35))
        glints = float(os.environ.get("FINGERPRINT_ILLUSTRATION_EYE_GLINTS", 0.45))
        speed = float(os.environ.get("FINGERPRINT_ILLUSTRATION_SPEED_LINES", 0.0))
        painterly = float(os.environ.get("FINGERPRINT_ILLUSTRATION_PAINTERLY", 0.25))
        crisp = float(os.environ.get("FINGERPRINT_ILLUSTRATION_CRISP", 0.5))

        # Sequence: lines → cel → saturation → glints → speed lines → painterly → crisp
        img = _bold_outline(img, intensity=lines)
        img = _cel_shade(img, intensity=cel)
        img = _boost_saturation(img, intensity=sat)
        if glints > 0.01:
            img = _eye_highlights(img, intensity=glints)
        if speed > 0.01:
            img = _speed_lines_overlay(img, intensity=speed)
        if painterly > 0.01:
            img = _painterly_texture(img, intensity=painterly)
        img = _unsharp(img, amount=crisp)
        return img
    if style_id == "fantasy_anime":
        # Controlled by env vars for intensity
        vibrancy = float(os.environ.get("FINGERPRINT_FANTASY_VIBRANT", 0.7))
        cel_shade = float(os.environ.get("FINGERPRINT_FANTASY_CEL_SHADE", 0.6))
        bold_lines = float(os.environ.get("FINGERPRINT_FANTASY_BOLD_LINES", 0.75))
        glow = float(os.environ.get("FINGERPRINT_FANTASY_GLOW", 0.9))
        vignette = float(os.environ.get("FINGERPRINT_FANTASY_VIGNETTE", 0.4))
        runes = float(os.environ.get("FINGERPRINT_FANTASY_RUNES", 0.4))
        aura = float(os.environ.get("FINGERPRINT_FANTASY_AURA", 0.5))
        depth = float(os.environ.get("FINGERPRINT_FANTASY_DEPTH", 0.3))

        # Sequence: lines/cel → color → depth → aura/runes → glow → vignette
        img = _bold_outline(img, intensity=bold_lines)
        img = _cel_shade(img, intensity=cel_shade)
        img = _boost_saturation(img, intensity=vibrancy)
        img = _atmospheric_depth(img, intensity=depth)
        if aura > 0.01:
            img = _energy_aura(img, intensity=aura)
        if runes > 0.01:
            img = _magic_runes_overlay(img, intensity=runes)
        img = _fantasy_glow(img, intensity=glow)
        img = _vignette(img, intensity=vignette)
        return img

    if style_id == "ghibli_anime":
        # Controlled by env vars for intensity
        painterly = float(os.environ.get("FINGERPRINT_GHIBLI_PAINTERLY", 0.6))
        greens = float(os.environ.get("FINGERPRINT_GHIBLI_GREEN_BOOST", os.environ.get("FINGERPRINT_GHIBLI_GREENS", 0.65)))
        warmth = float(os.environ.get("FINGERPRINT_GHIBLI_WARM_BALANCE", os.environ.get("FINGERPRINT_GHIBLI_WARMTH", 0.5)))

        img = _painterly_texture(img, intensity=painterly)
        img = _boost_greens(img, intensity=greens)
        img = _warm_balance(img, intensity=warmth)
        return img

    # Default: no fingerprint
    return img


# --- Accuracy improvements for high-quality inputs ---
def _sd_target_resize(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    """Resize input for SD, preserving aspect ratio and aligning to 64px grid.

    Returns the resized image and the original size.
    Uses `SD_MAX_RES` env (default: 768 if CUDA, else 512).
    """
    import math
    import torch
    max_res_env = os.environ.get("SD_MAX_RES")
    if max_res_env is not None:
        try:
            max_dim = max(256, int(max_res_env))
        except Exception:
            max_dim = 768 if torch.cuda.is_available() else 512
    else:
        max_dim = 768 if torch.cuda.is_available() else 512

    orig_w, orig_h = img.size
    orig_max = max(orig_w, orig_h)
    scale = min(1.0, max_dim / float(orig_max))
    new_w = max(256, int(math.floor((orig_w * scale) / 64.0) * 64))
    new_h = max(256, int(math.floor((orig_h * scale) / 64.0) * 64))
    if new_w < 256 or new_h < 256:
        new_w, new_h = 256, 256
    img_resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    return img_resized, (orig_w, orig_h)


def _adapt_params_for_size(orig_size: tuple[int, int], base_strength: float, base_steps: int) -> tuple[float, int]:
    """Adjust SD strength and steps based on original image size to retain detail."""
    orig_w, orig_h = orig_size
    orig_max = max(orig_w, orig_h)
    strength = base_strength
    steps = base_steps
    # For larger inputs, reduce strength slightly to keep more original detail,
    # and increase steps moderately for better quality.
    if orig_max >= 1024:
        strength = max(0.40, base_strength - 0.12)
        steps = base_steps + 8
    elif orig_max >= 768:
        strength = max(0.45, base_strength - 0.08)
        steps = base_steps + 4
    return strength, steps


def generate_anime_image(image_bytes: bytes, style_id: str) -> Optional[str]:
    """Generate anime-styled image locally using SD (for 90s) or AnimeGAN/post-process.

    Returns base64 PNG string or None on failure.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Prefer SD img2img for 90s anime with strict 512x512
        if style_id == "90s_anime":
            import torch

            # Adaptive resize (keeps aspect and aligns to 64px grid)
            img_sd, orig_size = _sd_target_resize(img)
            try:
                # If SD is disabled due to import error or env, go to fallback
                if _MODEL_CACHE.get("sd_disabled"):
                    raise RuntimeError("SD disabled; using fallback")
                pipe = _get_sd_img2img_pipe()
                prompt, negative = _sd_prompts_for_90s_anime()

                # Strength/steps adapt to original image size for detail retention
                base_strength = float(os.environ.get("SD_STRENGTH", "0.6"))
                base_steps = int(os.environ.get("SD_STEPS", "28"))
                strength, steps = _adapt_params_for_size(orig_size, base_strength, base_steps)
                guidance_scale = float(os.environ.get("SD_GUIDANCE", "7.5"))

                if torch.cuda.is_available():
                    from torch import autocast
                    with autocast("cuda"):
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative,
                            image=img_sd,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            num_inference_steps=steps,
                        )
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative,
                        image=img_sd,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=steps,
                    )
            except Exception as e:
                print(f"SD generation failed, falling back to AnimeGAN: {e}")
                # Fallback to AnimeGAN paprika model approximation
                model = _get_generator("paprika")
                import torch, numpy as np
                x = torch.from_numpy(np.array(img_sd)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    y = model(x).clamp(0, 1)
                out = (y.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
                img_out = Image.fromarray(out)
                img_out = _enforce_anime_look(img_out, strength=0.7)
            else:
                img_out = result.images[0]
            # Upscale back to original size if needed
            if img_out.size != orig_size:
                img_out = img_out.resize(orig_size, resample=Image.Resampling.LANCZOS)
            # Ensure distinct 90s look
            img_out = _apply_style_fingerprint(img_out, style_id)

        # Purely dataset-driven path for Ghibli
        elif style_id == "ghibli_anime":
            img_out = img.copy()

        # SD img2img for other high-accuracy styles
        elif style_id in STYLE_TO_MODEL and STYLE_TO_MODEL[style_id] is None and style_id != "90s_anime":
            import torch
            img_sd, orig_size = _sd_target_resize(img)
            try:
                if _MODEL_CACHE.get("sd_disabled"):
                    raise RuntimeError("SD disabled; using fallback")
                pipe = _get_sd_img2img_pipe()

                prompt, negative = _sd_prompts_for_style(style_id)
                params = STYLE_SD_PARAMS.get(style_id, {})
                base_strength = float(os.environ.get("SD_STRENGTH", str(params.get("strength", 0.55))))
                base_steps = int(os.environ.get("SD_STEPS", str(params.get("steps", 26))))
                strength, steps = _adapt_params_for_size(orig_size, base_strength, base_steps)
                guidance_scale = float(os.environ.get("SD_GUIDANCE", str(params.get("guidance", 7.0))))

                # Per-style LoRA via env (optional)
                repo_env = f"SD_LORA_{style_id.upper()}_REPO"
                weight_env = f"SD_LORA_{style_id.upper()}_WEIGHT"
                lora_repo = os.environ.get(repo_env)
                lora_weight = os.environ.get(weight_env)
                if lora_repo:
                    try:
                        # unload any previous LoRA to avoid compounding
                        try:
                            pipe.unload_lora_weights()
                        except Exception:
                            pass
                        pipe.load_lora_weights(lora_repo, weight_name=lora_weight)
                        try:
                            pipe.fuse_lora()
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"LoRA load failed for {style_id}: {e}")
                else:
                    # Try loading local LoRA if available
                    try:
                        pipe.unload_lora_weights()
                    except Exception:
                        pass
                    _try_load_local_lora(pipe, style_id)

                if torch.cuda.is_available():
                    from torch import autocast
                    with autocast("cuda"):
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative,
                            image=img_sd,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            num_inference_steps=steps,
                        )
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative,
                        image=img_sd,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=steps,
                    )
            except Exception as e:
                print(f"SD generation failed for {style_id}, falling back: {e}")
                # Fallback to AnimeGAN v2 portrait
                model = _get_generator("face_paint_512_v2")
                import torch, numpy as np
                x = torch.from_numpy(np.array(img_sd)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    y = model(x).clamp(0, 1)
                out = (y.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
                img_out = Image.fromarray(out)
                img_out = _enforce_anime_look(img_out, strength=0.6)
            else:
                img_out = result.images[0]
            # Upscale back to original size if needed
            if img_out.size != orig_size:
                img_out = img_out.resize(orig_size, resample=Image.Resampling.LANCZOS)
            # Apply per-style fingerprint for distinctiveness
            img_out = _apply_style_fingerprint(img_out, style_id)

        # Styles handled by AnimeGAN
        elif style_id in STYLE_TO_MODEL and STYLE_TO_MODEL[style_id] is not None:
            model_key = STYLE_TO_MODEL[style_id]
            model = _get_generator(model_key)
            # Prefer simple forward pass for general images
            import torch
            import numpy as np
            x = torch.from_numpy(np.array(img)).float()
            x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                y = model(x).clamp(0, 1)
            out = (y.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
            img_out = Image.fromarray(out)
        else:
            # Non-GAN styles via post-processing
            if style_id == "anime_sketch":
                img_out = _postprocess_sketch(img)
            elif style_id == "cartoon_anime":
                img_out = _postprocess_cartoon(img)
                img_out = _enforce_anime_look(img_out, strength=0.6)
            elif style_id == "cyberpunk_anime":
                # Use filter-based pipeline instead of AnimeGAN to avoid painterly smear
                # when SD is unavailable on Windows. Preserve scene detail and add cyberpunk traits.
                img_out = img.copy()
                # Optional light cartoon smoothing to clean noise without blurring edges
                try:
                    img_out = _postprocess_cartoon(img_out)
                except Exception:
                    pass
                img_out = _apply_style_fingerprint(img_out, style_id)
            elif style_id == "manga_anime":
                img_out = _postprocess_manga(img)
            elif style_id == "3d_anime":
                # Use cartoon stylization as proxy for 3D cell-shading
                img_out = _postprocess_cartoon(img)
            else:
                # Default to anime portrait
                model = _get_generator("face_paint_512_v2")
                import torch, numpy as np
                x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    y = model(x).clamp(0, 1)
                out = (y.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
                img_out = Image.fromarray(out)

        # Apply fingerprints except for dataset-only Ghibli
        if style_id in STYLE_TO_MODEL and style_id != "ghibli_anime":
            img_out = _apply_style_fingerprint(img_out, style_id)

        # Universal strict anime enforcement (optional), skipped for Ghibli
        try:
            if os.environ.get("STRICT_ANIME_MODE", "0") == "1" and style_id != "ghibli_anime":
                s = float(os.environ.get("STRICT_ANIME_STRENGTH", "0.6"))
                img_out = _enforce_anime_look(img_out, strength=s)
        except Exception as e:
            print(f"Strict anime mode failed: {e}")

        # Optional neural refiner (CNN) using backprop per-image, skipped for Ghibli
        try:
            if os.environ.get("ENABLE_CNN_REFINER", "0") == "1" and style_id != "ghibli_anime":
                from .nn_refiner import refine_with_cnn
                steps = int(os.environ.get("CNN_REFINER_STEPS", "8"))
                lr = float(os.environ.get("CNN_REFINER_LR", "0.001"))
                img_out = refine_with_cnn(img_out, steps=steps, lr=lr)
        except Exception as e:
            print(f"Neural refiner error (ignored): {e}")

        # Dataset-driven reference: transfer palette/tones from curated examples
        try:
            img_out = apply_dataset_reference(img_out, style_id)
        except Exception as e:
            print(f"Dataset reference application failed: {e}")

        # Apply trained dataset refiner if available (forward pass only)
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            weights_path = os.path.join(project_root, "models", "refiners", f"{style_id}.pt")
            if os.path.isfile(weights_path):
                img_out = refine_with_trained_cnn(img_out, weights_path)
        except Exception as e:
            print(f"Trained refiner application failed: {e}")

        if style_id != "ghibli_anime":
            img_out = _unsharp(img_out, amount=0.3)
        # Encode PNG to base64
        buf = io.BytesIO()
        img_out.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Local generation error: {e}")
