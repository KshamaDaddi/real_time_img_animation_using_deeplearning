AI Image Generator (Django)

This project is a Django app that generates styled images locally. It supports Stable Diffusion img2img when available and cleanly falls back to AnimeGAN on Windows or CPU-only setups. The application also includes animated GIF creation with various facial animations.

Quick Start (Windows)
- Python 3.11 recommended.
- Create and activate a virtual environment:
  - `python -m venv envi`
  - `envi\Scripts\activate`
- Install dependencies:
  - `pip install -r requirements.txt`
- Install PyTorch separately based on your platform:
  - CPU-only: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`
  - CUDA: see commands at https://pytorch.org/get-started/locally/
- Run the server:
  - `python manage.py runserver`
 - Open the app: `http://127.0.0.1:8000/` (Generate page)

Project Structure
- `generator/` Django app with views, routes, and templates.
- `ml/style_pipeline.py` ML pipeline with SD + AnimeGAN fallbacks.
- `ml/gif_creator.py` GIF animation creation with facial feature detection.
- `ml/prefetch_models.py` optional model prefetch script.
- `ml/test_run.py` local pipeline sanity test (bypasses web UI).
- `templates/` HTML templates and static assets.
 - `datasets/` optional curated reference images per style (see below).

Environment Variables (.env)
The project loads `.env` from the repo root. Create a `.env` file with any of the following (all optional):
- `DISABLE_SD=1` force AnimeGAN-only fallback if SD fails or is undesired.
- `SD_BASE_MODEL=runwayml/stable-diffusion-v1-5` base model (if using SD).
- `SD_GUIDANCE=7.5` guidance scale for SD (float).
- `SD_SCHEDULER=DDIM` scheduler name for SD.
- Accuracy tuning (img2img):
  - `SD_MAX_RES=768` max target dimension for SD resizing (GPU default: 768, CPU default: 512). Lower if you hit OOM.
  - `SD_STRENGTH=0.55` base strength (lower preserves more of the original image). Auto-adapts upward/downward based on input size.
  - `SD_STEPS=26` base steps (auto-adds steps for larger inputs).
  - `SD_GUIDANCE=7.0–7.5` guidance scale (higher adheres more to text prompt if provided).
- LoRA (optional, if using SD img2img for styles):
  - `LORA_CYBERPUNK_REPO=Joeythemonster/Cyberpunk-Anime-Style-LoRA`
  - `LORA_CYBERPUNK_FILE=cyberpunk_anime.safetensors`
  - Add similar `LORA_*` vars for other styles as needed.

Strict Anime Mode (Global)
Enable a universal pass to push all outputs toward bold anime aesthetics (useful when SD falls back or prompts are weak):
- `STRICT_ANIME_MODE=1` to enable
- `STRICT_ANIME_STRENGTH=0.6` intensity of outlines, cel shading, and crispness
Recommended SD base model for anime:
- `SD_BASE_MODEL=emilianJR/anything-v4.5` (good all-purpose anime). If SD fails on Windows, install Torch/torchvision per Quick Start.

Style Fingerprint Controls
You can fine-tune the post-process intensity for each style. All values are floats from `0.0` (disabled) to `1.0` (full effect).
- 90s Anime:
  - `FINGERPRINT_90S_GRAIN=0.06`
  - `FINGERPRINT_90S_SCANLINES=0.14`
  - `FINGERPRINT_90S_ABERRATION=1.0`
- Ghibli Anime:
  - `FINGERPRINT_GHIBLI_PAINTERLY=0.6` (subtle, watercolor-like texture)
  - `FINGERPRINT_GHIBLI_GREEN_BOOST=0.4` (enhances lush, natural greens)
  - `FINGERPRINT_GHIBLI_WARM_BALANCE=0.5` (adds a warm, nostalgic tint)
- Aesthetic Anime:
  - `FINGERPRINT_AESTHETIC_PASTEL=1.0`
  - `FINGERPRINT_AESTHETIC_BLUR=1.0`
- Japanese Anime:
  - `FINGERPRINT_JAPANESE_EDGES=1.0`
- Illustration Anime:
  - `FINGERPRINT_ILLUSTRATION_LINES=0.7` — Strength of bold outlines
  - `FINGERPRINT_ILLUSTRATION_CEL_SHADE=0.65` — Flat-color cel shading amount
  - `FINGERPRINT_ILLUSTRATION_SATURATION=0.25` — Color boost (vibrancy)
  - `FINGERPRINT_ILLUSTRATION_EYE_GLINTS=0.4` — Reflective highlights in eyes
  - `FINGERPRINT_ILLUSTRATION_SPEED_LINES=0.0` — Motion speed lines overlay
  - `FINGERPRINT_ILLUSTRATION_PAINTERLY=0.2` — Subtle painterly background texture
  - `FINGERPRINT_ILLUSTRATION_CRISP=0.4` — Final unsharp mask amount
- Fantasy Anime:
  - `FINGERPRINT_FANTASY_VIBRANT=0.7` (boost color saturation)
  - `FINGERPRINT_FANTASY_CEL_SHADE=0.6` (cel shading for clear forms)
  - `FINGERPRINT_FANTASY_BOLD_LINES=0.75` (strong outlines)
  - `FINGERPRINT_FANTASY_DEPTH=0.3` (misty atmospheric perspective)
  - `FINGERPRINT_FANTASY_AURA=0.5` (energy aura around subject)
  - `FINGERPRINT_FANTASY_RUNES=0.4` (glowing markings/sigils overlay)
  - `FINGERPRINT_FANTASY_GLOW=0.9` (magical glow and grade)
  - `FINGERPRINT_FANTASY_VIGNETTE=0.4` (cinematic focus)
- Cyberpunk Anime:
  - `FINGERPRINT_CYBERPUNK_NEON=0.9` — Neon magenta/cyan color grade strength
  - `FINGERPRINT_CYBERPUNK_GRAIN=0.2` — Film grain amount for gritty texture
  - `FINGERPRINT_CYBERPUNK_SCANLINES=0.15` — Display scanline intensity
  - `FINGERPRINT_CYBERPUNK_CHROMA=0.35` — Chromatic aberration strength (edge color split)
  - `FINGERPRINT_CYBERPUNK_RAIN=0.4` — Rain streak overlay intensity
  - `FINGERPRINT_CYBERPUNK_HOLOGRAMS=0.3` — Holographic billboard overlay intensity
  - `FINGERPRINT_CYBERPUNK_VIGNETTE=0.35` — Noir-style corner darkening
  - `FINGERPRINT_CYBERPUNK_RIMLIGHT=0.6` — Cyan rim lighting along silhouettes
- Chibi Anime:
  - `FINGERPRINT_CHIBI_KAWAII=0.7` (Controls the "cuteness" effect by subtly enlarging eyes and adding a soft blush to cheeks.)

### Cyberpunk Style Details

The cyberpunk style emphasizes the genre's "High Tech, Low Life" aesthetic:

- Dystopian mega-city mood via rain, haze, and vignette.
- Neon-drenched scenes with magenta/cyan grading and glow.
- Gritty film grain and scanlines for display artifacts.
- Chromatic aberration for optical distortion and edge color split.
- Holographic billboards to suggest corporate dominance and dense signage.
- Cyan rim lighting to enhance silhouettes during action shots.

Tune the above environment variables to push toward noir grunge or neon glamour, and set any to `0` to disable that effect.

Neural Refiner (optional)
You can enable a small CNN-based refiner that uses backprop for per-image clarity improvements.
It runs for a few iterations and works on CPU or GPU (PyTorch required):
- `ENABLE_CNN_REFINER=1` enable the CNN clarity refiner
  - `CNN_REFINER_STEPS=8` training iterations per image
  - `CNN_REFINER_LR=0.001` learning rate

Notes:
- This refiner is lightweight but will add a few seconds on CPU.
- If performance is slow, lower the steps or disable the refiner.
- Torch is required (see PyTorch install section in requirements and Quick Start).

Example `.env`
```
DISABLE_SD=0
SD_BASE_MODEL=runwayml/stable-diffusion-v1-5
SD_MAX_RES=768
SD_STRENGTH=0.55
SD_STEPS=26
SD_GUIDANCE=7.5

# Fingerprints
FINGERPRINT_90S_GRAIN=0.05
FINGERPRINT_90S_SCANLINES=0.12
FINGERPRINT_CYBERPUNK_NEON=0.7
FINGERPRINT_GHIBLI_WARMTH=0.8
```

Prefetch Models (optional)
- You can pre-download base models and optional LoRAs to avoid on-demand downloads:
  - `python -m ml.prefetch_models`
- This respects the `.env` variables above. If not prefetched, models download on first use.

Dataset Reference (pretrained datasets)
- You can guide the final color palette and tonal mood using curated example images.
- Place images under `datasets/<style_id>/` to condition outputs for that style.
  - Example: `datasets/ghibli_anime/forest_01.jpg`, `datasets/cyberpunk_anime/neon_street.png`
- The pipeline automatically applies palette transfer from a random reference image.
- Controls:
  - `ENABLE_DATASET_REFERENCE=1` enables conditioning (default: 1)
  - `DATASETS_DIR=<path>` overrides the datasets folder (default: `<project>/datasets`)
  - `DATASET_ALPHA_<STYLE_ID>=0.65` per-style blend strength (0.0–1.0)
    - Example: `DATASET_ALPHA_CYBERPUNK_ANIME=0.75`
- Fallback:
  - If `datasets/<style_id>/` is empty or missing, it falls back to `static/images/<style_id>.jpg` if available.
- Tips for better results:
  - Use clean, representative examples with strong style colors.
  - Prefer images that match your subject (portrait vs landscape) for more natural transfer.

GIF Animation Features
The application now includes advanced GIF creation with facial feature detection:
- Multiple animation types: Blinking Eyes, Crying, Talking, Winking, Waving, Bouncing, Face Movement
- Facial detection using OpenCV for accurate animation placement
- Adjustable animation speed (Slow, Medium, Fast)
- Real-time GIF generation with download capability

Local Sanity Test
- To verify generation without the web UI:
  - `python -m ml.test_run`
- This writes `static/images/generated_test.png` and will use AnimeGAN if SD is unavailable.

High-Quality Input Accuracy
- Adaptive SD resizing keeps the input aspect ratio and selects the largest feasible target size aligned to a 64px grid (`SD_MAX_RES`).
- SD parameters adapt to input size to preserve details:
  - Larger inputs reduce `strength` and add steps.
- Final outputs are upscaled back to the original input size (LANCZOS) before style fingerprints, improving perceived detail.

Troubleshooting
- If you see `DLL load failed while importing _C` or `xFormers` warnings on Windows, set `DISABLE_SD=1` to use AnimeGAN-only.
- `xFormers` is optional and not required for CPU-only; skip installing it on Windows.
- Ensure `python-dotenv` is installed; settings load `.env` automatically.
- If you hit CUDA OOM or slow CPU runs, lower `SD_MAX_RES` (e.g., `512`) and reduce fingerprint intensities.

Datasets Setup Quickstart
- Create folders per style under `datasets/` and add 5–20 curated images each.
- Verify conditioning is active by checking logs for "Dataset reference applied" messages.
- Tune `DATASET_ALPHA_*` per style to balance originality vs palette transfer.

Notes
- Stable Diffusion requires PyTorch; install torch/torchvision first for SD features.
- When SD is available, styles like `90s_anime` can use img2img with tuned prompts and optional LoRA files.