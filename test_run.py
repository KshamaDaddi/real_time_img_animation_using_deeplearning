import os
from pathlib import Path
from ml.style_pipeline import generate_anime_image

ROOT = Path(__file__).resolve().parents[1]
sample_path = ROOT / "static" / "images" / "japanese_anime.jpg"
out_path = ROOT / "static" / "images" / "generated_test.png"

def main():
    if not sample_path.exists():
        print(f"Sample image not found: {sample_path}")
        return
    image_bytes = sample_path.read_bytes()
    b64 = generate_anime_image(image_bytes, "90s_anime")
    if not b64:
        print("Generation failed")
        return
    import base64
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()