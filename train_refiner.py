import os
import sys
from typing import List, Dict

import numpy as np
from PIL import Image

# Ensure project root on path
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.nn_refiner import ClarityRefinerCNN, _pil_to_tensor
from ml.dataset_reference import _datasets_root, STYLE_ID_TO_DATASET_PATH, _image_paths_in


def _collect_images(paths: List[str], max_per_style: int = 200) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in paths[:max_per_style]:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    return imgs


def _train_refiner_on_images(images: List[Image.Image], epochs: int = 1, steps_per_img: int = 2, lr: float = 1e-3, device: str = None):
    import torch
    import torch.nn.functional as F

    refiner = ClarityRefinerCNN(device=device)
    opt = torch.optim.Adam(refiner.model.parameters(), lr=lr)

    # Sobel kernels for edge magnitude
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
    sobel_x = sobel_x.expand(3, 1, 3, 3).to(refiner.device)
    sobel_y = sobel_y.expand(3, 1, 3, 3).to(refiner.device)

    def edge_mag(z):
        # depthwise conv per-channel
        zx = F.conv2d(z, sobel_x, padding=1, groups=3)
        zy = F.conv2d(z, sobel_y, padding=1, groups=3)
        return torch.sqrt(zx * zx + zy * zy + 1e-6)

    images = images or []
    if not images:
        return refiner

    for _ in range(max(1, epochs)):
        for img in images:
            x = _pil_to_tensor(img).to(refiner.device)
            for _ in range(max(1, steps_per_img)):
                opt.zero_grad()
                y = refiner.model(x)
                # Loss: maintain similarity + encourage edges + keep colors in range
                mse = F.mse_loss(y, x)
                edges = edge_mag(y).mean()
                # Encourage stronger edges via negative term; small weight
                loss = mse - 0.05 * edges
                loss.backward()
                opt.step()

    return refiner


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train dataset-based CNN refiners per style.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_img", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only_style", type=str, default=None, help="Train only the specified style_id")
    args = parser.parse_args()

    base = _datasets_root()
    out_dir = os.path.join(_PROJECT_ROOT, "models", "refiners")
    os.makedirs(out_dir, exist_ok=True)

    # Build style -> image paths
    for style_id, mapped in STYLE_ID_TO_DATASET_PATH.items():
        if args.only_style and style_id != args.only_style:
            continue
        # Resolve to list of directories
        dirs = mapped if isinstance(mapped, list) else [mapped]
        all_paths: List[str] = []
        for d in dirs:
            dir_path = os.path.join(base, d)
            if os.path.isdir(dir_path):
                all_paths.extend(_image_paths_in(dir_path))

        if not all_paths:
            print(f"No dataset images for {style_id}; skipping")
            continue

        try:
            imgs = _collect_images(all_paths, max_per_style=args.max_images)
            print(f"Training refiner for {style_id} on {len(imgs)} images...")
            refiner = _train_refiner_on_images(
                imgs,
                epochs=args.epochs,
                steps_per_img=args.steps_per_img,
                lr=1e-3,
                device=args.device,
            )

            # Save weights
            import torch
            out_path = os.path.join(out_dir, f"{style_id}.pt")
            torch.save(refiner.model.state_dict(), out_path)
            print(f"Saved refiner weights to {out_path}")
        except Exception as e:
            print(f"Training failed for {style_id}: {e}")


if __name__ == "__main__":
    main()
