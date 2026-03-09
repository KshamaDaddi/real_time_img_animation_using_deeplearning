import os
import sys
import json
from typing import Dict, List

import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path for package imports
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse helpers and mapping from dataset_reference
from ml.dataset_reference import _project_root, _datasets_root, _image_paths_in, STYLE_ID_TO_DATASET_PATH


def _ensure_models_dir() -> str:
    root = _project_root()
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _compute_lab_stats(image_paths: List[str]) -> Dict[str, List[float]]:
    if not image_paths:
        return {"lab_mean": [0.0, 0.0, 0.0], "lab_std": [1.0, 1.0, 1.0]}

    means = []
    stds = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            m = [float(np.mean(lab[..., c])) for c in range(3)]
            s = [float(np.std(lab[..., c])) for c in range(3)]
            means.append(m)
            stds.append(s)
        except Exception:
            continue

    if not means:
        return {"lab_mean": [0.0, 0.0, 0.0], "lab_std": [1.0, 1.0, 1.0]}

    means_arr = np.array(means, dtype=np.float32)
    stds_arr = np.array(stds, dtype=np.float32)
    lab_mean = means_arr.mean(axis=0).tolist()
    lab_std = (stds_arr.mean(axis=0) + 1e-6).tolist()
    return {"lab_mean": lab_mean, "lab_std": lab_std}


def main():
    ds_root = _datasets_root()
    out_dir = _ensure_models_dir()
    out_path = os.path.join(out_dir, "style_stats.json")

    results: Dict[str, Dict[str, List[float]]] = {}

    for style_id, rel_path in STYLE_ID_TO_DATASET_PATH.items():
        rel_paths = rel_path if isinstance(rel_path, list) else [rel_path]
        all_paths: List[str] = []
        for rp in rel_paths:
            ds_dir = os.path.join(ds_root, rp)
            if not os.path.isdir(ds_dir):
                continue
            all_paths.extend(_image_paths_in(ds_dir))
        if not all_paths:
            print(f"Skip {style_id}: no dataset images found")
            continue
        stats = _compute_lab_stats(all_paths)
        results[style_id] = stats
        print(f"Computed stats for {style_id}: mean={stats['lab_mean']}, std={stats['lab_std']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved style stats to {out_path}")


if __name__ == "__main__":
    main()
