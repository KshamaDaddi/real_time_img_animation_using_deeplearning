import os
from typing import Optional

import numpy as np
from PIL import Image


def _pil_to_tensor(img: Image.Image):
    import torch
    arr = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)  # NCHW


def _tensor_to_pil(t):
    arr = t.squeeze(0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0)) * 255.0
    return Image.fromarray(arr.astype(np.uint8))


class ClarityRefinerCNN:
    """Tiny CNN used for per-image backprop refinement to improve clarity.

    Architecture: 3 conv layers with residual connection.
    Trained a few iterations per image to increase edge magnitude
    while keeping colors in range.
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        import torch.nn as nn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 3, 3, padding=1)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                y = self.act(self.conv1(x))
                y = self.act(self.conv2(y))
                y = self.conv3(y)
                return torch.clamp(x + y, 0.0, 1.0)

        self.model = Net().to(self.device)

    def load_weights(self, weights_path: str):
        import torch
        sd = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

    def infer(self, img: Image.Image) -> Image.Image:
        import torch
        x = _pil_to_tensor(img).to(self.device)
        with torch.no_grad():
            y = self.model(x)
        return _tensor_to_pil(y)

    def refine(self, img: Image.Image, steps: int = 8, lr: float = 1e-3) -> Image.Image:
        import torch
        import torch.nn.functional as F
        x = _pil_to_tensor(img).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Sobel kernels for edge magnitude
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        sobel_x = sobel_x.expand(3, 1, 3, 3).to(self.device)
        sobel_y = sobel_y.expand(3, 1, 3, 3).to(self.device)

        def edge_mag(z):
            # depthwise conv per-channel
            zx = F.conv2d(z, sobel_x, padding=1, groups=3)
            zy = F.conv2d(z, sobel_y, padding=1, groups=3)
            return torch.sqrt(zx * zx + zy * zy + 1e-6)

        for _ in range(max(1, steps)):
            opt.zero_grad()
            y = self.model(x)
            # Loss: maintain similarity + encourage edges + keep colors in range
            mse = F.mse_loss(y, x)
            edges = edge_mag(y).mean()
            # Encourage stronger edges via negative term
            loss = mse - 0.05 * edges
            loss.backward()
            opt.step()

        with torch.no_grad():
            out = self.model(x)
        return _tensor_to_pil(out)




def refine_with_cnn(img: Image.Image, steps: int = 8, lr: float = 1e-3, device: Optional[str] = None) -> Image.Image:
    try:
        refiner = ClarityRefinerCNN(device=device)
        return refiner.refine(img, steps=steps, lr=lr)
    except Exception as e:
        # Fail silently; return original image
        print(f"CNN refiner error (ignored): {e}")
        return img


def refine_with_trained_cnn(img: Image.Image, weights_path: str, device: Optional[str] = None) -> Image.Image:
    """Apply a trained CNN refiner (forward pass only)."""
    try:
        refiner = ClarityRefinerCNN(device=device)
        refiner.load_weights(weights_path)
        return refiner.infer(img)
    except Exception as e:
        print(f"Trained CNN refiner error (ignored): {e}")
        return img
