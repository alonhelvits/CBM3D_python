"""Example runner for the tensorized color BM3D backend.

The legacy ``run_cbm3d.py`` runner and NumPy implementation are intentionally
left untouched so their output can be used as a reference.
"""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from bm3d_tensor import BM3DConfig, run_cbm3d_tensor


IMAGE_PATH = Path("data/yosemite.png")
OUTPUT_DIR = Path("data/denoised_tensor")
IMAGE_SIZE = 512
SIGMA = 20.0
WIENER_SIGMA_MULTIPLIER = 1.0


def select_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work so elapsed times are meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_rgb(path: Path, size: int) -> np.ndarray:
    """Load RGB, center-crop it to a square, and resize it to ``size``."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    rgb = rgb[top:top + side, left:left + side]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def save_rgb(path: Path, image: torch.Tensor) -> None:
    """Move a ``[3,H,W]`` RGB result to CPU and save it with OpenCV."""
    rgb = image.detach().to(device="cpu").permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not write image: {path}")


def main() -> None:
    """Run tensor CBM3D on the best available device and save its results."""
    device = select_device()
    clean = load_rgb(IMAGE_PATH, IMAGE_SIZE)
    rng = np.random.default_rng(0)
    noisy = np.clip(clean + rng.normal(0, SIGMA, size=clean.shape), 0, 255)
    # OpenCV/NumPy uses [H,W,C]; the tensor backend deliberately exposes the
    # standard PyTorch channel-first layout [C,H,W].
    noisy_tensor = (
        torch.from_numpy(noisy)
        .permute(2, 0, 1)
        .to(device=device, dtype=torch.float32)
    )

    config = BM3DConfig.classic(SIGMA, color=True)
    config = replace(
        config,
        wiener=replace(
            config.wiener,
            sigma_multiplier=WIENER_SIGMA_MULTIPLIER,
        ),
    )

    synchronize(device)
    start = perf_counter()
    with torch.inference_mode():
        basic, final = run_cbm3d_tensor(noisy_tensor, SIGMA, config)
    synchronize(device)
    elapsed = perf_counter() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_rgb(OUTPUT_DIR / "01_basic.png", basic)
    save_rgb(OUTPUT_DIR / "02_final.png", final)
    print(f"Device: {device}")
    print(f"Image size: {tuple(noisy_tensor.shape)}")
    print(f"Runtime: {elapsed:.3f} seconds")
    print(f"Wiener sigma multiplier: {WIENER_SIGMA_MULTIPLIER}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
