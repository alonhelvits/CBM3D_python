"""Legacy NumPy color-BM3D runner used as the output/runtime reference.

For the tensorized CPU/CUDA/MPS implementation, run ``run_cbm3d_tensor.py``.
The two runners write to different output directories so their images can be
compared without overwriting one another.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from bm3d_color import run_bm3d_color


IMAGE_PATH = Path("data/yosemite.png")
OUTPUT_DIR = Path("data/denoised")
IMAGE_SIZE = 512
SIGMA = 20


def compute_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return RGB PSNR in dB after promoting both images to float64."""
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)

    mse = np.mean((reference - estimate) ** 2)
    if mse == 0:
        return float("inf")

    return 10 * np.log10(255.0**2 / mse)


def load_small_rgb(path: Path, size: int) -> np.ndarray:
    """Load RGB, center-crop it to a square, and resize it to ``size``."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Take a centered square crop before resizing.
    height, width = rgb.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    rgb = rgb[top:top + side, left:left + side]

    return cv2.resize(
        rgb,
        (size, size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float64)


def save_rgb(path: Path, image: np.ndarray) -> None:
    """Clip a floating RGB image and save it through OpenCV's BGR API."""
    image = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not write image: {path}")


def main() -> None:
    """Run the legacy color pipeline and save reference images and metrics."""
    clean = load_small_rgb(IMAGE_PATH, IMAGE_SIZE)

    # Add independent Gaussian noise to R, G, and B.
    rng = np.random.default_rng(0)
    noise = rng.normal(0, SIGMA, size=clean.shape)
    noisy = np.clip(clean + noise, 0, 255)

    start = perf_counter()

    basic, final = run_bm3d_color(
        noisy,
        SIGMA,

        # Stage 1: hard-threshold filtering
        16,         # Search radius
        8,          # Patch size
        16,         # Maximum group size
        3,          # Reference-patch step
        2500,       # Matching threshold
        False,      # SD aggregation weighting
        "BIOR",     # Spatial transform
        2.7,        # Hard-threshold multiplier

        # Stage 2: Wiener filtering
        16,         # Search radius
        8,          # Patch size
        32,         # Maximum group size
        3,          # Reference-patch step
        400,        # Matching threshold
        True,       # SD aggregation weighting
        "DCT",      # Spatial transform
    )

    elapsed = perf_counter() - start

    print(f"Image size:  {clean.shape}")
    print(f"Noise sigma: {SIGMA}")
    print(f"Runtime:     {elapsed:.2f} seconds")
    print(f"Noisy PSNR:  {compute_psnr(clean, noisy):.3f} dB")
    print(f"Basic PSNR:  {compute_psnr(clean, basic):.3f} dB")
    print(f"Final PSNR:  {compute_psnr(clean, final):.3f} dB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_rgb(OUTPUT_DIR / "01_clean.png", clean)
    save_rgb(OUTPUT_DIR / "02_noisy.png", noisy)
    save_rgb(OUTPUT_DIR / "03_basic.png", basic)
    save_rgb(OUTPUT_DIR / "04_final.png", final)

    comparison = np.concatenate(
        [clean, noisy, basic, final],
        axis=1,
    )
    save_rgb(OUTPUT_DIR / "05_comparison.png", comparison)

    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
