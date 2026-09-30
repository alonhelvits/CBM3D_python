"""Example runner for tensorized grayscale BM3D on CPU, CUDA, or MPS.

The script loads an image, converts it to grayscale, adds deterministic
Gaussian noise, runs both BM3D stages, reports PSNR, and saves the clean,
noisy, basic, final, and comparison images. Generated files are kept separate
from both the legacy and color-tensor runners.
"""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from bm3d_tensor import BM3DConfig, run_bm3d_tensor


IMAGE_PATH = Path("data/yosemite.png")
OUTPUT_DIR = Path("data/denoised_tensor_gray")
IMAGE_SIZE = 512
SIGMA = 20.0
WIENER_SIGMA_MULTIPLIER = 1.0
SAVE_REFERENCE_PATCHES = False


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


def load_grayscale(path: Path, size: int) -> np.ndarray:
    """Load grayscale, center-crop it to a square, and resize to ``size``."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    height, width = image.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    image = image[top:top + side, left:left + side]
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def compute_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return grayscale PSNR in dB."""
    difference = reference.astype(np.float64) - estimate.astype(np.float64)
    mse = np.mean(difference**2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def to_numpy(image: torch.Tensor) -> np.ndarray:
    """Move an ``[H,W]`` result to CPU and return a clipped uint8 image."""
    array = image.detach().to(device="cpu").numpy()
    return np.clip(array, 0, 255).astype(np.uint8)


def save_grayscale(path: Path, image: np.ndarray) -> None:
    """Save an ``[H,W]`` grayscale image."""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def main() -> None:
    """Run grayscale tensor BM3D and save outputs and comparison metrics."""
    device = select_device()
    clean = load_grayscale(IMAGE_PATH, IMAGE_SIZE)

    rng = np.random.default_rng(0)
    noise = rng.normal(0, SIGMA, size=clean.shape)
    noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.float32)

    # The grayscale API accepts [H,W] directly and returns [H,W]. Keeping the
    # tensor on the selected device ensures the complete pipeline runs there.
    noisy_tensor = torch.from_numpy(noisy).to(device=device, dtype=torch.float32)

    config = BM3DConfig.classic(SIGMA, color=False)
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
        if SAVE_REFERENCE_PATCHES:
            basic, final, reference_patches = run_bm3d_tensor(
                noisy_tensor,
                SIGMA,
                config,
                return_reference_patches=True,
            )
        else:
            basic, final = run_bm3d_tensor(noisy_tensor, SIGMA, config)
            reference_patches = None
    synchronize(device)
    elapsed = perf_counter() - start

    basic_image = to_numpy(basic)
    final_image = to_numpy(final)
    noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)

    print(f"Device: {device}")
    print(f"Image size: {tuple(noisy_tensor.shape)}")
    print(f"Noise sigma: {SIGMA}")
    print(f"Runtime: {elapsed:.3f} seconds")
    print(f"Wiener sigma multiplier: {WIENER_SIGMA_MULTIPLIER}")
    print(f"Noisy PSNR: {compute_psnr(clean, noisy):.3f} dB")
    print(f"Basic PSNR: {compute_psnr(clean, basic_image):.3f} dB")
    print(f"Final PSNR: {compute_psnr(clean, final_image):.3f} dB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_grayscale(OUTPUT_DIR / "01_clean.png", clean)
    save_grayscale(OUTPUT_DIR / "02_noisy.png", noisy_image)
    save_grayscale(OUTPUT_DIR / "03_basic.png", basic_image)
    save_grayscale(OUTPUT_DIR / "04_final.png", final_image)

    comparison = np.concatenate(
        (clean, noisy_image, basic_image, final_image),
        axis=1,
    )
    save_grayscale(OUTPUT_DIR / "05_comparison.png", comparison)

    if reference_patches is not None:
        # Save tensors and their spatial metadata together. Patch tensors have
        # shape [B,R,C,K,K]; for this runner B=C=1.
        torch.save(
            {
                "hard_patches": reference_patches.hard.patches.cpu(),
                "hard_reference_indices": reference_patches.hard.reference_indices.cpu(),
                "hard_group_sizes": reference_patches.hard.group_sizes.cpu(),
                "hard_patch_grid_shape": reference_patches.hard.patch_grid_shape,
                "wiener_patches": reference_patches.wiener.patches.cpu(),
                "wiener_reference_indices": (
                    reference_patches.wiener.reference_indices.cpu()
                ),
                "wiener_group_sizes": reference_patches.wiener.group_sizes.cpu(),
                "wiener_patch_grid_shape": reference_patches.wiener.patch_grid_shape,
            },
            OUTPUT_DIR / "06_reference_patches.pt",
        )
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
