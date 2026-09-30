"""Device-resident 2D transforms for dense BM3D patch tensors.

All transform functions treat the final two dimensions as a square patch and
preserve arbitrary leading dimensions. A typical patch bank is
``[B, C, L, K, K]``; a filtered group bucket is ``[M, G, K, K]``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from bior_2d import bior_2d_forward, bior_2d_reverse


_TORCH_MATRIX_CACHE: dict[tuple[str, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


@lru_cache(maxsize=None)
def _bior_numpy_matrices(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Derive exact linear BIOR operators from the legacy PyWavelets layout."""
    basis = np.eye(size * size, dtype=np.float64).reshape(size * size, size, size)
    forward = bior_2d_forward(basis).reshape(size * size, size * size)
    reverse = bior_2d_reverse(basis).reshape(size * size, size * size)
    return forward, reverse


def _dct_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct the orthonormal DCT-II matrix ``[K, K]``."""
    samples = torch.arange(size, device=device, dtype=dtype) + 0.5
    frequencies = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos(math.pi * frequencies * samples / size)
    matrix[0] *= math.sqrt(1.0 / size)
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def transform_matrices(
    transform: str,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached forward/reverse operators for a device and dtype."""
    key = (transform, size, str(device), dtype)
    cached = _TORCH_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    if transform == "DCT":
        forward = _dct_matrix(size, device=device, dtype=dtype)
        reverse = forward.transpose(0, 1)
    elif transform == "BIOR":
        forward_np, reverse_np = _bior_numpy_matrices(size)
        forward = torch.as_tensor(forward_np, device=device, dtype=dtype)
        reverse = torch.as_tensor(reverse_np, device=device, dtype=dtype)
    else:
        raise ValueError(f"unsupported transform: {transform}")

    _TORCH_MATRIX_CACHE[key] = (forward, reverse)
    return forward, reverse


def forward_transform(patches: torch.Tensor, transform: str) -> torch.Tensor:
    """Transform patches whose final dimensions are ``[K, K]``."""
    size = patches.shape[-1]
    if patches.shape[-2] != size:
        raise ValueError("patches must be square")
    forward, _ = transform_matrices(
        transform, size, device=patches.device, dtype=patches.dtype,
    )
    if transform == "DCT":
        return torch.matmul(torch.matmul(forward, patches), forward.transpose(0, 1))

    # The BIOR coefficient layout used by the reference code is represented
    # as one exact [K*K,K*K] linear operator.
    flat = patches.reshape(*patches.shape[:-2], size * size)
    return torch.matmul(flat, forward).reshape_as(patches)


def inverse_transform(patches: torch.Tensor, transform: str) -> torch.Tensor:
    """Invert patches whose final dimensions are ``[K, K]``."""
    size = patches.shape[-1]
    if patches.shape[-2] != size:
        raise ValueError("patches must be square")
    forward, reverse = transform_matrices(
        transform, size, device=patches.device, dtype=patches.dtype,
    )
    if transform == "DCT":
        return torch.matmul(torch.matmul(reverse, patches), forward)

    flat = patches.reshape(*patches.shape[:-2], size * size)
    return torch.matmul(flat, reverse).reshape_as(patches)


def transformed_patch_bank(image: torch.Tensor, patch_size: int, transform: str) -> torch.Tensor:
    """Convert ``[B,C,H,W]`` into transformed ``[B,C,L,K,K]`` patches.

    ``torch.nn.functional.unfold`` initially returns ``[B,C*K*K,L]``. The
    reshape makes channel, patch position, and within-patch axes explicit
    before the 2D transform is applied to every patch in parallel.
    """
    if image.ndim != 4:
        raise ValueError("image must have BCHW shape")
    batch, channels, _, _ = image.shape
    unfolded = F.unfold(image, kernel_size=patch_size, stride=1)
    patch_count = unfolded.shape[-1]
    patches = unfolded.reshape(
        batch, channels, patch_size, patch_size, patch_count,
    ).permute(0, 1, 4, 2, 3)
    return forward_transform(patches, transform)
