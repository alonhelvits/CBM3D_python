"""Batched collaborative filters for transformed BM3D patch groups.

Public functions consume dense group buckets with shape ``[M, G, K, K]``:
``M`` groups, ``G`` similar patches per group, and ``K x K`` coefficients per
patch. Group sizes are powers of two so a cached Hadamard matrix can act on
the ``G`` axis.
"""

from __future__ import annotations

import math

import torch


_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def hadamard_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an unnormalized Sylvester Hadamard matrix with shape ``[G, G]``."""
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    _HADAMARD_CACHE[key] = matrix
    return matrix


def sd_weight(groups: torch.Tensor) -> torch.Tensor:
    """Return the legacy inverse sample-SD weight with shape ``[M]``.

    The statistic includes all ``G*K*K`` transformed coefficients in a group,
    matching ``utils.sd_weighting`` in the reference implementation.
    """
    flat = groups.flatten(1)
    count = flat.shape[1]
    if count <= 1:
        return torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype)
    total = flat.sum(dim=1)
    square_total = flat.square().sum(dim=1)
    variance = (square_total - total.square() / count) / (count - 1)
    return torch.where(variance > 0, variance.rsqrt(), torch.ones_like(variance))


def hard_threshold_groups(
    groups: torch.Tensor,
    sigma: torch.Tensor | float,
    threshold_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-threshold ``[M, G, K, K]`` groups and return groups plus ``[M]`` weights."""
    group_size = groups.shape[1]
    matrix = hadamard_matrix(group_size, device=groups.device, dtype=groups.dtype)
    # [M,G,K,K] -> [M,K,K,G]: Hadamard acts across similar patches,
    # independently for every 2D-transform coefficient.
    values = groups.permute(0, 2, 3, 1)
    transformed = torch.matmul(values, matrix)
    sigma_tensor = torch.as_tensor(sigma, device=groups.device, dtype=groups.dtype)
    threshold = threshold_multiplier * sigma_tensor * math.sqrt(group_size)
    mask = transformed.abs() > threshold
    nonzero = mask.flatten(1).sum(dim=1)
    filtered = torch.where(mask, transformed, torch.zeros_like(transformed))
    filtered = torch.matmul(filtered, matrix) / group_size
    filtered = filtered.permute(0, 3, 1, 2)
    denominator = sigma_tensor.square() * nonzero.to(groups.dtype)
    weight = torch.where(nonzero > 0, denominator.reciprocal(), torch.ones_like(denominator))
    return filtered, weight


def wiener_filter_groups(
    noisy_groups: torch.Tensor,
    basic_groups: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    sigma_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wiener-filter two ``[M, G, K, K]`` banks and return ``[M]`` weights.

    ``noisy_groups`` provides the values being filtered. ``basic_groups``
    estimates their signal power and therefore determines the Wiener gain.
    """
    if noisy_groups.shape != basic_groups.shape:
        raise ValueError("noisy_groups and basic_groups must have identical shapes")
    group_size = noisy_groups.shape[1]
    matrix = hadamard_matrix(
        group_size, device=noisy_groups.device, dtype=noisy_groups.dtype,
    )
    # Move G last so a single batched matmul transforms every coefficient.
    noisy = noisy_groups.permute(0, 2, 3, 1)
    basic = basic_groups.permute(0, 2, 3, 1)
    noisy_h = torch.matmul(noisy, matrix)
    basic_h = torch.matmul(basic, matrix)

    sigma_tensor = torch.as_tensor(
        sigma, device=noisy_groups.device, dtype=noisy_groups.dtype,
    ) * sigma_multiplier
    power = basic_h.square() / group_size
    gain = power / (power + sigma_tensor.square())
    filtered_h = noisy_h * gain / group_size
    filtered = torch.matmul(filtered_h, matrix).permute(0, 3, 1, 2)

    gain_sum = gain.flatten(1).sum(dim=1)
    denominator = sigma_tensor.square() * gain_sum
    weight = torch.where(gain_sum > 0, denominator.reciprocal(), torch.ones_like(denominator))
    return filtered, weight
