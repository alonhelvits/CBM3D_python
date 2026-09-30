"""Typed configuration for the tensor BM3D implementation.

Algorithm parameters are separated from runtime chunk sizes. Changing a
``RuntimeConfig`` value changes memory use and launch granularity, but not the
set of BM3D operations or their mathematical parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


TransformName = Literal["DCT", "BIOR"]


@dataclass(frozen=True)
class HardThresholdConfig:
    """Stage-one geometry, matching, transform, threshold, and weight choices."""

    search_radius: int = 16
    patch_size: int = 8
    max_group_size: int = 16
    reference_step: int = 3
    match_threshold: float = 2500.0
    transform: TransformName = "BIOR"
    use_sd_weight: bool = False
    threshold_multiplier: float = 2.7


@dataclass(frozen=True)
class WienerConfig:
    """Stage-two settings.

    ``sigma_multiplier`` changes the effective sigma in the Wiener gain and
    its legacy aggregation weight. It is deliberately independent of the
    matching threshold and the stage-one sigma.
    """

    search_radius: int = 16
    patch_size: int = 8
    max_group_size: int = 32
    reference_step: int = 3
    match_threshold: float = 400.0
    transform: TransformName = "DCT"
    use_sd_weight: bool = True
    sigma_multiplier: float = 1.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Memory/performance controls that do not change BM3D's math.

    ``reference_chunk_size`` and ``displacement_chunk_size`` bound the
    matcher's temporary ``[B, Rc, Dc, K*K]`` candidate tensor.
    ``group_chunk_size`` bounds the filtered ``[M, G, K, K]`` tensor.
    """

    reference_chunk_size: int = 1024
    displacement_chunk_size: int = 64
    group_chunk_size: int = 512


@dataclass(frozen=True)
class BM3DConfig:
    """Complete algorithm and runtime configuration for both BM3D stages."""

    hard: HardThresholdConfig = field(default_factory=HardThresholdConfig)
    wiener: WienerConfig = field(default_factory=WienerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def classic(cls, sigma: float, *, color: bool = False) -> "BM3DConfig":
        """Return the parameter choices used by the legacy entry points."""
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        # Color patch-size selection uses the actual noise level in Y.
        y_noise_scale = math.sqrt(0.299**2 + 0.587**2 + 0.114**2)
        match_sigma = sigma * y_noise_scale if color else sigma
        hard_patch = 8  # BIOR always selects an 8x8 patch in the legacy code.
        wiener_patch = 8 if match_sigma < 40.0 else 12
        return cls(
            hard=HardThresholdConfig(
                patch_size=hard_patch,
                match_threshold=2500.0 if sigma < 35.0 else 5000.0,
            ),
            wiener=WienerConfig(
                patch_size=wiener_patch,
                match_threshold=400.0 if sigma < 35.0 else 3500.0,
            ),
        )


def validate_config(config: BM3DConfig) -> None:
    """Reject geometries that cannot produce valid patch or Hadamard groups."""
    for name, stage in (("hard", config.hard), ("wiener", config.wiener)):
        if stage.search_radius < 0:
            raise ValueError(f"{name} search_radius must be non-negative")
        if stage.patch_size <= 0 or stage.reference_step <= 0:
            raise ValueError(f"{name} patch_size and reference_step must be positive")
        if stage.max_group_size <= 0 or stage.max_group_size & (stage.max_group_size - 1):
            raise ValueError(f"{name} max_group_size must be a positive power of two")
        if stage.max_group_size > (2 * stage.search_radius + 1) ** 2:
            raise ValueError(f"{name} max_group_size exceeds its search window")
        if stage.match_threshold <= 0:
            raise ValueError(f"{name} match_threshold must be positive")
        if stage.transform not in ("DCT", "BIOR"):
            raise ValueError(f"unsupported {name} transform: {stage.transform}")
        if stage.transform == "BIOR" and stage.patch_size & (stage.patch_size - 1):
            raise ValueError("BIOR patch_size must be a power of two")

    if config.hard.threshold_multiplier <= 0:
        raise ValueError("hard threshold_multiplier must be positive")
    if config.wiener.sigma_multiplier <= 0:
        raise ValueError("wiener sigma_multiplier must be positive")

    runtime = config.runtime
    if min(
        runtime.reference_chunk_size,
        runtime.displacement_chunk_size,
        runtime.group_chunk_size,
    ) <= 0:
        raise ValueError("all runtime chunk sizes must be positive")
