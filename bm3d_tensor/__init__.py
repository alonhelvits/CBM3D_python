"""PyTorch BM3D backend for CPU, CUDA, and Apple MPS.

The public grayscale API accepts ``HW``, ``1HW``, or ``B1HW`` tensors. The
color API accepts RGB ``3HW`` or ``B3HW`` tensors. Outputs retain the input
layout and device. Float32/float64 inputs retain their dtype; other inputs are
promoted to float32.
"""

from .config import BM3DConfig, HardThresholdConfig, RuntimeConfig, WienerConfig
from .core import run_bm3d_tensor, run_cbm3d_tensor

__all__ = [
    "BM3DConfig",
    "HardThresholdConfig",
    "RuntimeConfig",
    "WienerConfig",
    "run_bm3d_tensor",
    "run_cbm3d_tensor",
]
