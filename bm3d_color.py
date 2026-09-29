import numpy as np

from bm3d_1st_step_color import bm3d_1st_step_color
from bm3d_2nd_step_color import bm3d_2nd_step_color
from color_utils import rgb_sigma_to_yuv, rgb_to_yuv, yuv_to_rgb
from utils import symetrize


def _validate_parameters(image_shape,
                         n_H, k_H, N_H, p_H, tauMatch_H, tau_2D_H,
                         n_W, k_W, N_W, p_W, tauMatch_W, tau_2D_W):
    height, width, _ = image_shape
    for stage, n, k, group_limit, step, match_threshold, transform in (
        ('hard-threshold', n_H, k_H, N_H, p_H, tauMatch_H, tau_2D_H),
        ('Wiener', n_W, k_W, N_W, p_W, tauMatch_W, tau_2D_W),
    ):
        if n < 0 or k <= 0 or step <= 0 or match_threshold <= 0.:
            raise ValueError(f"Invalid {stage} geometry or matching threshold")
        if group_limit <= 0 or group_limit & (group_limit - 1):
            raise ValueError(f"{stage} group limit must be a positive power of two")
        if group_limit > (2 * n + 1) ** 2:
            raise ValueError(f"{stage} group limit exceeds its search window")
        if transform not in ('BIOR', 'DCT'):
            raise ValueError(f"Unsupported {stage} transform: {transform}")
        if height < k or width < k:
            raise ValueError(f"Image is smaller than the {stage} patch size")


def run_bm3d_color(noisy_rgb, sigma,
                    n_H, k_H, N_H, p_H, tauMatch_H, useSD_H,
                    tau_2D_H, lambda3D_H,
                    n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    """Denoise a full-resolution RGB image through a signed YUV transform.

    Matching uses Y only. Y, U, and V are filtered independently with their
    transformed noise sigmas and shared match locations. Both aggregation
    stages use the scalar weight derived from Y.
    """
    noisy_yuv = rgb_to_yuv(noisy_rgb)
    sigma_yuv = rgb_sigma_to_yuv(sigma)

    # Patch-size selection follows the actual noise level in the match channel.
    sigma_y = sigma_yuv[0]
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma_y < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma_y < 40.) else 12
    _validate_parameters(
        noisy_yuv.shape,
        n_H, k_H, N_H, p_H, tauMatch_H, tau_2D_H,
        n_W, k_W, N_W, p_W, tauMatch_W, tau_2D_W,
    )

    noisy_yuv_p = symetrize(noisy_yuv, n_H)
    basic_yuv = bm3d_1st_step_color(
        sigma_yuv, noisy_yuv_p, n_H, k_H, N_H, p_H,
        lambda3D_H, tauMatch_H, useSD_H, tau_2D_H,
    )
    basic_yuv = basic_yuv[n_H:-n_H, n_H:-n_H]
    if not np.isfinite(basic_yuv).all():
        raise RuntimeError("Color BM3D stage one produced non-finite values")

    basic_yuv_p = symetrize(basic_yuv, n_W)
    noisy_yuv_p = symetrize(noisy_yuv, n_W)
    denoised_yuv = bm3d_2nd_step_color(
        sigma_yuv, noisy_yuv_p, basic_yuv_p, n_W, k_W, N_W,
        p_W, tauMatch_W, useSD_W, tau_2D_W,
    )
    denoised_yuv = denoised_yuv[n_W:-n_W, n_W:-n_W]
    if not np.isfinite(denoised_yuv).all():
        raise RuntimeError("Color BM3D stage two produced non-finite values")

    # YUV remains unclipped throughout filtering. Clip only after RGB recovery.
    basic_rgb = np.clip(yuv_to_rgb(basic_yuv), 0., 255.)
    denoised_rgb = np.clip(yuv_to_rgb(denoised_yuv), 0., 255.)
    return basic_rgb, denoised_rgb
