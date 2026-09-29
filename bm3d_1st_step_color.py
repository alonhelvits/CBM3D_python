import numpy as np

from bior_2d import bior_2d_forward, bior_2d_reverse
from build_3D_group import build_3D_group
from dct_2d import dct_2d_forward, dct_2d_reverse
from ht_filtering_hadamard import ht_filtering_hadamard
from image_to_patches import image2patches
from precompute_BM import precompute_BM
from utils import get_kaiserWindow, ind_initialize, sd_weighting


def _forward_transform(patches, transform):
    if transform == 'DCT':
        return dct_2d_forward(patches)
    if transform == 'BIOR':
        return bior_2d_forward(patches)
    raise ValueError("tau_2D must be 'DCT' or 'BIOR'")


def _reverse_transform(patches, transform):
    if transform == 'DCT':
        return dct_2d_reverse(patches)
    if transform == 'BIOR':
        return bior_2d_reverse(patches)
    raise ValueError("tau_2D must be 'DCT' or 'BIOR'")


def bm3d_1st_step_color(sigma_yuv, img_noisy, nHard, kHard, NHard,
                        pHard, lambdaHard3D, tauMatch, useSD, tau_2D):
    """Build a YUV basic estimate using Y matches and shared Y weights."""
    if img_noisy.ndim != 3 or img_noisy.shape[2] != 3:
        raise ValueError("img_noisy must have shape (height, width, 3)")
    sigma_yuv = np.asarray(sigma_yuv, dtype=np.float64)
    if (sigma_yuv.shape != (3,) or not np.isfinite(sigma_yuv).all()
            or not np.all(sigma_yuv > 0.)):
        raise ValueError("sigma_yuv must contain three positive values")

    height, width, _ = img_noisy.shape
    row_ind = ind_initialize(height - kHard + 1, nHard, pHard)
    column_ind = ind_initialize(width - kHard + 1, nHard, pHard)
    kaiser_window = get_kaiserWindow(kHard)

    # Patch locations are computed once from luminance and reused by U and V.
    match_table, threshold_count = precompute_BM(
        img_noisy[..., 0], kHW=kHard, NHW=NHard,
        nHW=nHard, tauMatch=tauMatch,
    )
    group_len = sum(
        int(threshold_count[i_r, j_r])
        for i_r in row_ind for j_r in column_ind
    )

    numerator = np.zeros((height, width, 3), dtype=np.float64)
    denominator = np.zeros((height - 2 * nHard, width - 2 * nHard), dtype=np.float64)
    denominator = np.pad(denominator, nHard, 'constant', constant_values=1.)
    weight_table = np.zeros((height, width), dtype=np.float64)

    # Channels are processed sequentially to keep peak memory near grayscale BM3D.
    for channel in range(3):
        patches = image2patches(img_noisy[..., channel], kHard, kHard)
        transformed_patches = _forward_transform(patches, tau_2D)
        filtered_groups = np.zeros((group_len, kHard, kHard), dtype=np.float64)

        acc_pointer = 0
        for i_r in row_ind:
            for j_r in column_ind:
                group_size = int(threshold_count[i_r, j_r])
                group = build_3D_group(
                    transformed_patches,
                    match_table[i_r, j_r],
                    group_size,
                )
                group, filter_weight = ht_filtering_hadamard(
                    group, sigma_yuv[channel], lambdaHard3D, not useSD,
                )
                group = group.transpose((2, 0, 1))
                filtered_groups[acc_pointer:acc_pointer + group_size] = group
                acc_pointer += group_size

                # Only luminance determines the scalar shared by all channels.
                if channel == 0:
                    weight = sd_weighting(group) if useSD else filter_weight
                    weight_table[i_r, j_r] = weight

        filtered_groups = _reverse_transform(filtered_groups, tau_2D)

        acc_pointer = 0
        for i_r in row_ind:
            for j_r in column_ind:
                group_size = int(threshold_count[i_r, j_r])
                coordinates = match_table[i_r, j_r]
                group = filtered_groups[acc_pointer:acc_pointer + group_size]
                acc_pointer += group_size
                weight = weight_table[i_r, j_r]

                for n in range(group_size):
                    ni, nj = coordinates[n]
                    weighted_window = kaiser_window * weight
                    numerator[ni:ni + kHard, nj:nj + kHard, channel] += (
                        group[n] * weighted_window
                    )
                    if channel == 0:
                        denominator[ni:ni + kHard, nj:nj + kHard] += weighted_window

    if np.any(denominator <= 0.) or not np.isfinite(denominator).all():
        raise RuntimeError("Color stage-one aggregation left invalid denominator values")
    return numerator / denominator[..., np.newaxis]
