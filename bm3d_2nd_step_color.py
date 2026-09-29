import numpy as np

from bior_2d import bior_2d_forward, bior_2d_reverse
from build_3D_group import build_3D_group
from dct_2d import dct_2d_forward, dct_2d_reverse
from image_to_patches import image2patches
from precompute_BM import precompute_BM
from utils import get_kaiserWindow, ind_initialize, sd_weighting
from wiener_filtering_hadamard import wiener_filtering_hadamard


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


def bm3d_2nd_step_color(sigma_yuv, img_noisy, img_basic, nWien,
                        kWien, NWien, pWien, tauMatch, useSD, tau_2D):
    """Refine YUV channels with Y matches and a shared Y aggregation weight."""
    if img_noisy.ndim != 3 or img_noisy.shape[2] != 3:
        raise ValueError("img_noisy must have shape (height, width, 3)")
    if img_basic.shape != img_noisy.shape:
        raise ValueError("img_basic must have the same shape as img_noisy")
    sigma_yuv = np.asarray(sigma_yuv, dtype=np.float64)
    if (sigma_yuv.shape != (3,) or not np.isfinite(sigma_yuv).all()
            or not np.all(sigma_yuv > 0.)):
        raise ValueError("sigma_yuv must contain three positive values")

    height, width, _ = img_noisy.shape
    row_ind = ind_initialize(height - kWien + 1, nWien, pWien)
    column_ind = ind_initialize(width - kWien + 1, nWien, pWien)
    kaiser_window = get_kaiserWindow(kWien)

    # The cleaner basic luminance controls matching for all three channels.
    match_table, threshold_count = precompute_BM(
        img_basic[..., 0], kHW=kWien, NHW=NWien,
        nHW=nWien, tauMatch=tauMatch,
    )
    group_len = sum(
        int(threshold_count[i_r, j_r])
        for i_r in row_ind for j_r in column_ind
    )

    numerator = np.zeros((height, width, 3), dtype=np.float64)
    denominator = np.zeros((height - 2 * nWien, width - 2 * nWien), dtype=np.float64)
    denominator = np.pad(denominator, nWien, 'constant', constant_values=1.)
    weight_table = np.zeros((height, width), dtype=np.float64)

    for channel in range(3):
        noisy_patches = image2patches(img_noisy[..., channel], kWien, kWien)
        basic_patches = image2patches(img_basic[..., channel], kWien, kWien)
        transformed_noisy = _forward_transform(noisy_patches, tau_2D)
        transformed_basic = _forward_transform(basic_patches, tau_2D)
        filtered_groups = np.zeros((group_len, kWien, kWien), dtype=np.float64)

        acc_pointer = 0
        for i_r in row_ind:
            for j_r in column_ind:
                group_size = int(threshold_count[i_r, j_r])
                noisy_group = build_3D_group(
                    transformed_noisy,
                    match_table[i_r, j_r],
                    group_size,
                )
                basic_group = build_3D_group(
                    transformed_basic,
                    match_table[i_r, j_r],
                    group_size,
                )
                group, filter_weight = wiener_filtering_hadamard(
                    noisy_group, basic_group, sigma_yuv[channel], not useSD,
                )
                group = group.transpose((2, 0, 1))
                filtered_groups[acc_pointer:acc_pointer + group_size] = group
                acc_pointer += group_size

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
                    numerator[ni:ni + kWien, nj:nj + kWien, channel] += (
                        group[n] * weighted_window
                    )
                    if channel == 0:
                        denominator[ni:ni + kWien, nj:nj + kWien] += weighted_window

    if np.any(denominator <= 0.) or not np.isfinite(denominator).all():
        raise RuntimeError("Color stage-two aggregation left invalid denominator values")
    return numerator / denominator[..., np.newaxis]
