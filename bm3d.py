import numpy as np

from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


if __name__ == '__main__':
    import os
    import cv2
    # <hyper parameter> -------------------------------------------------------------------------------
    # Stage 1: produce the basic estimate with collaborative hard thresholding.
    n_H = 16          # Block-matching search radius; candidates span (2*n_H + 1)^2 positions.
    k_H = 8           # Patch width/height; run_bm3d selects 8 or 12 based on transform and sigma.
    N_H = 16          # Maximum similar patches per 3D group; should be a power of two for Hadamard.
    p_H = 3           # Step in pixels between reference patches; smaller is slower but overlaps more.
    lambda3D_H = 2.7  # Hard threshold multiplier: threshold = lambda3D_H * sigma * sqrt(group_size).
    useSD_H = False   # True uses inverse group standard deviation instead of nonzero-count weighting.
    tau_2D_H = 'BIOR' # Spatial transform applied to every patch: 'BIOR' or 'DCT'.

    # Stage 2: refine the basic estimate with collaborative Wiener filtering.
    n_W = 16          # Block-matching search radius on the basic estimate.
    k_W = 8           # Patch width/height; also selected inside run_bm3d according to sigma/transform.
    N_W = 32          # Maximum similar patches per Wiener group; should be a power of two.
    p_W = 3           # Step in pixels between Wiener-stage reference patches.
    useSD_W = True    # True weights reconstructed groups by inverse standard deviation.
    tau_2D_W = 'DCT'  # Spatial transform for Wiener-stage patches: 'BIOR' or 'DCT'.
    # <\ hyper parameter> -----------------------------------------------------------------------------

    im_dir = 'test_data/image'
    save_dir = 'temp_test_result'
    os.makedirs(save_dir, exist_ok=True)
    # for im_name in os.listdir(im_dir):
    for im_name in ['Cameraman.png',]:
        # sigma_list = [2, 5, 10, 20, 30, 40, 60, 80, 100]
        sigma_list = [20]  # Assumed Gaussian-noise standard deviations in 0-255 pixel units.
        for sigma in sigma_list:
            print(im_name, '  ', sigma)
            # A candidate matches when its patch SSD is below tauMatch * patch_area.
            tauMatch_H = 2500 if sigma < 35 else 5000  # Stage-1 matching threshold on the noisy image.
            tauMatch_W = 400 if sigma < 35 else 3500   # Stricter stage-2 threshold on the basic estimate.
            noisy_dir = 'test_data/sigma' + str(sigma)

            im_path = os.path.join(im_dir, im_name)
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            noisy_im_path = os.path.join(noisy_dir, im_name)
            noisy_im = cv2.imread(noisy_im_path, cv2.IMREAD_GRAYSCALE)

            im1, im2 = run_bm3d(noisy_im, sigma,
                                n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                                n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

            psnr_1st = compute_psnr(im, im1)
            psnr_2nd = compute_psnr(im, im2)

            im1 = (np.clip(im1, 0, 255)).astype(np.uint8)
            im2 = (np.clip(im2, 0, 255)).astype(np.uint8)

            save_name = im_name[:-4] + '_s' + str(sigma) + '_py_1st_P' + '%.4f' % psnr_1st + '.png'
            cv2.imwrite(os.path.join(save_dir, save_name), im1)
            save_name = im_name[:-4] + '_s' + str(sigma) + '_py_2nd_P' + '%.4f' % psnr_2nd + '.png'
            cv2.imwrite(os.path.join(save_dir, save_name), im2)
