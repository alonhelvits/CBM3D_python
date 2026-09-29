import cv2
import numpy as np

from bm3d_color import run_bm3d_color
from color_utils import YUV_NOISE_SCALES, rgb_sigma_to_yuv, rgb_to_yuv, yuv_to_rgb
from psnr import compute_psnr


def test_color_transform_round_trip():
    rgb = np.random.default_rng(0).uniform(0., 255., size=(8, 9, 3))

    reconstructed = yuv_to_rgb(rgb_to_yuv(rgb))

    np.testing.assert_allclose(reconstructed, rgb, atol=1e-10)


def test_rgb_sigma_is_scaled_by_color_matrix_row_norms():
    sigma_yuv = rgb_sigma_to_yuv(20)

    np.testing.assert_allclose(
        sigma_yuv,
        [13.37110317, 10.86615547, 16.16714817],
        atol=1e-8,
    )
    np.testing.assert_allclose(sigma_yuv, 20 * YUV_NOISE_SCALES)


def test_small_color_image_end_to_end():
    gray = cv2.imread(
        "test_data/image/Cameraman.png", cv2.IMREAD_GRAYSCALE,
    )[:64, :64].astype(np.float64)
    clean = np.stack([
        gray,
        np.clip(0.85 * gray + 20., 0., 255.),
        np.clip(0.65 * gray + 45., 0., 255.),
    ], axis=-1)

    sigma = 20
    noise = np.random.default_rng(0).normal(0., sigma, size=clean.shape)
    noisy = np.clip(clean + noise, 0., 255.)

    basic, final = run_bm3d_color(
        noisy,
        sigma,
        8,
        8,
        8,
        4,
        2500,
        False,
        "BIOR",
        2.7,
        8,
        8,
        16,
        4,
        400,
        True,
        "DCT",
    )

    assert basic.shape == clean.shape
    assert final.shape == clean.shape
    assert np.isfinite(basic).all()
    assert np.isfinite(final).all()
    assert np.all((0. <= basic) & (basic <= 255.))
    assert np.all((0. <= final) & (final <= 255.))
    assert compute_psnr(clean, basic) > compute_psnr(clean, noisy)
    assert compute_psnr(clean, final) > compute_psnr(clean, noisy)
    assert compute_psnr(clean, final) > compute_psnr(clean, basic)


def test_constant_color_image_stays_finite():
    black = np.zeros((40, 40, 3), dtype=np.float64)

    basic, final = run_bm3d_color(
        black,
        20,
        8,
        8,
        8,
        4,
        2500,
        False,
        "BIOR",
        2.7,
        8,
        8,
        8,
        4,
        400,
        True,
        "DCT",
    )

    np.testing.assert_array_equal(basic, black)
    np.testing.assert_array_equal(final, black)
