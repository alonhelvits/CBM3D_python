import cv2
import numpy as np

from bior_2d import bior_2d_forward, bior_2d_reverse
from bm3d import run_bm3d
from dct_2d import dct_2d_forward, dct_2d_reverse
from ht_filtering_hadamard import ht_filtering_hadamard
from psnr import compute_psnr
from utils import add_gaussian_noise
from wiener_filtering_hadamard import wiener_filtering_hadamard


def test_transforms_round_trip():
    patches = np.random.default_rng(0).normal(size=(4, 8, 8))

    dct_reconstructed = dct_2d_reverse(dct_2d_forward(patches))
    bior_reconstructed = bior_2d_reverse(bior_2d_forward(patches))

    np.testing.assert_allclose(dct_reconstructed, patches, atol=1e-10)
    np.testing.assert_allclose(bior_reconstructed, patches, atol=1e-10)


def test_hadamard_filters_return_finite_groups():
    group = np.random.default_rng(0).normal(size=(8, 8, 4))

    hard, hard_weight = ht_filtering_hadamard(group, 20, 2.7, True)
    wiener, wiener_weight = wiener_filtering_hadamard(group, group, 20, True)

    assert hard.shape == group.shape
    assert wiener.shape == group.shape
    assert np.isfinite(hard).all()
    assert np.isfinite(wiener).all()
    assert np.isfinite(hard_weight)
    assert np.isfinite(wiener_weight)


def test_noise_generation_is_reproducible():
    image = np.full((8, 8), 128, dtype=np.uint8)

    first = add_gaussian_noise(image, 20, seed=0)
    second = add_gaussian_noise(image, 20, seed=0)

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8


def test_small_image_end_to_end():
    clean = cv2.imread("test_data/image/Cameraman.png", cv2.IMREAD_GRAYSCALE)[:64, :64]
    noisy = cv2.imread("test_data/sigma20/Cameraman.png", cv2.IMREAD_GRAYSCALE)[:64, :64]

    basic, final = run_bm3d(
        noisy,
        20,
        16,
        8,
        16,
        3,
        2500,
        False,
        "BIOR",
        2.7,
        16,
        8,
        32,
        3,
        400,
        True,
        "DCT",
    )

    assert basic.shape == clean.shape
    assert final.shape == clean.shape
    assert np.isfinite(basic).all()
    assert np.isfinite(final).all()
    assert compute_psnr(clean, basic) > compute_psnr(clean, noisy)
    assert compute_psnr(clean, final) > compute_psnr(clean, basic)
