"""Numerical parity and device-residency tests for the tensor backend."""

import cv2
import numpy as np
import pytest
import torch

from bior_2d import bior_2d_forward, bior_2d_reverse
from bm3d import run_bm3d
from bm3d_color import run_bm3d_color
from bm3d_tensor import (
    BM3DConfig,
    HardThresholdConfig,
    RuntimeConfig,
    WienerConfig,
    run_bm3d_tensor,
    run_cbm3d_tensor,
)
from bm3d_tensor.filters import hard_threshold_groups, wiener_filter_groups
from bm3d_tensor.matching import block_match
from bm3d_tensor.transforms import forward_transform, inverse_transform
from dct_2d import dct_2d_forward, dct_2d_reverse
from ht_filtering_hadamard import ht_filtering_hadamard
from precompute_BM import precompute_BM
from utils import ind_initialize, symetrize
from wiener_filtering_hadamard import wiener_filtering_hadamard


def _small_config(wiener_sigma_multiplier=1.0):
    return BM3DConfig(
        hard=HardThresholdConfig(
            search_radius=4,
            patch_size=8,
            max_group_size=8,
            reference_step=4,
            match_threshold=2500,
            transform="BIOR",
            use_sd_weight=False,
            threshold_multiplier=2.7,
        ),
        wiener=WienerConfig(
            search_radius=4,
            patch_size=8,
            max_group_size=8,
            reference_step=4,
            match_threshold=400,
            transform="DCT",
            use_sd_weight=True,
            sigma_multiplier=wiener_sigma_multiplier,
        ),
        runtime=RuntimeConfig(
            reference_chunk_size=32,
            displacement_chunk_size=16,
            group_chunk_size=32,
        ),
    )


@pytest.mark.parametrize(
    ("name", "size", "numpy_forward", "numpy_inverse"),
    (
        ("DCT", 8, dct_2d_forward, dct_2d_reverse),
        ("DCT", 12, dct_2d_forward, dct_2d_reverse),
        ("BIOR", 8, bior_2d_forward, bior_2d_reverse),
    ),
)
def test_tensor_transforms_match_numpy(name, size, numpy_forward, numpy_inverse):
    patches = np.random.default_rng(0).normal(size=(5, size, size))
    tensor = torch.from_numpy(patches)

    actual_forward = forward_transform(tensor, name).numpy()
    actual_reverse = inverse_transform(torch.from_numpy(numpy_forward(patches)), name).numpy()

    np.testing.assert_allclose(actual_forward, numpy_forward(patches), atol=1e-12)
    np.testing.assert_allclose(
        actual_reverse,
        numpy_inverse(numpy_forward(patches)),
        atol=1e-12,
    )


@pytest.mark.parametrize("group_size", (1, 2, 4, 8, 16, 32))
def test_tensor_collaborative_filters_match_numpy(group_size):
    numpy_group = np.random.default_rng(group_size).normal(
        scale=100, size=(8, 8, group_size),
    )
    tensor_group = torch.from_numpy(
        numpy_group.transpose(2, 0, 1).copy(),
    ).unsqueeze(0)

    expected_hard, expected_hard_weight = ht_filtering_hadamard(
        numpy_group, 20, 2.7, True,
    )
    actual_hard, actual_hard_weight = hard_threshold_groups(
        tensor_group, 20, 2.7,
    )
    np.testing.assert_allclose(
        actual_hard[0].permute(1, 2, 0).numpy(), expected_hard, atol=1e-12,
    )
    np.testing.assert_allclose(actual_hard_weight.item(), expected_hard_weight)

    expected_wiener, expected_wiener_weight = wiener_filtering_hadamard(
        numpy_group, numpy_group, 20, True,
    )
    actual_wiener, actual_wiener_weight = wiener_filter_groups(
        tensor_group, tensor_group, 20,
    )
    np.testing.assert_allclose(
        actual_wiener[0].permute(1, 2, 0).numpy(), expected_wiener, atol=1e-12,
    )
    np.testing.assert_allclose(actual_wiener_weight.item(), expected_wiener_weight)


def test_tensor_block_matching_matches_numpy_reference_grid():
    rng = np.random.default_rng(2)
    image = rng.normal(120, 20, size=(16, 17))
    search_radius = 2
    patch_size = 4
    max_group_size = 8
    reference_step = 3
    match_threshold = 800
    padded = symetrize(image, search_radius)

    expected_matches, expected_sizes = precompute_BM(
        padded,
        kHW=patch_size,
        NHW=max_group_size,
        nHW=search_radius,
        tauMatch=match_threshold,
    )
    actual = block_match(
        torch.from_numpy(padded)[None, None],
        patch_size=patch_size,
        max_group_size=max_group_size,
        search_radius=search_radius,
        reference_step=reference_step,
        match_threshold=match_threshold,
        reference_chunk_size=3,
        displacement_chunk_size=4,
    )

    patch_columns = padded.shape[1] - patch_size + 1
    rows = ind_initialize(padded.shape[0] - patch_size + 1, search_radius, reference_step)
    columns = ind_initialize(padded.shape[1] - patch_size + 1, search_radius, reference_step)
    for reference, (row, column) in enumerate(
        (row, column) for row in rows for column in columns
    ):
        group_size = int(expected_sizes[row, column])
        assert int(actual.group_sizes[0, reference]) == group_size
        expected = expected_matches[row, column, :group_size]
        expected = expected[:, 0] * patch_columns + expected[:, 1]
        torch.testing.assert_close(
            actual.indices[0, reference, :group_size],
            torch.from_numpy(expected),
        )


def test_grayscale_tensor_pipeline_matches_legacy_float64():
    noisy = cv2.imread(
        "test_data/sigma20/Cameraman.png", cv2.IMREAD_GRAYSCALE,
    )[:32, :32]
    expected_basic, expected_final = run_bm3d(
        noisy,
        20,
        4, 8, 8, 4, 2500, False, "BIOR", 2.7,
        4, 8, 8, 4, 400, True, "DCT",
    )

    actual_basic, actual_final = run_bm3d_tensor(
        torch.from_numpy(noisy).to(torch.float64), 20, _small_config(),
    )

    np.testing.assert_allclose(actual_basic.numpy(), expected_basic, atol=1e-10)
    np.testing.assert_allclose(actual_final.numpy(), expected_final, atol=1e-10)


def test_reference_patch_capture_preserves_outputs_and_returns_stage_metadata():
    noisy = cv2.imread(
        "test_data/sigma20/Cameraman.png", cv2.IMREAD_GRAYSCALE,
    )[:24, :24]
    tensor_input = torch.from_numpy(noisy).to(torch.float64)
    config = _small_config()

    expected_basic, expected_final = run_bm3d_tensor(tensor_input, 20, config)
    actual_basic, actual_final, references = run_bm3d_tensor(
        tensor_input,
        20,
        config,
        return_reference_patches=True,
    )

    # Merely retaining the streamed reference estimates must not affect either
    # stage output.
    torch.testing.assert_close(actual_basic, expected_basic, rtol=0, atol=0)
    torch.testing.assert_close(actual_final, expected_final, rtol=0, atol=0)

    for stage_data, stage_config in (
        (references.hard, config.hard),
        (references.wiener, config.wiener),
    ):
        reference_count = stage_data.reference_indices.numel()
        assert stage_data.patches.shape == (
            1,
            reference_count,
            1,
            stage_config.patch_size,
            stage_config.patch_size,
        )
        assert stage_data.group_sizes.shape == (1, reference_count)
        assert stage_data.patches.dtype == tensor_input.dtype
        assert stage_data.patches.device == tensor_input.device
        assert stage_data.reference_indices.device == tensor_input.device
        assert stage_data.group_sizes.device == tensor_input.device
        assert bool(torch.isfinite(stage_data.patches).all())

        patch_rows, patch_columns = stage_data.patch_grid_shape
        assert patch_rows * patch_columns >= reference_count
        assert bool((stage_data.reference_indices < patch_rows * patch_columns).all())


def test_color_tensor_pipeline_matches_legacy_float64():
    gray = cv2.imread(
        "test_data/image/Cameraman.png", cv2.IMREAD_GRAYSCALE,
    )[:32, :32].astype(np.float64)
    noisy = np.stack(
        (gray, np.clip(0.85 * gray + 20, 0, 255), np.clip(0.65 * gray + 45, 0, 255)),
        axis=-1,
    )
    expected_basic, expected_final = run_bm3d_color(
        noisy,
        20,
        4, 8, 8, 4, 2500, False, "BIOR", 2.7,
        4, 8, 8, 4, 400, True, "DCT",
    )

    tensor_input = torch.from_numpy(noisy).permute(2, 0, 1)
    actual_basic, actual_final, references = run_cbm3d_tensor(
        tensor_input,
        20,
        _small_config(),
        return_reference_patches=True,
    )

    np.testing.assert_allclose(
        actual_basic.permute(1, 2, 0).numpy(), expected_basic, atol=1e-10,
    )
    np.testing.assert_allclose(
        actual_final.permute(1, 2, 0).numpy(), expected_final, atol=1e-10,
    )
    assert references.hard.patches.shape[0] == 1
    assert references.hard.patches.shape[2:] == (3, 8, 8)
    assert references.wiener.patches.shape[0] == 1
    assert references.wiener.patches.shape[2:] == (3, 8, 8)


def test_wiener_sigma_multiplier_is_an_explicit_algorithm_control():
    group = torch.from_numpy(
        np.random.default_rng(0).normal(scale=30, size=(2, 8, 8)),
    ).to(torch.float64).unsqueeze(0)

    default, _ = wiener_filter_groups(group, group, 20, sigma_multiplier=1.0)
    stronger, _ = wiener_filter_groups(group, group, 20, sigma_multiplier=1.5)

    assert not torch.equal(default, stronger)
    assert stronger.square().sum() < default.square().sum()


def test_float32_pipeline_stays_on_the_selected_device():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    noisy = cv2.imread(
        "test_data/sigma20/Cameraman.png", cv2.IMREAD_GRAYSCALE,
    )[:24, :24]
    for device in devices:
        tensor = torch.from_numpy(noisy).to(device=device, dtype=torch.float32)
        basic, final, references = run_bm3d_tensor(
            tensor,
            20,
            _small_config(),
            return_reference_patches=True,
        )
        assert basic.device.type == device.type
        assert final.device.type == device.type
        assert basic.dtype == torch.float32
        assert final.dtype == torch.float32
        assert bool(torch.isfinite(basic).all())
        assert bool(torch.isfinite(final).all())
        for stage_data in (references.hard, references.wiener):
            assert stage_data.patches.device.type == device.type
            assert stage_data.patches.dtype == torch.float32
            assert bool(torch.isfinite(stage_data.patches).all())
