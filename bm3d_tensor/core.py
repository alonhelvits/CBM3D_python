"""End-to-end tensor BM3D orchestration.

Shape convention:
    ``B`` batch, ``C`` channels, ``H/W`` padded image dimensions, ``K`` patch
    size, ``L`` patch positions, ``R`` reference patches, ``G`` similar
    patches in one group, and ``M`` groups in a processing chunk.

The stage engine deliberately loops only over the small channel count, the
power-of-two group-size buckets, and bounded chunks. Matching, transforms,
collaborative filtering, and aggregation within each chunk are tensorized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import BM3DConfig, HardThresholdConfig, WienerConfig, validate_config
from .filters import hard_threshold_groups, sd_weight, wiener_filter_groups
from .matching import MatchResult, block_match
from .transforms import inverse_transform, transformed_patch_bank


RGB_TO_YUV = (
    (0.299, 0.587, 0.114),
    (-0.14713, -0.28886, 0.436),
    (0.615, -0.51499, -0.10001),
)
YUV_TO_RGB = tuple(
    tuple(row)
    for row in np.linalg.inv(np.asarray(RGB_TO_YUV, dtype=np.float64)).tolist()
)


@dataclass(frozen=True)
class _InputLayout:
    """Original public rank, used to restore the API's output layout."""

    dimensions: int
    channels: int


def _prepare_image(image: torch.Tensor, channels: int) -> tuple[torch.Tensor, _InputLayout]:
    """Validate a public input and normalize it to floating ``[B,C,H,W]``."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    dimensions = image.ndim
    if channels == 1 and dimensions == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif dimensions == 3 and image.shape[0] == channels:
        image = image.unsqueeze(0)
    elif dimensions == 4 and image.shape[1] == channels:
        pass
    else:
        expected = "HW, CHW, or BCHW" if channels == 1 else "CHW or BCHW"
        raise ValueError(f"expected a {expected} tensor with {channels} channel(s)")

    if image.shape[-2] == 0 or image.shape[-1] == 0:
        raise ValueError("image must not be empty")
    if not image.is_floating_point():
        image = image.to(torch.float32)
    elif image.dtype not in (torch.float32, torch.float64):
        image = image.to(torch.float32)
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image must contain only finite values")
    return image, _InputLayout(dimensions=dimensions, channels=channels)


def _restore_image(image: torch.Tensor, layout: _InputLayout) -> torch.Tensor:
    """Undo only the singleton dimensions inserted by :func:`_prepare_image`."""
    if layout.dimensions == 2:
        return image[0, 0]
    if layout.dimensions == 3:
        return image[0]
    return image


def _crop_padding(image: torch.Tensor, amount: int) -> torch.Tensor:
    """Remove equal spatial padding while handling the zero-padding case."""
    if amount == 0:
        return image
    return image[..., amount:-amount, amount:-amount]


def symmetric_pad(image: torch.Tensor, amount: int) -> torch.Tensor:
    """Match ``numpy.pad(..., mode='symmetric')`` for BCHW tensors."""
    if amount == 0:
        return image

    def symmetric_indices(length: int) -> torch.Tensor:
        raw = torch.arange(
            -amount, length + amount, device=image.device, dtype=torch.long,
        )
        folded = torch.remainder(raw, 2 * length)
        return torch.where(folded < length, folded, 2 * length - 1 - folded)

    rows = symmetric_indices(image.shape[-2])
    columns = symmetric_indices(image.shape[-1])
    return image.index_select(-2, rows).index_select(-1, columns)


def _kaiser_window(size: int, reference: torch.Tensor) -> torch.Tensor:
    """Create the legacy ``[K,K]`` Kaiser window on a tensor's device/dtype."""
    values = np.kaiser(size, 2.0)
    window = np.outer(values, values)
    return torch.as_tensor(window, device=reference.device, dtype=reference.dtype)


def _matching(
    match_image: torch.Tensor,
    stage: HardThresholdConfig | WienerConfig,
    config: BM3DConfig,
) -> MatchResult:
    """Dispatch block matching using a stage's geometry and runtime chunks."""
    runtime = config.runtime
    return block_match(
        match_image,
        patch_size=stage.patch_size,
        max_group_size=stage.max_group_size,
        search_radius=stage.search_radius,
        reference_step=stage.reference_step,
        match_threshold=stage.match_threshold,
        reference_chunk_size=runtime.reference_chunk_size,
        displacement_chunk_size=runtime.displacement_chunk_size,
    )


def _aggregate_groups(
    accumulator: torch.Tensor,
    denominator: torch.Tensor | None,
    patches: torch.Tensor,
    local_patch_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    weights: torch.Tensor,
    kaiser: torch.Tensor,
    *,
    image_width: int,
    patch_columns: int,
) -> None:
    """Overlap-add one ``[M,G,K,K]`` patch chunk into ``[B,H,W]`` buffers.

    ``local_patch_indices`` is ``[M,G]`` and contains top-left locations in
    the flattened patch grid. Expanding each location by ``K*K`` pixel offsets
    creates ``[M*G,K*K]`` destination and value tensors for one scatter-add.
    """
    group_count, group_size, patch_size, _ = patches.shape
    # Collapse M and G because every member patch aggregates independently.
    member_indices = local_patch_indices.reshape(group_count * group_size)
    member_batches = batch_indices[:, None].expand(-1, group_size).reshape(-1)
    rows = torch.div(member_indices, patch_columns, rounding_mode="floor")
    columns = torch.remainder(member_indices, patch_columns)
    image_area = accumulator.shape[-2] * accumulator.shape[-1]
    bases = member_batches * image_area + rows * image_width + columns

    pixel_rows = torch.arange(patch_size, device=patches.device, dtype=torch.long)
    pixel_columns = torch.arange(patch_size, device=patches.device, dtype=torch.long)
    offset_rows, offset_columns = torch.meshgrid(pixel_rows, pixel_columns, indexing="ij")
    pixel_offsets = (offset_rows * image_width + offset_columns).flatten()
    # [M*G,1] + [1,K*K] -> one destination per reconstructed patch pixel.
    destinations = bases[:, None] + pixel_offsets[None, :]

    member_weights = weights[:, None].expand(-1, group_size).reshape(-1, 1)
    window = kaiser.flatten().unsqueeze(0)
    values = patches.reshape(group_count * group_size, -1) * member_weights * window
    accumulator.reshape(-1).scatter_add_(0, destinations.flatten(), values.flatten())

    if denominator is not None:
        denominator_values = (member_weights * window).expand_as(values)
        denominator.reshape(-1).scatter_add_(
            0, destinations.flatten(), denominator_values.flatten(),
        )


def _run_stage(
    noisy: torch.Tensor,
    sigma_channels: torch.Tensor,
    config: BM3DConfig,
    *,
    basic: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one complete hard or Wiener stage on padded ``[B,C,H,W]`` images.

    Dimension names used below:

    - ``B``: images in the batch.
    - ``C``: channels (1 for grayscale, 3 for YUV color).
    - ``H,W``: padded image height and width.
    - ``K``: square patch size.
    - ``L``: all valid overlapping patch positions.
    - ``R``: reference-patch positions evaluated by block matching.
    - ``Nmax``: maximum number of similar patches retained per reference.
    - ``G``: actual power-of-two group size for one bucket.
    - ``M``: groups in the current processing chunk.

    Args:
        noisy: Padded noisy image, shape ``[B,C,H,W]``.
        sigma_channels: Noise sigma for each channel, shape ``[C]``. For color
            these are the RGB sigma transformed into Y, U, and V sigmas.
        config: Hard-stage, Wiener-stage, and runtime chunk configuration.
        basic: Padded basic estimate with shape ``[B,C,H,W]``. ``None`` means
            run hard thresholding; a tensor means run Wiener filtering.

    Returns:
        The padded stage estimate with shape ``[B,C,H,W]``.

    Matching yields ``[B,R,Nmax]`` indices and ``[B,R]`` group sizes. Each
    group-size bucket is gathered as ``[M,G,K,K]``, filtered, reconstructed,
    and immediately aggregated so a global table of filtered groups is never
    allocated.
    """
    # Stage 1 has no basic estimate and filters groups with hard thresholding.
    # Stage 2 receives the stage-1 estimate and uses it to calculate both
    # matches and Wiener gains.
    is_wiener = basic is not None
    stage: HardThresholdConfig | WienerConfig = config.wiener if is_wiener else config.hard

    # BM3D matches only one channel: grayscale for C=1 or luminance Y for C=3.
    # Stage 1 matches noisy Y; stage 2 matches the cleaner basic-estimate Y.
    # match_image: [B,1,H,W].
    match_image = basic[:, :1] if basic is not None else noisy[:, :1]

    # matches.indices:     [B,R,Nmax], linear indices into the KxK patch grid.
    # matches.group_sizes: [B,R], actual G selected for every reference.
    matches = _matching(match_image, stage, config)

    # Image and patch-grid geometry. The patch grid has
    # patch_rows=(H-K+1), patch_columns=(W-K+1), and L positions in total.
    batch, channels, height, width = noisy.shape
    patch_size = stage.patch_size
    patch_rows, patch_columns = matches.patch_grid_shape
    patch_count = patch_rows * patch_columns
    reference_count = matches.indices.shape[1]
    max_group_size = stage.max_group_size

    # kaiser: [K,K]. Every reconstructed patch is multiplied by this window
    # before overlap-add aggregation to reduce patch-boundary artifacts.
    kaiser = _kaiser_window(patch_size, noisy)

    # Match indices are local to each image's L-position patch grid. Offsetting
    # image b by b*L converts them into indices for a flattened [B*L,K,K] bank.
    # batch_offsets: [B,1,1], broadcasting over R and Nmax.
    batch_offsets = (
        torch.arange(batch, device=noisy.device, dtype=torch.long)[:, None, None]
        * patch_count
    )

    # global_matches: [B*R,Nmax], used to gather transformed patch values.
    global_matches = (matches.indices + batch_offsets).reshape(
        batch * reference_count, max_group_size,
    )

    # local_matches: [B*R,Nmax], still local to one image. Aggregation needs
    # these values to recover each patch's top-left (row,column) position.
    local_matches = matches.indices.reshape(batch * reference_count, max_group_size)

    # Flatten the B and R axes because subsequent bucketing treats every
    # image/reference pair as an independent group.
    # group_sizes: [B*R].
    group_sizes = matches.group_sizes.reshape(-1)

    # One scalar aggregation weight per image/reference group: [B*R].
    # Channel 0 calculates these weights; color channels 1 and 2 reuse them.
    reference_weights = torch.empty(
        batch * reference_count, device=noisy.device, dtype=noisy.dtype,
    )

    # Maps each flattened reference back to its image: [B*R]. This prevents
    # scatter aggregation from mixing members of different batch images.
    reference_batches = torch.arange(
        batch, device=noisy.device, dtype=torch.long,
    ).repeat_interleave(reference_count)

    # numerator:   [B,C,H,W], weighted reconstructed-patch sums.
    # denominator: [B,H,W], shared sums of aggregation weights/windows.
    # Color uses a shared denominator because all channels reuse Y's matches
    # and group weights.
    numerator = torch.zeros_like(noisy)
    denominator = torch.zeros(
        (batch, height, width), device=noisy.device, dtype=noisy.dtype,
    )

    # Hadamard transforms require power-of-two group lengths. References with
    # different G values are processed in separate dense buckets.
    # Example for Nmax=32: [1,2,4,8,16,32].
    group_size_values: list[int] = []
    group_size = 1
    while group_size <= max_group_size:
        group_size_values.append(group_size)
        group_size *= 2

    # Process channels sequentially to avoid keeping C full patch banks in
    # memory simultaneously. Channel 0 is grayscale/Y and therefore runs first
    # to populate reference_weights and the shared denominator.
    for channel in range(channels):
        # Input slice: [B,1,H,W]. transformed_patch_bank first extracts every
        # overlapping patch, then applies DCT/BIOR:
        #     [B,1,H,W] -> [B,1,L,K,K] -> [B*L,K,K].
        noisy_bank = transformed_patch_bank(
            noisy[:, channel:channel + 1], patch_size, stage.transform,
        )[:, 0].reshape(batch * patch_count, patch_size, patch_size)

        if is_wiener:
            assert basic is not None
            # The Wiener stage needs aligned noisy and basic coefficient banks,
            # each [B*L,K,K]. The basic bank determines the Wiener gain.
            basic_bank = transformed_patch_bank(
                basic[:, channel:channel + 1], patch_size, stage.transform,
            )[:, 0].reshape(batch * patch_count, patch_size, patch_size)
        else:
            basic_bank = None

        # Accumulate one output channel at a time: [B,H,W]. It is copied into
        # numerator[:,channel] only after every group-size bucket is complete.
        channel_accumulator = torch.zeros(
            (batch, height, width), device=noisy.device, dtype=noisy.dtype,
        )

        # Variable group lengths cannot form one dense tensor. Bucketing by G
        # produces dense [M,G,K,K] batches for Hadamard matrix multiplication.
        for size in group_size_values:
            # selected: flattened reference indices whose actual group size is
            # this bucket's G. Its length is the total groups in the bucket.
            selected = torch.nonzero(group_sizes == size, as_tuple=False).flatten()

            # Bound peak gather/filter/scatter memory by processing at most M
            # groups at once. Changing group_chunk_size does not change BM3D's
            # mathematical parameters.
            for start in range(0, selected.numel(), config.runtime.group_chunk_size):
                # chosen: [M], indices into the flattened B*R reference axis.
                chosen = selected[start:start + config.runtime.group_chunk_size]

                # Keep only the first G valid members of each reference group.
                # selected_global_matches: [M,G].
                selected_global_matches = global_matches[chosen, :size]

                # Advanced indexing gathers one transformed patch per match:
                # [B*L,K,K] indexed by [M,G] -> [M,G,K,K].
                noisy_groups = noisy_bank[selected_global_matches]

                if is_wiener:
                    assert basic_bank is not None
                    # basic_groups is aligned element-for-element with
                    # noisy_groups and also has shape [M,G,K,K].
                    basic_groups = basic_bank[selected_global_matches]

                    # filtered: [M,G,K,K] transformed-domain estimates.
                    # filter_weights: [M], one legacy Wiener weight per group.
                    filtered, filter_weights = wiener_filter_groups(
                        noisy_groups,
                        basic_groups,
                        sigma_channels[channel],
                        sigma_multiplier=config.wiener.sigma_multiplier,
                    )
                else:
                    # Hard thresholding has the same input/output shapes as the
                    # Wiener filter: groups [M,G,K,K], weights [M].
                    filtered, filter_weights = hard_threshold_groups(
                        noisy_groups,
                        sigma_channels[channel],
                        config.hard.threshold_multiplier,
                    )

                if channel == 0:
                    # Weight mode is configured per stage. Store Y/grayscale's
                    # [M] weights at their B*R positions for later U/V reuse.
                    weights = sd_weight(filtered) if stage.use_sd_weight else filter_weights
                    reference_weights[chosen] = weights
                else:
                    # U and V deliberately share Y's scalar group weights.
                    # weights: [M].
                    weights = reference_weights[chosen]

                # Invert only the per-patch 2D transform; the collaborative
                # filter already inverted its Hadamard transform over G.
                # Shape remains [M,G,K,K], but values are spatial patch pixels.
                spatial_patches = inverse_transform(filtered, stage.transform)

                # Overlap-add these M*G patches immediately. local match
                # positions are [M,G]; batch ids and weights are both [M].
                # Only channel 0 updates denominator [B,H,W].
                _aggregate_groups(
                    channel_accumulator,
                    denominator if channel == 0 else None,
                    spatial_patches,
                    local_matches[chosen, :size],
                    reference_batches[chosen],
                    weights,
                    kaiser,
                    image_width=width,
                    patch_columns=patch_columns,
                )

        # channel_accumulator [B,H,W] -> its slot in [B,C,H,W].
        numerator[:, channel] = channel_accumulator

    # Pixels outside the covered reference region can have denominator zero in
    # the padded border. They will be cropped by the caller, but replacing zero
    # with one here prevents NaNs. Broadcasting [B,H,W] as [B,1,H,W] divides
    # every channel by the same aggregation denominator.
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator),
    )
    return numerator / safe_denominator.unsqueeze(1)


def _validate_spatial_shape(image: torch.Tensor, config: BM3DConfig) -> None:
    height, width = image.shape[-2:]
    for name, stage in (("hard", config.hard), ("wiener", config.wiener)):
        if height < stage.patch_size or width < stage.patch_size:
            raise ValueError(f"image is smaller than the {name} patch size")


def run_bm3d_tensor(
    noisy: torch.Tensor,
    sigma: float,
    config: BM3DConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise a grayscale tensor while keeping all work on its device.

    Args:
        noisy: ``[H,W]``, ``[1,H,W]``, or ``[B,1,H,W]`` tensor.
        sigma: Gaussian-noise standard deviation in 0--255 pixel units.
        config: Explicit stage/runtime settings, or classic defaults.

    Returns:
        ``(basic, final)`` tensors with the input's rank and device. Float32
        and float64 are preserved; other input dtypes are promoted to float32.
    """
    image, layout = _prepare_image(noisy, channels=1)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive number")
    config = config or BM3DConfig.classic(sigma)
    validate_config(config)
    _validate_spatial_shape(image, config)
    sigma_channels = image.new_tensor([sigma])

    hard_radius = config.hard.search_radius
    noisy_hard = symmetric_pad(image, hard_radius)
    basic_padded = _run_stage(noisy_hard, sigma_channels, config)
    basic = _crop_padding(basic_padded, hard_radius)

    wiener_radius = config.wiener.search_radius
    noisy_wiener = symmetric_pad(image, wiener_radius)
    basic_wiener = symmetric_pad(basic, wiener_radius)
    final_padded = _run_stage(
        noisy_wiener, sigma_channels, config, basic=basic_wiener,
    )
    final = _crop_padding(final_padded, wiener_radius)
    return _restore_image(basic, layout), _restore_image(final, layout)


def run_cbm3d_tensor(
    noisy_rgb: torch.Tensor,
    sigma: float,
    config: BM3DConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denoise an RGB tensor while keeping all work on its device.

    Args:
        noisy_rgb: RGB ``[3,H,W]`` or ``[B,3,H,W]`` tensor in ``[0,255]``.
        sigma: Per-RGB-channel Gaussian-noise standard deviation.
        config: Explicit stage/runtime settings, or classic color defaults.

    Returns:
        Clipped RGB ``(basic, final)`` tensors with the input's rank and
        device. Float32/float64 are preserved; other dtypes become float32.
        Internally the pipeline uses signed, full-resolution YUV, Y-only
        matching, per-channel sigma, and Y-derived shared weights.
    """
    rgb, layout = _prepare_image(noisy_rgb, channels=3)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive number")
    if bool((rgb < 0).any()) or bool((rgb > 255).any()):
        raise ValueError("RGB values must be in the range [0, 255]")
    config = config or BM3DConfig.classic(sigma, color=True)
    validate_config(config)
    _validate_spatial_shape(rgb, config)

    color_matrix = rgb.new_tensor(RGB_TO_YUV)
    inverse_color_matrix = rgb.new_tensor(YUV_TO_RGB)
    # [3,3] x [B,3,H,W] -> [B,3,H,W]. Row norms propagate independent RGB
    # noise into the marginal Y/U/V sigmas used by collaborative filtering.
    yuv = torch.einsum("oc,bchw->bohw", color_matrix, rgb)
    sigma_channels = sigma * torch.linalg.vector_norm(color_matrix, dim=1)

    hard_radius = config.hard.search_radius
    noisy_hard = symmetric_pad(yuv, hard_radius)
    basic_padded = _run_stage(noisy_hard, sigma_channels, config)
    basic_yuv = _crop_padding(basic_padded, hard_radius)

    wiener_radius = config.wiener.search_radius
    noisy_wiener = symmetric_pad(yuv, wiener_radius)
    basic_wiener = symmetric_pad(basic_yuv, wiener_radius)
    final_padded = _run_stage(
        noisy_wiener, sigma_channels, config, basic=basic_wiener,
    )
    final_yuv = _crop_padding(final_padded, wiener_radius)

    basic_rgb = torch.einsum("oc,bchw->bohw", inverse_color_matrix, basic_yuv).clamp(0, 255)
    final_rgb = torch.einsum("oc,bchw->bohw", inverse_color_matrix, final_yuv).clamp(0, 255)
    return _restore_image(basic_rgb, layout), _restore_image(final_rgb, layout)
