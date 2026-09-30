"""Chunked, reference-grid-only block matching.

Images arrive as ``[B,1,H,W]``. All overlapping patches are represented as
``[B,L,K*K]``. The result contains at most ``Nmax`` linear patch indices for
each of the ``R`` references: ``[B,R,Nmax]``. Linear indices address the
``(H-K+1) x (W-K+1)`` patch grid, not image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MatchResult:
    """Patch matches and group sizes for every image/reference pair.

    Attributes:
        indices: Linear patch-grid positions, shape ``[B, R, Nmax]``.
        group_sizes: Thresholded powers of two, shape ``[B, R]``.
        reference_indices: Shared reference-grid positions, shape ``[R]``.
        patch_grid_shape: Spatial shape whose flattened size is ``L``.
    """

    indices: torch.Tensor
    group_sizes: torch.Tensor
    reference_indices: torch.Tensor
    patch_grid_shape: tuple[int, int]


def reference_axis_indices(
    patch_grid_size: int,
    search_radius: int,
    step: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Reproduce the legacy stepped reference coordinates for one axis."""
    stop = patch_grid_size - search_radius
    if stop <= search_radius:
        raise ValueError("image is too small for the requested search radius and patch size")
    indices = torch.arange(search_radius, stop, step, device=device, dtype=torch.long)
    final = stop - 1
    if indices.numel() == 0:
        indices = torch.tensor([final], device=device, dtype=torch.long)
    elif int(indices[-1]) < final:
        indices = torch.cat((indices, indices.new_tensor([final])))
    return indices


def _power_of_two_group_sizes(counts: torch.Tensor, maximum: int) -> torch.Tensor:
    """Round threshold counts down to a supported Hadamard group size."""
    counts = counts.clamp(min=1, max=maximum)
    result = torch.ones_like(counts)
    size = 2
    while size <= maximum:
        result = torch.where(counts >= size, size, result)
        size *= 2
    return result


def block_match(
    image: torch.Tensor,
    *,
    patch_size: int,
    max_group_size: int,
    search_radius: int,
    reference_step: int,
    match_threshold: float,
    reference_chunk_size: int = 256,
    displacement_chunk_size: int = 64,
) -> MatchResult:
    """Find similar patches for a single matching channel.

    Args:
        image: Padded matching image, shape ``[B, 1, H, W]``.
        patch_size: Patch width/height ``K``.
        max_group_size: Number of retained candidates ``Nmax``.
        search_radius: Candidate radius; there are
            ``D=(2*search_radius+1)^2`` displacements.
        reference_step: Distance between reference patch top-left positions.
        match_threshold: Per-pixel SSD threshold. The comparison uses
            ``match_threshold*K*K`` exactly as the legacy implementation.
        reference_chunk_size: Maximum ``Rc`` references processed together.
        displacement_chunk_size: Maximum ``Dc`` offsets processed together.

    Returns:
        A :class:`MatchResult` containing ``[B,R,Nmax]`` match indices and
        ``[B,R]`` power-of-two group sizes.

    The largest temporary gather has shape ``[B,Rc,Dc,K*K]``. A running
    ``topk`` reduces each displacement chunk to ``[B,Rc,Nmax]`` so a full
    ``[B,R,D]`` distance table is never materialized.
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("block_match expects a BCHW tensor with one channel")
    if max_group_size > (2 * search_radius + 1) ** 2:
        raise ValueError("max_group_size exceeds the number of search candidates")

    batch, _, height, width = image.shape
    patch_rows = height - patch_size + 1
    patch_columns = width - patch_size + 1
    if patch_rows <= 0 or patch_columns <= 0:
        raise ValueError("image is smaller than patch_size")

    row_indices = reference_axis_indices(
        patch_rows, search_radius, reference_step, device=image.device,
    )
    column_indices = reference_axis_indices(
        patch_columns, search_radius, reference_step, device=image.device,
    )
    ref_rows, ref_columns = torch.meshgrid(row_indices, column_indices, indexing="ij")
    ref_rows = ref_rows.flatten()
    ref_columns = ref_columns.flatten()
    reference_indices = ref_rows * patch_columns + ref_columns

    offsets = torch.arange(
        -search_radius, search_radius + 1, device=image.device, dtype=torch.long,
    )
    offset_rows, offset_columns = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_rows = offset_rows.flatten()
    offset_columns = offset_columns.flatten()
    center_offset = search_radius * (2 * search_radius + 1) + search_radius

    # F.unfold: [B,K*K,L]. Transposing makes linear patch lookup the second
    # axis, which lets one advanced-index operation gather all candidates.
    patches = F.unfold(image, kernel_size=patch_size, stride=1).transpose(1, 2)
    threshold = float(match_threshold) * patch_size * patch_size
    all_matches: list[torch.Tensor] = []
    all_sizes: list[torch.Tensor] = []

    for ref_start in range(0, reference_indices.numel(), reference_chunk_size):
        ref_stop = min(ref_start + reference_chunk_size, reference_indices.numel())
        chunk_rows = ref_rows[ref_start:ref_stop]
        chunk_columns = ref_columns[ref_start:ref_stop]
        chunk_reference_indices = reference_indices[ref_start:ref_stop]
        # [B,Rc,K*K] -> [B,Rc,1,K*K], ready to broadcast over Dc candidates.
        reference_patches = patches[:, chunk_reference_indices, :].unsqueeze(2)
        ref_count = chunk_reference_indices.numel()

        best_distances = torch.full(
            (batch, ref_count, max_group_size),
            torch.inf,
            device=image.device,
            dtype=image.dtype,
        )
        best_indices = chunk_reference_indices.view(1, ref_count, 1).expand(
            batch, -1, max_group_size,
        ).clone()
        threshold_counts = torch.zeros(
            (batch, ref_count), device=image.device, dtype=torch.long,
        )

        for displacement_start in range(
            0, offset_rows.numel(), displacement_chunk_size,
        ):
            displacement_stop = min(
                displacement_start + displacement_chunk_size, offset_rows.numel(),
            )
            dy = offset_rows[displacement_start:displacement_stop]
            dx = offset_columns[displacement_start:displacement_stop]
            candidate_rows = chunk_rows[:, None] + dy[None, :]
            candidate_columns = chunk_columns[:, None] + dx[None, :]
            candidate_indices = candidate_rows * patch_columns + candidate_columns
            # candidate_indices [Rc,Dc] gathers [B,Rc,Dc,K*K].
            candidate_patches = patches[:, candidate_indices, :]
            distances = (candidate_patches - reference_patches).square().sum(dim=-1)
            threshold_counts += (distances < threshold).sum(dim=-1)

            # Make self-match deterministic even for constant images with many ties.
            if displacement_start <= center_offset < displacement_stop:
                distances = distances.clone()
                distances[..., center_offset - displacement_start] = -1.0

            candidate_indices = candidate_indices.unsqueeze(0).expand(batch, -1, -1)
            # Merge the previous Nmax winners with this Dc-sized chunk, then
            # immediately discard all but the new Nmax winners.
            combined_distances = torch.cat((best_distances, distances), dim=-1)
            combined_indices = torch.cat((best_indices, candidate_indices), dim=-1)
            best_distances, order = torch.topk(
                combined_distances,
                k=max_group_size,
                dim=-1,
                largest=False,
                sorted=True,
            )
            best_indices = torch.gather(combined_indices, -1, order)

        all_matches.append(best_indices)
        all_sizes.append(_power_of_two_group_sizes(threshold_counts, max_group_size))

    return MatchResult(
        indices=torch.cat(all_matches, dim=1),
        group_sizes=torch.cat(all_sizes, dim=1),
        reference_indices=reference_indices,
        patch_grid_shape=(patch_rows, patch_columns),
    )
