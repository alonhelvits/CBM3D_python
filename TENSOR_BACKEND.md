# Tensor BM3D backend

`bm3d_tensor` is an independent PyTorch implementation of the same algorithm
as the original NumPy/SciPy/PyWavelets code. The legacy implementation remains
available as a readable reference and as the numerical oracle used by
`tests/test_bm3d_tensor.py`.

The tensor backend supports CPU, CUDA, and Apple MPS. Inputs, intermediate
values, and outputs stay on the input tensor's device; there are no transfers
inside the hot path.

## Tensor dimension notation

The implementation uses the following symbols consistently:

| Symbol | Meaning |
| --- | --- |
| `B` | Number of images in the batch |
| `C` | Channels: `1` for grayscale or `3` for YUV color |
| `H`, `W` | Padded image height and width |
| `K` | Patch width and height |
| `L` | Number of overlapping patches, `(H-K+1)*(W-K+1)` |
| `R` | Number of reference patches selected by `reference_step` |
| `D` | Search displacements, `(2*search_radius+1)^2` |
| `G` | Actual group size; one of `1, 2, 4, ..., max_group_size` |
| `M` | Number of same-size groups in the current processing chunk |

Public tensors use channel-first layouts:

- Grayscale: `HW`, `1HW`, or `B1HW`.
- Color: `3HW` or `B3HW`, in RGB order.
- Results have the same rank/layout and device as their input.
- Float32 and float64 inputs preserve dtype; other inputs become float32.

Internally, images are normalized to `BCHW`. Color images are converted to
signed, full-resolution YUV. Matching is performed only on Y, while the match
locations and Y-derived aggregation weights are reused for U and V.

## Algorithm flow and critical shapes

Each hard-threshold or Wiener stage follows the same tensor pipeline:

1. Symmetrically pad the image on the spatial axes: `B,C,H,W`.
2. Extract every overlapping patch with `unfold`:
   `B,C*K*K,L`, then reshape it to `B,C,L,K,K`.
3. Match only the `R` reference patches on luminance. The matcher returns:
   - linear patch indices: `B,R,Nmax`;
   - power-of-two group sizes: `B,R`.
4. Bucket references by group size `G`. This turns variable-length groups into
   dense batches with shape `M,G,K,K`.
5. Apply the 2D patch transform, the Hadamard transform along `G`, and the
   hard-threshold or Wiener operation in parallel.
6. Invert the patch transform and overlap-add the patches using `scatter_add_`
   into `B,H,W` numerator and denominator images.
7. Divide the accumulators and remove the symmetric padding.

### Block matching

The legacy matcher materializes distances for every padded-image position.
The tensor matcher stores patches as `B,L,K*K`, evaluates only the reference
grid, and processes two independent chunk dimensions:

- a reference chunk with at most `Rc` references;
- a displacement chunk with at most `Dc` candidates.

The temporary candidate tensor is therefore `B,Rc,Dc,K*K`, instead of a
full-image distance table. After each displacement chunk, a running `topk`
keeps only `Nmax` candidates with shapes `B,Rc,Nmax`. Threshold counts are
rounded down to a power of two to preserve the Hadamard group requirement.
The reference patch is explicitly forced to the first position, including for
constant images where all distances tie.

### Collaborative filters

After bucketing, a group tensor is `M,G,K,K`. It is permuted to `M,K,K,G` so
the cached `G,G` Hadamard matrix operates on the similar-patch axis.

For hard thresholding, coefficients below
`threshold_multiplier * sigma * sqrt(G)` are removed. For Wiener filtering,
the gain is

```text
power = basic_hadamard**2 / G
gain  = power / (power + effective_sigma**2)
```

The inverse Hadamard transform returns `M,G,K,K`. DCT matrices and exact BIOR
linear operators are cached per `(patch size, device, dtype)`.

### Aggregation

For every chunk, the `M*G` filtered patches are flattened to
`M*G,K*K`. Their linear top-left locations are expanded by the `K*K` pixel
offsets, producing scatter destinations of the same shape. Kaiser-windowed
patches are accumulated into the numerator, and their scalar group weights are
accumulated into the denominator. This replaces Python loops over reference
patches, group members, and patch pixels.

Channels are processed sequentially. This keeps the peak patch-bank memory
close to the grayscale case while still running every large operation in
parallel on the selected device.

## Basic use

```python
import torch

from bm3d_tensor import BM3DConfig, run_cbm3d_tensor

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

noisy = noisy_rgb_chw.to(device=device, dtype=torch.float32)
config = BM3DConfig.classic(sigma=20, color=True)

with torch.inference_mode():
    basic, final = run_cbm3d_tensor(noisy, sigma=20, config=config)
```

Use `run_bm3d_tensor` for grayscale tensors. Float32 is recommended for CUDA
and MPS. Float64 CPU execution is useful when comparing against the legacy
implementation.

## Changing Wiener strength

Algorithm settings are immutable dataclasses so experiments are explicit and
reproducible:

```python
from dataclasses import replace

config = BM3DConfig.classic(sigma=20, color=True)
config = replace(
    config,
    wiener=replace(config.wiener, sigma_multiplier=1.25),
)
```

`sigma_multiplier` produces
`effective_sigma = channel_sigma * sigma_multiplier` in the Wiener gain and
legacy aggregation weight. It does not alter block matching, the hard stage,
or RGB-to-YUV noise conversion.

## Memory and performance controls

`RuntimeConfig` exposes three controls that do not change algorithm logic:

| Setting | Tensor dimension controlled | Tradeoff |
| --- | --- | --- |
| `reference_chunk_size` | `Rc` in matching | Larger values reduce launches but use more memory |
| `displacement_chunk_size` | `Dc` in matching | Larger values reduce `topk` updates but use more memory |
| `group_chunk_size` | `M` in filtering | Larger values batch more groups but enlarge gather/scatter tensors |

These controls can be reduced on memory-constrained devices independently of
external image tiling. Tiling still needs enough halo for the matching search
radius and patch support.

## Parity and runners

The tests compare matching indices, group sizes, DCT/BIOR transforms,
Hadamard filters, aggregation behavior, and complete grayscale/color outputs.
Float64 CPU results match the legacy pipeline within `1e-10`; float32 results
can differ slightly around hard thresholds and tied matches.

- `python run_cbm3d.py` runs the unchanged NumPy reference implementation.
- `python run_cbm3d_tensor.py` selects CUDA, then MPS, then CPU and runs the
  tensor implementation.

Tensor results are written to `data/denoised_tensor`, separate from legacy
outputs.
