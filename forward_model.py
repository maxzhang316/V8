"""Optimized PtyRAD-aligned 2D mixed-state forward model.

Physics / numerical convention
------------------------------
This keeps the verified V0.7 convention unchanged:

    object patch
    × Fourier-shifted mixed-state probe
    -> ortho FFT
    -> incoherent |Psi|^2 sum over probe modes
    -> detector fftshift

The important performance change is object-patch extraction.

OLD:
    object.unfold(...).unfold(...)[y, x]

For a 586x586 object and 128x128 probe this creates a logical sliding-window
view of roughly:

    459 x 459 x 128 x 128

and the backward pass through advanced indexing of that huge view is extremely
slow.

NEW:
    directly gather only the B requested object patches

so a batch of 100 scans touches only:

    100 x 128 x 128

object pixels.

The public functions and tensor contracts are unchanged, so existing
reconstruction.py code can continue importing:

    forward_from_crop_and_shift
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn


# ============================================================================
# Small device caches
# ============================================================================
#
# These tensors are tiny and independent of reconstruction parameters.
# Caching avoids rebuilding arange / fftfreq tensors for every batch.
#

_PATCH_OFFSET_CACHE: dict[
    tuple[str, int, int, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}

_FREQUENCY_CACHE: dict[
    tuple[str, int, int, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}

_TWO_PI: Final[float] = 2.0 * math.pi


def _device_index(
    device: torch.device,
) -> int:
    return -1 if device.index is None else int(device.index)


def _cache_key(
    device: torch.device,
    h: int,
    w: int,
) -> tuple[str, int, int, int]:
    return (
        device.type,
        _device_index(device),
        int(h),
        int(w),
    )


def _get_patch_offsets(
    device: torch.device,
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached local [row, col] offsets for patch gathering."""

    key = _cache_key(
        device,
        h,
        w,
    )

    cached = _PATCH_OFFSET_CACHE.get(
        key
    )

    if cached is not None:
        return cached

    row_offsets = torch.arange(
        h,
        dtype=torch.long,
        device=device,
    ).view(
        1,
        h,
        1,
    )

    col_offsets = torch.arange(
        w,
        dtype=torch.long,
        device=device,
    ).view(
        1,
        1,
        w,
    )

    cached = (
        row_offsets,
        col_offsets,
    )

    _PATCH_OFFSET_CACHE[
        key
    ] = cached

    return cached


def _get_shift_frequency_axes(
    device: torch.device,
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached -2*pi*frequency axes for Fourier probe shifting.

    Shapes:
        phase_y: [1, H, 1]
        phase_x: [1, 1, W]
    """

    key = _cache_key(
        device,
        h,
        w,
    )

    cached = _FREQUENCY_CACHE.get(
        key
    )

    if cached is not None:
        return cached

    phase_y = (
        -_TWO_PI
        * torch.fft.fftfreq(
            h,
            d=1.0,
            device=device,
            dtype=torch.float32,
        )
    ).view(
        1,
        h,
        1,
    )

    phase_x = (
        -_TWO_PI
        * torch.fft.fftfreq(
            w,
            d=1.0,
            device=device,
            dtype=torch.float32,
        )
    ).view(
        1,
        1,
        w,
    )

    cached = (
        phase_y,
        phase_x,
    )

    _FREQUENCY_CACHE[
        key
    ] = cached

    return cached


# ============================================================================
# Probe sub-pixel shifting
# ============================================================================


def subpixel_shift_probes(
    probe: torch.Tensor,
    shifts_yx: torch.Tensor,
) -> torch.Tensor:
    """Fourier-shift all probe modes for every scan in the current batch.

    Parameters
    ----------
    probe:
        complex64 [M, H, W]

    shifts_yx:
        float32 [B, 2], order [shift_y, shift_x]

    Returns
    -------
    complex64 [B, M, H, W]
    """

    if (
        probe.ndim
        != 3
    ):
        raise ValueError(
            "probe must be [N_modes,H,W]"
        )

    if (
        shifts_yx.ndim
        != 2
        or shifts_yx.shape[
            1
        ]
        != 2
    ):
        raise ValueError(
            "shifts_yx must be [N,2] [y,x]"
        )

    if (
        probe.dtype
        != torch.complex64
    ):
        raise TypeError(
            "probe must be complex64"
        )

    if (
        shifts_yx.dtype
        != torch.float32
    ):
        raise TypeError(
            "shifts_yx must be float32"
        )

    if (
        probe.device
        != shifts_yx.device
    ):
        raise ValueError(
            "probe and shifts_yx must share a device"
        )

    h = int(
        probe.shape[
            -2
        ]
    )

    w = int(
        probe.shape[
            -1
        ]
    )

    phase_y_axis, phase_x_axis = (
        _get_shift_frequency_axes(
            probe.device,
            h,
            w,
        )
    )

    # [B,1,1]
    shift_y = shifts_yx[
        :,
        0,
    ].view(
        -1,
        1,
        1,
    )

    shift_x = shifts_yx[
        :,
        1,
    ].view(
        -1,
        1,
        1,
    )

    # [B,H,W]
    #
    # This is mathematically identical to the previous meshgrid expression,
    # but avoids constructing two HxW grids every forward call.
    phase = (
        shift_y
        * phase_y_axis
        + shift_x
        * phase_x_axis
    )

    # exp(i*phase), complex64 because phase is float32.
    phase_factor = torch.polar(
        torch.ones_like(
            phase
        ),
        phase,
    )

    # Probe is trainable, so this FFT must remain inside the autograd graph.
    probe_fft = torch.fft.fft2(
        probe,
        dim=(
            -2,
            -1,
        ),
    )

    shifted = torch.fft.ifft2(
        probe_fft[
            None,
            :,
            :,
            :,
        ]
        * phase_factor[
            :,
            None,
            :,
            :,
        ],
        dim=(
            -2,
            -1,
        ),
    )

    return shifted


# ============================================================================
# Efficient object-patch extraction
# ============================================================================


def _validate_crop_positions(
    crop_positions: torch.Tensor,
    *,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
) -> None:
    """Validate crop bounds.

    This function synchronizes once when tensors are on CUDA, so it should not
    be used in the hot reconstruction loop.  The prepared dataset already
    validates its geometry, therefore forward_from_crop_and_shift defaults to
    validate_bounds=False.

    Direct callers of extract_object_patches keep validation enabled by default.
    """

    h_obj, w_obj = (
        int(
            object_shape[
                0
            ]
        ),
        int(
            object_shape[
                1
            ]
        ),
    )

    h_probe, w_probe = (
        int(
            probe_shape[
                0
            ]
        ),
        int(
            probe_shape[
                1
            ]
        ),
    )

    y = crop_positions[
        :,
        0,
    ]

    x = crop_positions[
        :,
        1,
    ]

    invalid = (
        (y < 0)
        | (x < 0)
        | (
            y
            + h_probe
            > h_obj
        )
        | (
            x
            + w_probe
            > w_obj
        )
    )

    if bool(
        invalid.any().item()
    ):
        raise ValueError(
            "At least one object crop is outside the object canvas."
        )


def extract_object_patches(
    object_tensor: torch.Tensor,
    crop_positions: torch.Tensor,
    probe_shape: tuple[int, int],
    *,
    validate_bounds: bool = True,
) -> torch.Tensor:
    """Gather only the object patches requested by the current batch.

    Parameters
    ----------
    object_tensor:
        [H_obj, W_obj]

    crop_positions:
        integer [B,2] top-left crop positions [y,x]

    probe_shape:
        (H_probe, W_probe)

    Returns
    -------
    [B, H_probe, W_probe]

    Notes
    -----
    This implementation deliberately avoids ``Tensor.unfold`` over the whole
    object.  Gradients still propagate back to object_tensor through the direct
    gather operation.
    """

    if (
        object_tensor.ndim
        != 2
    ):
        raise ValueError(
            "object_tensor must be 2D [H,W]"
        )

    if (
        crop_positions.ndim
        != 2
        or crop_positions.shape[
            1
        ]
        != 2
    ):
        raise ValueError(
            "crop_positions must be [B,2] [y,x]"
        )

    if (
        crop_positions.dtype
        not in (
            torch.int32,
            torch.int64,
        )
    ):
        raise TypeError(
            "crop_positions must be integer"
        )

    if (
        object_tensor.device
        != crop_positions.device
    ):
        raise ValueError(
            "object_tensor and crop_positions must share a device"
        )

    h_probe, w_probe = (
        int(
            probe_shape[
                0
            ]
        ),
        int(
            probe_shape[
                1
            ]
        ),
    )

    h_obj, w_obj = (
        int(
            object_tensor.shape[
                0
            ]
        ),
        int(
            object_tensor.shape[
                1
            ]
        ),
    )

    if validate_bounds:
        _validate_crop_positions(
            crop_positions,
            object_shape=(
                h_obj,
                w_obj,
            ),
            probe_shape=(
                h_probe,
                w_probe,
            ),
        )

    # Use int64 for flattened tensor indexing.
    positions = crop_positions.to(
        dtype=torch.long,
        copy=False,
    )

    y = positions[
        :,
        0,
    ].view(
        -1,
        1,
        1,
    )

    x = positions[
        :,
        1,
    ].view(
        -1,
        1,
        1,
    )

    row_offsets, col_offsets = (
        _get_patch_offsets(
            object_tensor.device,
            h_probe,
            w_probe,
        )
    )

    # Build flat indices directly.
    #
    # [B,H,1] + [1,H,1] -> [B,H,1]
    rows = (
        y
        + row_offsets
    )

    # [B,1,1] + [1,1,W] -> [B,1,W]
    cols = (
        x
        + col_offsets
    )

    # Broadcasting gives [B,H,W].
    flat_indices = (
        rows
        * w_obj
        + cols
    )

    object_flat = object_tensor.reshape(
        -1
    )

    patches = object_flat[
        flat_indices
    ]

    return patches


# ============================================================================
# Verified forward operator
# ============================================================================


def forward_from_crop_and_shift(
    object_tensor: torch.Tensor,
    probe: torch.Tensor,
    crop_positions: torch.Tensor,
    probe_shifts: torch.Tensor,
    *,
    eps: float = 1e-10,
    validate_bounds: bool = False,
) -> torch.Tensor:
    """Mixed-state ptychography forward model from integer crops + shifts.

    ``validate_bounds`` defaults to False for reconstruction performance.
    Prepared datasets already validate that their scan geometry fits the object
    canvas.  Set it True for standalone debugging/tests.
    """

    if (
        object_tensor.ndim
        != 2
        or object_tensor.dtype
        != torch.complex64
    ):
        raise ValueError(
            "object_tensor must be complex64 [H,W]"
        )

    if (
        probe.ndim
        != 3
        or probe.dtype
        != torch.complex64
    ):
        raise ValueError(
            "probe must be complex64 [M,H,W]"
        )

    if (
        crop_positions.dtype
        not in (
            torch.int32,
            torch.int64,
        )
    ):
        raise TypeError(
            "crop_positions must be integer"
        )

    if (
        probe_shifts.dtype
        != torch.float32
    ):
        raise TypeError(
            "probe_shifts must be float32"
        )

    if not (
        object_tensor.device
        == probe.device
        == crop_positions.device
        == probe_shifts.device
    ):
        raise ValueError(
            "All forward tensors must share a device."
        )

    if (
        crop_positions.shape[
            0
        ]
        != probe_shifts.shape[
            0
        ]
    ):
        raise ValueError(
            "crop_positions and probe_shifts must contain the same batch size."
        )

    probe_shape = (
        int(
            probe.shape[
                -2
            ]
        ),
        int(
            probe.shape[
                -1
            ]
        ),
    )

    object_patches = extract_object_patches(
        object_tensor,
        crop_positions,
        probe_shape,
        validate_bounds=validate_bounds,
    )

    shifted_probes = subpixel_shift_probes(
        probe,
        probe_shifts,
    )

    # [B,1,H,W] * [B,M,H,W] -> [B,M,H,W]
    exit_wave = (
        object_patches[
            :,
            None,
            :,
            :,
        ]
        * shifted_probes
    )

    # Same verified detector-plane convention.
    diffraction_wave = torch.fft.fft2(
        exit_wave,
        dim=(
            -2,
            -1,
        ),
        norm="ortho",
    )

    detector_intensity = (
        diffraction_wave
        .abs()
        .square()
        .sum(
            dim=1
        )
    )

    detector_intensity = torch.fft.fftshift(
        detector_intensity,
        dim=(
            -2,
            -1,
        ),
    )

    return (
        detector_intensity
        + float(
            eps
        )
    ).to(
        torch.float32
    )


# ============================================================================
# Module interface
# ============================================================================


class ForwardModel(
    nn.Module
):
    def __init__(
        self,
        detector_shape: tuple[int, int] = (
            124,
            124,
        ),
        eps: float = 1e-10,
    ) -> None:
        super().__init__()

        self.detector_shape = tuple(
            map(
                int,
                detector_shape,
            )
        )

        self.eps = float(
            eps
        )

    def forward(
        self,
        object: torch.Tensor,
        probe: torch.Tensor,
        scan_positions: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tuple(
                map(
                    int,
                    probe.shape[
                        -2:
                    ],
                )
            )
            != self.detector_shape
        ):
            raise ValueError(
                "probe detector shape does not match ForwardModel.detector_shape"
            )

        rounded = torch.round(
            scan_positions
        )

        crop_positions = rounded.to(
            torch.long
        )

        shifts = (
            scan_positions
            - rounded
        ).to(
            torch.float32
        )

        return forward_from_crop_and_shift(
            object_tensor=object,
            probe=probe,
            crop_positions=crop_positions,
            probe_shifts=shifts,
            eps=self.eps,
            validate_bounds=False,
        )


def forward_model(
    object: torch.Tensor,
    probe: torch.Tensor,
    scan_positions: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible functional interface."""

    return ForwardModel(
        detector_shape=tuple(
            map(
                int,
                probe.shape[
                    -2:
                ],
            )
        ),
    )(
        object,
        probe,
        scan_positions,
    )