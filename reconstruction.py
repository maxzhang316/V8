from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from constraints import (
    apply_v8_projections,
    limit_position_iteration_step_,
    orthogonalize_and_sort_probe_modes,
)
from experimental_data import (
    electron_wavelength_angstrom,
    load_prepared_dataset,
)
from forward_model import forward_from_crop_and_shift
from output_manager import OutputManager, mode_fractions

PLATFORM_RECONSTRUCTION_VERSION = "0.8.4"

# ============================================================================
# Device / reproducibility
# ============================================================================


def choose_device(name: str) -> torch.device:
    """Select the requested compute device without silent CUDA fallback."""
    name = str(name).lower()

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this PyTorch build does not have "
                "CUDA available. Check torch.__version__, torch.version.cuda "
                "and install a CUDA-enabled PyTorch wheel."
            )
        device = torch.device("cuda:0")

    elif name == "cpu":
        device = torch.device("cpu")

    elif name == "auto":
        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        raise ValueError(
            f"Unsupported device setting: {name}"
        )

    print("=" * 72)
    print("COMPUTE DEVICE")
    print("=" * 72)
    print("Requested:", name)
    print("Selected:", device)
    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print("=" * 72)

    return device


def set_seed(seed: int) -> None:
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def synchronize_if_cuda(
    device: torch.device,
) -> None:
    """Synchronize CUDA so perf_counter() measures actual GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )


# ============================================================================
# Object parameterisation
# ============================================================================


def build_complex_object(
    obja: torch.Tensor,
    objp: torch.Tensor,
) -> torch.Tensor:
    """Return O=A*exp(i*phi) as complex64."""
    return (
        torch.complex(
            obja,
            torch.zeros_like(
                obja
            ),
        )
        * torch.exp(
            torch.complex(
                torch.zeros_like(
                    objp
                ),
                objp,
            )
        )
    ).to(
        torch.complex64
    )


# ============================================================================
# Backward-compatible object-amplitude fallback
# ============================================================================


@torch.no_grad()
def project_object_amplitude_positive_(
    obja: torch.Tensor,
) -> dict[str, float]:
    """Fallback projection onto A(x,y) >= 0.

    When constraints.obja_thresh is active, that more informative relaxed
    threshold supersedes this fallback.
    """

    before_min = float(
        obja.min()
        .detach()
        .cpu()
    )

    before_max = float(
        obja.max()
        .detach()
        .cpu()
    )

    negative_fraction = float(
        (
            obja
            < 0
        )
        .to(
            torch.float32
        )
        .mean()
        .detach()
        .cpu()
    )

    obja.clamp_(
        min=0.0
    )

    return {
        "obja_min_before_positive_projection": before_min,
        "obja_max_before_positive_projection": before_max,
        "obja_negative_fraction_before_projection": negative_fraction,
        "obja_min_after_positive_projection": float(
            obja.min()
            .detach()
            .cpu()
        ),
        "obja_max_after_positive_projection": float(
            obja.max()
            .detach()
            .cpu()
        ),
        "obja_mean_after_positive_projection": float(
            obja.mean()
            .detach()
            .cpu()
        ),
    }


# ============================================================================
# Probe initialization
# ============================================================================


def simulate_stem_probe(
    *,
    kv: float,
    conv_angle_mrad: float,
    npix: int,
    dx_angstrom: float,
    c10_angstrom: float,
    device: torch.device,
) -> torch.Tensor:
    wavelength = (
        electron_wavelength_angstrom(
            kv
        )
    )

    k_aperture = (
        float(
            conv_angle_mrad
        )
        / 1.0e3
        / wavelength
    )

    frequency = torch.fft.fftshift(
        torch.fft.fftfreq(
            npix,
            d=float(
                dx_angstrom
            ),
            dtype=torch.float32,
            device=device,
        )
    )

    ky, kx = torch.meshgrid(
        frequency,
        frequency,
        indexing="ij",
    )

    kr = torch.sqrt(
        kx.square()
        + ky.square()
    )

    aperture = (
        kr
        <= k_aperture
    ).to(
        torch.float32
    )

    alpha_r = (
        kr
        * wavelength
    )

    chi = (
        math.pi
        * float(
            c10_angstrom
        )
        * alpha_r.square()
        / wavelength
    )

    pupil = torch.polar(
        aperture,
        -chi,
    ).to(
        torch.complex64
    )

    probe = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(
                pupil,
                dim=(
                    -2,
                    -1,
                ),
            ),
            dim=(
                -2,
                -1,
            ),
        ),
        dim=(
            -2,
            -1,
        ),
    )

    probe = (
        probe
        / torch.sqrt(
            torch.sum(
                torch.abs(
                    probe
                ).square()
            )
            .real
            .clamp_min(
                1.0e-12
            )
        )
    )

    return probe.to(
        torch.complex64
    )


def _orthogonalize_candidate(
    candidate: torch.Tensor,
    basis: list[torch.Tensor],
) -> torch.Tensor:
    flat = candidate.reshape(
        -1
    )

    for existing in basis:
        b = existing.reshape(
            -1
        )

        coeff = (
            torch.sum(
                torch.conj(
                    b
                )
                * flat
            )
            / torch.sum(
                torch.conj(
                    b
                )
                * b
            )
            .real
            .clamp_min(
                1.0e-12
            )
        )

        flat = (
            flat
            - coeff
            * b
        )

    norm = (
        torch.linalg.vector_norm(
            flat
        )
        .clamp_min(
            1.0e-12
        )
    )

    return (
        flat
        / norm
    ).reshape(
        candidate.shape
    ).to(
        torch.complex64
    )


def initialize_mixed_probe(
    *,
    primary_probe: torch.Tensor,
    n_modes: int,
    new_mode_power: float,
    basis: str = "hermite",
) -> torch.Tensor:
    """Initialize a deterministic mixed-state basis around a physical probe."""

    if n_modes < 1:
        raise ValueError(
            "n_modes must be >= 1"
        )

    if n_modes == 1:
        return primary_probe[
            None
        ]

    basis = str(
        basis
    ).lower()

    h, w = primary_probe.shape

    yy = torch.linspace(
        -1.0,
        1.0,
        h,
        device=primary_probe.device,
    )

    xx = torch.linspace(
        -1.0,
        1.0,
        w,
        device=primary_probe.device,
    )

    gy, gx = torch.meshgrid(
        yy,
        xx,
        indexing="ij",
    )

    if basis == "hermite":
        modifiers = [
            gx,
            gy,
            gx
            * gy,
            gx.square()
            - gy.square(),
            (
                2.0
                * gx.square()
                - 1.0
            ),
            (
                2.0
                * gy.square()
                - 1.0
            ),
        ]

    elif basis == "shifted":
        modifiers = [
            gx,
            gy,
            gx
            * gy,
            gx.square()
            - gy.square(),
        ]

    else:
        raise ValueError(
            "Unsupported probe_mode_basis: "
            f"{basis}"
        )

    primary_norm = (
        torch.linalg.vector_norm(
            primary_probe
        )
    )

    modes: list[torch.Tensor] = [
        primary_probe
    ]

    for index in range(
        1,
        n_modes,
    ):
        modifier = modifiers[
            (
                index
                - 1
            )
            % len(
                modifiers
            )
        ]

        candidate = (
            primary_probe
            * modifier.to(
                primary_probe.dtype
            )
        )

        candidate = (
            _orthogonalize_candidate(
                candidate,
                modes,
            )
        )

        candidate = (
            candidate
            * primary_norm
            * math.sqrt(
                float(
                    new_mode_power
                )
            )
        )

        modes.append(
            candidate
        )

    probe = torch.stack(
        modes,
        dim=0,
    )

    probe = (
        probe
        / torch.sqrt(
            torch.sum(
                torch.abs(
                    probe
                ).square()
            )
            .real
            .clamp_min(
                1.0e-12
            )
        )
    )

    return (
        orthogonalize_and_sort_probe_modes(
            probe
        )
    )


@torch.no_grad()
def normalize_probe_to_measurement(
    probe: torch.Tensor,
    measured: torch.Tensor,
) -> float:
    target = (
        measured.sum(
            dim=(
                -2,
                -1,
            )
        )
        .mean()
    )

    current = (
        torch.sum(
            torch.abs(
                probe
            ).square()
        )
        .real
        .clamp_min(
            1.0e-12
        )
    )

    probe.mul_(
        torch.sqrt(
            target
            / current
        )
    )

    return float(
        target.cpu()
    )


# ============================================================================
# Data loss and regularization
# ============================================================================


def normalized_mse(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Dimensionless normalized MSE used by V0.8.4.

    L = mean((source-target)^2) / mean(target)^2
    """

    scale = (
        target.mean()
        .clamp_min(
            1.0e-12
        )
    )

    return (
        torch.mean(
            (
                source
                - target
            ).square()
        )
        / scale.square()
    )


def normalized_rmse(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Legacy helper retained for diagnostics/backward compatibility.

    V0.8.4 hybrid_data_loss does NOT use this function for the intensity term.
    """

    return (
        torch.mean(
            (
                source
                - target
            ).square()
        )
        .sqrt()
        / target.mean()
        .clamp_min(
            1.0e-12
        )
    )


def hybrid_data_loss(
    predicted: torch.Tensor,
    measured: torch.Tensor,
    *,
    amplitude_weight: float,
    intensity_weight: float,
    eps: float,
    charbonnier_eps: float,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Hybrid amplitude/intensity data term.

    The amplitude branch is retained for compatibility with old YAML profiles.
    The V0.8.4 intensity branch is normalized MSE rather than normalized RMSE.
    """

    pred_amp = torch.sqrt(
        predicted.clamp_min(
            0.0
        )
        + float(
            eps
        )
    )

    meas_amp = torch.sqrt(
        measured.clamp_min(
            0.0
        )
        + float(
            eps
        )
    )

    residual = (
        pred_amp
        - meas_amp
    )

    amp_loss = torch.sqrt(
        residual.square()
        + float(
            charbonnier_eps
        )
        ** 2
    ).mean()

    amp_loss = (
        amp_loss
        / meas_amp.mean()
        .clamp_min(
            1.0e-12
        )
    )

    int_loss = normalized_mse(
        predicted,
        measured,
    )

    total = (
        float(
            amplitude_weight
        )
        * amp_loss
        + float(
            intensity_weight
        )
        * int_loss
    )

    return total, {
        "amplitude_data_loss": amp_loss,
        "intensity_data_loss": int_loss,
    }


def probe_total_intensity_map(
    probe: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(
        torch.abs(
            probe
        ).square(),
        dim=0,
    ).real


def probe_intensity_anchor_nrmse(
    probe: torch.Tensor,
    reference_intensity_map: torch.Tensor,
) -> torch.Tensor:
    current = (
        probe_total_intensity_map(
            probe
        )
    )

    current = (
        current
        / current.sum()
        .clamp_min(
            1.0e-12
        )
    )

    reference = (
        reference_intensity_map
        / reference_intensity_map.sum()
        .clamp_min(
            1.0e-12
        )
    )

    return (
        torch.linalg.vector_norm(
            (
                current
                - reference
            ).reshape(
                -1
            )
        )
        / torch.linalg.vector_norm(
            reference.reshape(
                -1
            )
        )
        .clamp_min(
            1.0e-12
        )
    )


def make_probe_support_mask(
    *,
    npix: int,
    dx_angstrom: float,
    kv: float,
    conv_angle_mrad: float,
    radius_scale: float,
    device: torch.device,
) -> torch.Tensor:
    wavelength = (
        electron_wavelength_angstrom(
            kv
        )
    )

    aperture = (
        float(
            conv_angle_mrad
        )
        / 1.0e3
        / wavelength
        * float(
            radius_scale
        )
    )

    frequency = torch.fft.fftshift(
        torch.fft.fftfreq(
            npix,
            d=float(
                dx_angstrom
            ),
            dtype=torch.float32,
            device=device,
        )
    )

    ky, kx = torch.meshgrid(
        frequency,
        frequency,
        indexing="ij",
    )

    kr = torch.sqrt(
        kx.square()
        + ky.square()
    )

    return (
        kr
        <= aperture
    ).to(
        torch.float32
    )


def probe_support_leakage(
    probe: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    reciprocal = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(
                probe,
                dim=(
                    -2,
                    -1,
                ),
            ),
            dim=(
                -2,
                -1,
            ),
        ),
        dim=(
            -2,
            -1,
        ),
    )

    power = torch.abs(
        reciprocal
    ).square()

    outside = (
        power
        * (
            1.0
            - support_mask[
                None
            ]
        )
    )

    return (
        outside.sum()
        / power.sum()
        .clamp_min(
            1.0e-12
        )
    )


def position_huber_regularizer(
    current_continuous: torch.Tensor,
    initial_continuous: torch.Tensor,
    delta_px: float,
) -> torch.Tensor:
    delta = (
        current_continuous
        - initial_continuous
    )

    norm = (
        torch.linalg.vector_norm(
            delta,
            dim=1,
        )
    )

    d = float(
        delta_px
    )

    quadratic = (
        0.5
        * norm.square()
    )

    linear = (
        d
        * (
            norm
            - 0.5
            * d
        )
    )

    huber = torch.where(
        norm
        <= d,
        quadratic,
        linear,
    )

    return huber.mean()


def object_amplitude_bound_penalty(
    obja: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    below = torch.relu(
        float(
            lower
        )
        - obja
    )

    above = torch.relu(
        obja
        - float(
            upper
        )
    )

    return torch.mean(
        below.square()
        + above.square()
    )


def linear_weight(
    *,
    iteration: int,
    start_iter: int,
    end_iter: int,
    start_value: float,
    end_value: float,
) -> float:
    if iteration <= start_iter:
        return float(
            start_value
        )

    if iteration >= end_iter:
        return float(
            end_value
        )

    t = (
        iteration
        - start_iter
    ) / max(
        end_iter
        - start_iter,
        1,
    )

    return float(
        start_value
        + t
        * (
            end_value
            - start_value
        )
    )


# ============================================================================
# Staging / optimizer helpers
# ============================================================================


@dataclass
class StageState:
    object_active: bool
    probe_active: bool
    positions_active: bool
    name: str


def stage_for_iteration(
    iteration: int,
    stage_cfg: dict[str, Any],
    *,
    optimize_positions: bool,
) -> StageState:
    probe_start = int(
        stage_cfg.get(
            "probe_start",
            1,
        )
    )

    position_start = int(
        stage_cfg.get(
            "position_start",
            999999,
        )
    )

    if iteration < probe_start:
        return StageState(
            True,
            False,
            False,
            "object_warmup",
        )

    if not optimize_positions:
        return StageState(
            True,
            True,
            False,
            "object_probe_fixed_pos",
        )

    if iteration < position_start:
        return StageState(
            True,
            True,
            False,
            "probe_refinement",
        )

    return StageState(
        True,
        True,
        True,
        "joint_refinement",
    )


def cosine_lr(
    *,
    base_lr: float,
    iteration: int,
    start_iter: int,
    end_iter: int,
    warmup_iters: int,
    final_scale: float,
) -> float:
    if iteration < start_iter:
        return 0.0

    active_index = (
        iteration
        - start_iter
        + 1
    )

    if (
        warmup_iters
        > 0
        and active_index
        <= warmup_iters
    ):
        return (
            float(
                base_lr
            )
            * active_index
            / warmup_iters
        )

    denom = max(
        end_iter
        - start_iter
        - warmup_iters
        + 1,
        1,
    )

    progress = min(
        max(
            (
                active_index
                - warmup_iters
                - 1
            )
            / denom,
            0.0,
        ),
        1.0,
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    scale = (
        float(
            final_scale
        )
        + (
            1.0
            - float(
                final_scale
            )
        )
        * cosine
    )

    return (
        float(
            base_lr
        )
        * scale
    )


def probe_guard_scale(
    anchor_nrmse: float,
    guard_cfg: dict[str, Any],
) -> float:
    """Return a soft probe-LR damping factor."""

    if not bool(
        guard_cfg.get(
            "enabled",
            True,
        )
    ):
        return 1.0

    soft = float(
        guard_cfg.get(
            "probe_anchor_soft_threshold",
            0.16,
        )
    )

    hard = float(
        guard_cfg.get(
            "probe_anchor_hard_threshold",
            0.30,
        )
    )

    minimum = float(
        guard_cfg.get(
            "minimum_probe_lr_scale",
            0.30,
        )
    )

    minimum = min(
        max(
            minimum,
            0.0,
        ),
        1.0,
    )

    if anchor_nrmse <= soft:
        return 1.0

    if anchor_nrmse >= hard:
        return minimum

    t = (
        anchor_nrmse
        - soft
    ) / max(
        hard
        - soft,
        1.0e-12,
    )

    return (
        1.0
        - t
        * (
            1.0
            - minimum
        )
    )


def set_parameter_activity(
    state: "ReconstructionState",
    stage: StageState,
) -> None:
    state.obja.requires_grad_(
        stage.object_active
    )

    state.objp.requires_grad_(
        stage.object_active
    )

    state.probe.requires_grad_(
        stage.probe_active
    )

    state.probe_pos_shifts.requires_grad_(
        stage.positions_active
    )


def set_group_lr(
    optimizer: torch.optim.Optimizer,
    name: str,
    value: float,
) -> None:
    for group in optimizer.param_groups:
        if group.get(
            "name"
        ) == name:
            group[
                "lr"
            ] = float(
                value
            )
            return

    raise KeyError(
        name
    )


def clip_parameter_gradient(
    parameter: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """Clip one accumulated parameter gradient on its current device."""

    if (
        parameter.grad
        is None
        or max_norm
        <= 0
    ):
        return torch.zeros(
            (),
            dtype=torch.float32,
            device=parameter.device,
        )

    grad = parameter.grad

    norm = (
        torch.linalg.vector_norm(
            grad.detach()
        )
    )

    limit = torch.as_tensor(
        float(
            max_norm
        ),
        dtype=norm.dtype,
        device=norm.device,
    )

    scale = torch.clamp(
        limit
        / torch.clamp(
            norm,
            min=1.0e-12,
        ),
        max=1.0,
    )

    grad.mul_(
        scale
    )

    return norm.detach()


# ============================================================================
# State
# ============================================================================


class ReconstructionState(
    nn.Module
):
    def __init__(
        self,
        *,
        object_shape: tuple[int, int],
        probe_init: torch.Tensor,
        initial_probe_pos_shifts: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.obja = nn.Parameter(
            torch.ones(
                object_shape,
                dtype=torch.float32,
                device=device,
            )
        )

        self.objp = nn.Parameter(
            1.0e-8
            * torch.rand(
                object_shape,
                dtype=torch.float32,
                device=device,
            )
        )

        self.probe = nn.Parameter(
            probe_init.to(
                device=device,
                dtype=torch.complex64,
            )
        )

        self.probe_pos_shifts = nn.Parameter(
            initial_probe_pos_shifts.to(
                device=device,
                dtype=torch.float32,
            )
        )


# ============================================================================
# Quality metrics and snapshot helpers
# ============================================================================


@torch.no_grad()
def current_quality_metrics(
    *,
    state: ReconstructionState,
    crop_positions: torch.Tensor,
    initial_continuous_positions: torch.Tensor,
    initial_probe_intensity: torch.Tensor,
    support_mask: torch.Tensor,
) -> dict[str, float]:
    anchor = float(
        probe_intensity_anchor_nrmse(
            state.probe,
            initial_probe_intensity,
        ).cpu()
    )

    support = float(
        probe_support_leakage(
            state.probe,
            support_mask,
        ).cpu()
    )

    current_positions = (
        crop_positions.to(
            torch.float32
        )
        + state.probe_pos_shifts
    )

    drift = (
        current_positions
        - initial_continuous_positions
    )

    position_rms = float(
        torch.sqrt(
            torch.mean(
                drift.square()
            )
        ).cpu()
    )

    return {
        "probe_anchor_nrmse": anchor,
        "probe_support_leakage": support,
        "position_rms_drift_px": position_rms,
    }


def selection_score(
    *,
    data_loss: float,
    metrics: dict[str, float],
    selection_cfg: dict[str, Any],
) -> float:
    return float(
        data_loss
        + float(
            selection_cfg.get(
                "probe_anchor_weight",
                0.02,
            )
        )
        * metrics[
            "probe_anchor_nrmse"
        ]
        + float(
            selection_cfg.get(
                "probe_support_weight",
                0.02,
            )
        )
        * metrics[
            "probe_support_leakage"
        ]
        + float(
            selection_cfg.get(
                "position_drift_weight",
                0.002,
            )
        )
        * metrics[
            "position_rms_drift_px"
        ]
    )


@torch.no_grad()
def snapshot_state(
    state: ReconstructionState,
    crop_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "obja": (
            state.obja
            .detach()
            .clone()
        ),
        "objp": (
            state.objp
            .detach()
            .clone()
        ),
        "probe": (
            state.probe
            .detach()
            .clone()
        ),
        "probe_pos_shifts": (
            state.probe_pos_shifts
            .detach()
            .clone()
        ),
        "crop_positions": (
            crop_positions
            .detach()
            .clone()
        ),
    }


@torch.no_grad()
def restore_snapshot(
    state: ReconstructionState,
    crop_positions: torch.Tensor,
    snapshot: dict[str, torch.Tensor],
) -> None:
    state.obja.copy_(
        snapshot[
            "obja"
        ]
    )

    state.objp.copy_(
        snapshot[
            "objp"
        ]
    )

    state.probe.copy_(
        snapshot[
            "probe"
        ]
    )

    state.probe_pos_shifts.copy_(
        snapshot[
            "probe_pos_shifts"
        ]
    )

    crop_positions.copy_(
        snapshot[
            "crop_positions"
        ]
    )


# ============================================================================
# Main reconstruction
# ============================================================================


def run_reconstruction(
    config: dict[str, Any],
) -> dict[str, Any]:
    project_cfg = config[
        "project"
    ]

    recon_cfg = config[
        "reconstruction"
    ]

    reg_cfg = config.get(
        "regularization",
        {},
    )

    guard_cfg = config.get(
        "quality_guard",
        {},
    )

    constraint_cfg = config.get(
        "constraints",
        {},
    )

    selection_cfg = config.get(
        "selection",
        {},
    )

    output_cfg = config[
        "output"
    ]

    obja_thresh_enabled = bool(
        constraint_cfg.get(
            "obja_thresh",
            {},
        ).get(
            "enabled",
            False,
        )
    )

    seed = int(
        project_cfg.get(
            "seed",
            0,
        )
    )

    set_seed(
        seed
    )

    device = choose_device(
        project_cfg.get(
            "device",
            "auto",
        )
    )

    print(
        f"Device: {device}"
    )

    (
        measured_np,
        _continuous_positions_np,
        crop_positions_np,
        initial_shifts_np,
        metadata,
    ) = load_prepared_dataset(
        config[
            "data"
        ][
            "prepared_npz_path"
        ]
    )

    measured = (
        torch.from_numpy(
            measured_np
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    crop_positions = (
        torch.from_numpy(
            crop_positions_np
        )
        .to(
            device=device,
            dtype=torch.long,
        )
    )

    initial_shifts = (
        torch.from_numpy(
            initial_shifts_np
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    initial_continuous = (
        crop_positions.to(
            torch.float32
        )
        + initial_shifts
    )

    object_shape = tuple(
        int(
            value
        )
        for value
        in metadata[
            "recommended_object_shape"
        ]
    )

    detector_shape = tuple(
        int(
            value
        )
        for value
        in config[
            "forward"
        ][
            "detector_shape"
        ]
    )

    print(
        f"fitRBF: "
        f"{metadata['fitRBF_px']:.6f} px"
    )

    print(
        f"dx: "
        f"{metadata['dx_angstrom_per_object_pixel']:.9f} "
        "Å/object-pixel"
    )

    print(
        f"scan step: "
        f"{metadata['scan_step_object_pixels']:.9f} "
        "object pixels"
    )

    print(
        f"initial position random std: "
        f"{metadata['pos_scan_rand_std_px']:.6f} px"
    )

    print(
        f"object shape: "
        f"{object_shape}"
    )

    if obja_thresh_enabled:
        amp_cfg = constraint_cfg.get(
            "obja_thresh",
            {},
        )

        print(
            "Object amplitude threshold: enabled "
            f"thresh={amp_cfg.get('thresh', [0.85, 1.15])} "
            f"relax={float(amp_cfg.get('relax', 0.0)):.4f} "
            f"start_iter={int(amp_cfg.get('start_iter', 1))}"
        )

    else:
        print(
            "Object amplitude threshold: disabled "
            "(positive-only fallback active)"
        )

    primary_probe = simulate_stem_probe(
        kv=float(
            recon_cfg[
                "probe_kv"
            ]
        ),
        conv_angle_mrad=float(
            recon_cfg[
                "probe_conv_angle_mrad"
            ]
        ),
        npix=int(
            detector_shape[
                0
            ]
        ),
        dx_angstrom=float(
            metadata[
                "dx_angstrom_per_object_pixel"
            ]
        ),
        c10_angstrom=float(
            recon_cfg.get(
                "probe_C10_angstrom",
                0.0,
            )
        ),
        device=device,
    )

    probe_init = initialize_mixed_probe(
        primary_probe=primary_probe,
        n_modes=int(
            recon_cfg[
                "n_modes"
            ]
        ),
        new_mode_power=float(
            recon_cfg.get(
                "probe_mode_init_power",
                0.02,
            )
        ),
        basis=str(
            recon_cfg.get(
                "probe_mode_basis",
                "hermite",
            )
        ),
    )

    # Probe/object amplitude has a multiplicative gauge ambiguity.
    #
    # For pure-phase datasets, mean integrated diffraction power is a valid
    # proxy for incident probe power.  For absorptive/complex objects it is not:
    # transmission |O|<1 lowers the measured integrated power.
    #
    # Synthetic complex-object datasets may therefore provide the known
    # incident probe power explicitly in metadata.  If absent, preserve the
    # historical V0.8.4 behaviour for A/B/C and real-data compatibility.
    if "incident_probe_power" in metadata:
        fixed_probe_power = float(metadata["incident_probe_power"])

        with torch.no_grad():
            current_probe_power = (
                torch.sum(torch.abs(probe_init).square())
                .real
                .clamp_min(1.0e-12)
            )
            target_probe_power = torch.as_tensor(
                fixed_probe_power,
                device=probe_init.device,
                dtype=current_probe_power.dtype,
            )
            probe_init.mul_(
                torch.sqrt(target_probe_power / current_probe_power)
            )

        print(
            "Probe power target: metadata incident_probe_power = "
            f"{fixed_probe_power:.9g}"
        )
    else:
        fixed_probe_power = normalize_probe_to_measurement(
            probe_init,
            measured,
        )
        print(
            "Probe power target: historical mean integrated measurement = "
            f"{fixed_probe_power:.9g}"
        )

    initial_probe_intensity = (
        probe_total_intensity_map(
            probe_init
        )
        .detach()
        .clone()
    )

    support_cfg = reg_cfg.get(
        "probe_k_support",
        {},
    )

    support_mask = make_probe_support_mask(
        npix=int(
            detector_shape[
                0
            ]
        ),
        dx_angstrom=float(
            metadata[
                "dx_angstrom_per_object_pixel"
            ]
        ),
        kv=float(
            recon_cfg[
                "probe_kv"
            ]
        ),
        conv_angle_mrad=float(
            recon_cfg[
                "probe_conv_angle_mrad"
            ]
        ),
        radius_scale=float(
            support_cfg.get(
                "radius_scale",
                1.08,
            )
        ),
        device=device,
    )

    state = ReconstructionState(
        object_shape=object_shape,
        probe_init=probe_init,
        initial_probe_pos_shifts=initial_shifts,
        device=device,
    ).to(
        device
    )

    if not bool(
        recon_cfg.get(
            "optimize_positions",
            False,
        )
    ):
        state.probe_pos_shifts.requires_grad_(
            False
        )

    print(
        "Tensor devices:"
    )

    print(
        "  measured:",
        measured.device,
    )

    print(
        "  obja:",
        state.obja.device,
    )

    print(
        "  objp:",
        state.objp.device,
    )

    print(
        "  probe:",
        state.probe.device,
    )

    print(
        "  probe_pos_shifts:",
        state.probe_pos_shifts.device,
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    state.obja
                ],
                "lr": 0.0,
                "name": "obja",
            },
            {
                "params": [
                    state.objp
                ],
                "lr": 0.0,
                "name": "objp",
            },
            {
                "params": [
                    state.probe
                ],
                "lr": 0.0,
                "name": "probe",
            },
            {
                "params": [
                    state.probe_pos_shifts
                ],
                "lr": 0.0,
                "name": "probe_pos_shifts",
            },
        ],
        betas=tuple(
            float(
                value
            )
            for value
            in recon_cfg.get(
                "adam_betas",
                [
                    0.9,
                    0.999,
                ],
            )
        ),
        eps=float(
            recon_cfg.get(
                "adam_eps",
                1.0e-8,
            )
        ),
    )

    output = OutputManager(
        config
    )

    history: list[
        dict[str, Any]
    ] = []

    iterations = int(
        recon_cfg[
            "iterations"
        ]
    )

    batch_size = min(
        int(
            recon_cfg[
                "batch_size"
            ]
        ),
        int(
            measured.shape[
                0
            ]
        ),
    )

    n_scan = int(
        measured.shape[
            0
        ]
    )

    n_batches = (
        n_scan
        + batch_size
        - 1
    ) // batch_size

    stage_cfg = recon_cfg.get(
        "stages",
        {},
    )

    warmup_iters = int(
        stage_cfg.get(
            "lr_warmup_iters",
            5,
        )
    )

    final_lr_scale = float(
        stage_cfg.get(
            "final_lr_scale",
            0.25,
        )
    )

    object_start = int(
        stage_cfg.get(
            "object_start",
            1,
        )
    )

    probe_start = int(
        stage_cfg.get(
            "probe_start",
            1,
        )
    )

    position_start = int(
        stage_cfg.get(
            "position_start",
            999999,
        )
    )

    optimize_positions = bool(
        recon_cfg.get(
            "optimize_positions",
            False,
        )
    )

    print(
        f"Optimize positions: "
        f"{optimize_positions}"
    )

    if not optimize_positions:
        print(
            "V0.8.4 fixed-position baseline: crop positions and sub-pixel "
            "shifts remain exactly at their prepared initial values."
        )

    print(
        "V0.8.4 intensity loss: normalized MSE"
    )

    print(
        "V0.8.4 optimizer cadence: "
        f"{n_batches} mini-batch backward passes -> "
        "1 Adam step per reconstruction iteration"
    )

    best_score = float(
        "inf"
    )

    best_iteration: (
        int
        | None
    ) = None

    best_snapshot: (
        dict[
            str,
            torch.Tensor,
        ]
        | None
    ) = None

    best_row: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    last_measured: (
        torch.Tensor
        | None
    ) = None

    last_predicted: (
        torch.Tensor
        | None
    ) = None

    previous_stage_name = None

    # ------------------------------------------------------------------------
    # Timing
    #
    # Two timings are recorded:
    #
    # 1) reconstruction_core_seconds
    #    Sum of synchronized per-iteration reconstruction compute time.
    #    Excludes post-iteration quality bookkeeping, plotting, checkpoint
    #    writing, HDF5 export and final-summary export.
    #
    # 2) reconstruction_wall_seconds
    #    End-to-end wall-clock across the reconstruction loop, including
    #    plotting/checkpoint work triggered inside the loop.
    #
    # CUDA synchronization is required because GPU kernels are asynchronous.
    # ------------------------------------------------------------------------

    synchronize_if_cuda(
        device
    )

    reconstruction_wall_start = (
        time.perf_counter()
    )

    reconstruction_core_seconds = 0.0

    for iteration in range(
        1,
        iterations
        + 1,
    ):
        synchronize_if_cuda(
            device
        )

        start_time = (
            time.perf_counter()
        )

        stage = stage_for_iteration(
            iteration,
            stage_cfg,
            optimize_positions=optimize_positions,
        )

        set_parameter_activity(
            state,
            stage,
        )

        quality_before = current_quality_metrics(
            state=state,
            crop_positions=crop_positions,
            initial_continuous_positions=initial_continuous,
            initial_probe_intensity=initial_probe_intensity,
            support_mask=support_mask,
        )

        guard_scale = (
            probe_guard_scale(
                quality_before[
                    "probe_anchor_nrmse"
                ],
                guard_cfg,
            )
            if stage.probe_active
            else 0.0
        )

        lr_obja = cosine_lr(
            base_lr=float(
                recon_cfg[
                    "lr_obja"
                ]
            ),
            iteration=iteration,
            start_iter=object_start,
            end_iter=iterations,
            warmup_iters=warmup_iters,
            final_scale=final_lr_scale,
        )

        lr_objp = cosine_lr(
            base_lr=float(
                recon_cfg[
                    "lr_objp"
                ]
            ),
            iteration=iteration,
            start_iter=object_start,
            end_iter=iterations,
            warmup_iters=warmup_iters,
            final_scale=final_lr_scale,
        )

        lr_probe = (
            cosine_lr(
                base_lr=float(
                    recon_cfg[
                        "lr_probe"
                    ]
                ),
                iteration=iteration,
                start_iter=probe_start,
                end_iter=iterations,
                warmup_iters=warmup_iters,
                final_scale=final_lr_scale,
            )
            * guard_scale
        )

        lr_position = cosine_lr(
            base_lr=float(
                recon_cfg[
                    "lr_probe_pos_shifts"
                ]
            ),
            iteration=iteration,
            start_iter=position_start,
            end_iter=iterations,
            warmup_iters=warmup_iters,
            final_scale=final_lr_scale,
        )

        set_group_lr(
            optimizer,
            "obja",
            (
                lr_obja
                if stage.object_active
                else 0.0
            ),
        )

        set_group_lr(
            optimizer,
            "objp",
            (
                lr_objp
                if stage.object_active
                else 0.0
            ),
        )

        set_group_lr(
            optimizer,
            "probe",
            (
                lr_probe
                if stage.probe_active
                else 0.0
            ),
        )

        set_group_lr(
            optimizer,
            "probe_pos_shifts",
            (
                lr_position
                if stage.positions_active
                else 0.0
            ),
        )

        if (
            stage.name
            != previous_stage_name
        ):
            print(
                f"\n=== V0.8.4 stage: {stage.name} "
                f"(iter {iteration}) ===\n"
                f"object={stage.object_active} "
                f"probe={stage.probe_active} "
                f"positions={stage.positions_active} "
                f"(optimize_positions={optimize_positions})"
            )

            previous_stage_name = (
                stage.name
            )

        previous_continuous = (
            crop_positions.to(
                torch.float32
            )
            + state.probe_pos_shifts.detach()
        ).clone()

        order = (
            torch.randperm(
                n_scan,
                device=device,
            )
            if bool(
                recon_cfg.get(
                    "shuffle",
                    True,
                )
            )
            else torch.arange(
                n_scan,
                device=device,
            )
        )

        weighted_total = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        weighted_data = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        weighted_amp = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        weighted_int = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        samples_seen = 0

        # --------------------------------------------------------------------
        # V0.8.4 core change:
        # zero once, accumulate the complete scan-pass gradient, clip once,
        # step once.
        # --------------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        anchor_cfg = reg_cfg.get(
            "probe_intensity_anchor",
            {},
        )

        anchor_weight = 0.0

        if (
            stage.probe_active
            and anchor_cfg.get(
                "enabled",
                False,
            )
        ):
            anchor_weight = linear_weight(
                iteration=iteration,
                start_iter=int(
                    anchor_cfg.get(
                        "start_iter",
                        probe_start,
                    )
                ),
                end_iter=iterations,
                start_value=float(
                    anchor_cfg.get(
                        "weight_start",
                        0.03,
                    )
                ),
                end_value=float(
                    anchor_cfg.get(
                        "weight_end",
                        0.01,
                    )
                ),
            )

        pos_cfg = reg_cfg.get(
            "position_huber",
            {},
        )

        amp_bound_cfg = reg_cfg.get(
            "object_amplitude_soft_bounds",
            {},
        )

        for batch_index, batch_start in enumerate(
            range(
                0,
                n_scan,
                batch_size,
            ),
            start=1,
        ):
            idx = order[
                batch_start:
                batch_start
                + batch_size
            ]

            measured_batch = (
                measured[
                    idx
                ]
            )

            crop_batch = (
                crop_positions[
                    idx
                ]
            )

            shift_batch = (
                state.probe_pos_shifts[
                    idx
                ]
            )

            object_complex = build_complex_object(
                state.obja,
                state.objp,
            )

            predicted = forward_from_crop_and_shift(
                object_tensor=object_complex,
                probe=state.probe,
                crop_positions=crop_batch,
                probe_shifts=shift_batch,
                eps=float(
                    config[
                        "forward"
                    ].get(
                        "eps",
                        1.0e-10,
                    )
                ),
            )

            data_loss, components = hybrid_data_loss(
                predicted,
                measured_batch,
                amplitude_weight=float(
                    recon_cfg[
                        "data_loss"
                    ].get(
                        "amplitude_weight",
                        0.0,
                    )
                ),
                intensity_weight=float(
                    recon_cfg[
                        "data_loss"
                    ].get(
                        "intensity_weight",
                        1.0,
                    )
                ),
                eps=float(
                    recon_cfg[
                        "data_loss"
                    ].get(
                        "eps",
                        1.0e-8,
                    )
                ),
                charbonnier_eps=float(
                    recon_cfg[
                        "data_loss"
                    ].get(
                        "charbonnier_eps",
                        1.0e-3,
                    )
                ),
            )

            total_loss = (
                data_loss
            )

            if (
                stage.probe_active
                and anchor_cfg.get(
                    "enabled",
                    False,
                )
            ):
                anchor_loss = (
                    probe_intensity_anchor_nrmse(
                        state.probe,
                        initial_probe_intensity,
                    )
                )

                total_loss = (
                    total_loss
                    + anchor_weight
                    * anchor_loss
                )

            if (
                stage.probe_active
                and support_cfg.get(
                    "enabled",
                    False,
                )
            ):
                support_loss = probe_support_leakage(
                    state.probe,
                    support_mask,
                )

                total_loss = (
                    total_loss
                    + float(
                        support_cfg.get(
                            "weight",
                            0.025,
                        )
                    )
                    * support_loss
                )

            if (
                stage.positions_active
                and pos_cfg.get(
                    "enabled",
                    False,
                )
            ):
                current_continuous_batch = (
                    crop_batch.to(
                        torch.float32
                    )
                    + shift_batch
                )

                pos_loss = position_huber_regularizer(
                    current_continuous_batch,
                    initial_continuous[
                        idx
                    ],
                    float(
                        pos_cfg.get(
                            "delta_px",
                            0.35,
                        )
                    ),
                )

                total_loss = (
                    total_loss
                    + float(
                        pos_cfg.get(
                            "weight",
                            0.0025,
                        )
                    )
                    * pos_loss
                )

            if amp_bound_cfg.get(
                "enabled",
                False,
            ):
                amp_bound_loss = object_amplitude_bound_penalty(
                    state.obja,
                    float(
                        amp_bound_cfg.get(
                            "lower",
                            0.35,
                        )
                    ),
                    float(
                        amp_bound_cfg.get(
                            "upper",
                            1.85,
                        )
                    ),
                )

                total_loss = (
                    total_loss
                    + float(
                        amp_bound_cfg.get(
                            "weight",
                            0.001,
                        )
                    )
                    * amp_bound_loss
                )

            count = int(
                measured_batch.shape[
                    0
                ]
            )

            # This scaling makes the accumulated gradient the weighted mean
            # across the whole scan pass, while retaining mini-batch memory use.
            gradient_weight = (
                float(
                    count
                )
                / float(
                    n_scan
                )
            )

            (
                total_loss
                * gradient_weight
            ).backward()

            weighted_total = (
                weighted_total
                + total_loss.detach()
                .to(
                    torch.float32
                )
                * count
            )

            weighted_data = (
                weighted_data
                + data_loss.detach()
                .to(
                    torch.float32
                )
                * count
            )

            weighted_amp = (
                weighted_amp
                + components[
                    "amplitude_data_loss"
                ]
                .detach()
                .to(
                    torch.float32
                )
                * count
            )

            weighted_int = (
                weighted_int
                + components[
                    "intensity_data_loss"
                ]
                .detach()
                .to(
                    torch.float32
                )
                * count
            )

            samples_seen += (
                count
            )

            last_measured = (
                measured_batch.detach()
            )

            last_predicted = (
                predicted.detach()
            )

            if (
                n_batches
                >= 50
                and (
                    batch_index
                    == 1
                    or batch_index
                    % 20
                    == 0
                    or batch_index
                    == n_batches
                )
            ):
                print(
                    f"  [iter {iteration:04d}] "
                    f"batch {batch_index:03d}/"
                    f"{n_batches:03d} "
                    "(gradient accumulation)"
                )

        # --------------------------------------------------------------------
        # One accumulated-gradient clip per active parameter group.
        # --------------------------------------------------------------------

        clip_cfg = recon_cfg.get(
            "gradient_clip",
            {},
        )

        grad_norms = {
            "grad_obja": (
                clip_parameter_gradient(
                    state.obja,
                    float(
                        clip_cfg.get(
                            "obja",
                            0.0,
                        )
                    ),
                )
            ),
            "grad_objp": (
                clip_parameter_gradient(
                    state.objp,
                    float(
                        clip_cfg.get(
                            "objp",
                            0.0,
                        )
                    ),
                )
            ),
            "grad_probe": (
                clip_parameter_gradient(
                    state.probe,
                    float(
                        clip_cfg.get(
                            "probe",
                            0.0,
                        )
                    ),
                )
                if stage.probe_active
                else torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=device,
                )
            ),
            "grad_position": (
                clip_parameter_gradient(
                    state.probe_pos_shifts,
                    float(
                        clip_cfg.get(
                            "probe_pos_shifts",
                            0.0,
                        )
                    ),
                )
                if stage.positions_active
                else torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=device,
                )
            ),
        }

        # Exactly one optimizer update per reconstruction iteration.
        optimizer.step()

        # --------------------------------------------------------------------
        # Iteration-level projected-Adam stabilization.
        # --------------------------------------------------------------------

        projection_metrics: dict[
            str,
            float,
        ] = {}

        if stage.positions_active:
            projection_metrics.update(
                limit_position_iteration_step_(
                    crop_positions,
                    state.probe_pos_shifts,
                    previous_continuous,
                    float(
                        guard_cfg.get(
                            "max_position_step_per_iteration_px",
                            0.08,
                        )
                    ),
                )
            )

        projection_metrics.update(
            apply_v8_projections(
                obja=state.obja,
                objp=state.objp,
                probe=state.probe,
                crop_positions=crop_positions,
                probe_pos_shifts=state.probe_pos_shifts,
                fixed_probe_power=fixed_probe_power,
                constraints_cfg=constraint_cfg,
                iteration=iteration,
                probe_active=stage.probe_active,
                positions_active=stage.positions_active,
                object_shape=object_shape,
                probe_shape=detector_shape,
            )
        )

        # Whenever obja_thresh is not active on this iteration, retain the
        # previous non-negative-amplitude fallback.
        if float(
            projection_metrics.get(
                "obja_thresh_applied",
                0.0,
            )
        ) < 0.5:
            projection_metrics.update(
                project_object_amplitude_positive_(
                    state.obja
                )
            )

        denom = float(
            max(
                samples_seen,
                1,
            )
        )

        loss_values = (
            torch.stack(
                [
                    weighted_total
                    / denom,
                    weighted_data
                    / denom,
                    weighted_amp
                    / denom,
                    weighted_int
                    / denom,
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )

        (
            mean_total,
            mean_data,
            mean_amp,
            mean_int,
        ) = (
            float(
                value
            )
            for value
            in loss_values
        )

        if not all(
            math.isfinite(
                value
            )
            for value
            in (
                mean_total,
                mean_data,
                mean_amp,
                mean_int,
            )
        ):
            raise FloatingPointError(
                "Non-finite V0.8.4 reconstruction loss."
            )

        grad_values = (
            torch.stack(
                [
                    grad_norms[
                        "grad_obja"
                    ],
                    grad_norms[
                        "grad_objp"
                    ],
                    grad_norms[
                        "grad_probe"
                    ],
                    grad_norms[
                        "grad_position"
                    ],
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )

        grad_norm_values = {
            "grad_obja": float(
                grad_values[
                    0
                ]
            ),
            "grad_objp": float(
                grad_values[
                    1
                ]
            ),
            "grad_probe": float(
                grad_values[
                    2
                ]
            ),
            "grad_position": float(
                grad_values[
                    3
                ]
            ),
        }

        synchronize_if_cuda(
            device
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        reconstruction_core_seconds += (
            elapsed
        )

        quality = current_quality_metrics(
            state=state,
            crop_positions=crop_positions,
            initial_continuous_positions=initial_continuous,
            initial_probe_intensity=initial_probe_intensity,
            support_mask=support_mask,
        )

        score = selection_score(
            data_loss=mean_data,
            metrics=quality,
            selection_cfg=selection_cfg,
        )

        fractions = mode_fractions(
            state.probe
        )

        continuous_now = (
            crop_positions.to(
                torch.float32
            )
            + state.probe_pos_shifts
        )

        shifts_now = (
            state.probe_pos_shifts
            .detach()
        )

        row: dict[
            str,
            Any,
        ] = {
            "iteration": iteration,
            "stage": stage.name,
            "total_loss": mean_total,
            "data_loss": mean_data,
            "amplitude_data_loss": mean_amp,
            "intensity_data_loss": mean_int,
            "selection_score": score,
            "seconds": elapsed,
            "lr_obja": lr_obja,
            "lr_objp": lr_objp,
            "lr_probe": lr_probe,
            "lr_position": lr_position,
            "probe_guard_scale": guard_scale,
            "probe_anchor_nrmse": quality[
                "probe_anchor_nrmse"
            ],
            "probe_support_leakage": quality[
                "probe_support_leakage"
            ],
            "position_rms_drift_px": quality[
                "position_rms_drift_px"
            ],
            "position_shift_std_y": float(
                shifts_now[
                    :,
                    0,
                ]
                .std()
                .cpu()
            ),
            "position_shift_std_x": float(
                shifts_now[
                    :,
                    1,
                ]
                .std()
                .cpu()
            ),
            "position_continuous_std_y": float(
                continuous_now[
                    :,
                    0,
                ]
                .std()
                .detach()
                .cpu()
            ),
            "position_continuous_std_x": float(
                continuous_now[
                    :,
                    1,
                ]
                .std()
                .detach()
                .cpu()
            ),
            "optimizer_steps_per_iteration": 1,
            "minibatches_per_iteration": n_batches,
            **grad_norm_values,
            "object_blur_applied": float(
                projection_metrics.get(
                    "object_blur_applied",
                    0.0,
                )
            ),
            "probe_mask_k_applied": float(
                projection_metrics.get(
                    "probe_mask_k_applied",
                    0.0,
                )
            ),
            "probe_ortho_applied": float(
                projection_metrics.get(
                    "probe_ortho_applied",
                    0.0,
                )
            ),
            "position_step_max_before_clip": float(
                projection_metrics.get(
                    "position_step_max_before_clip",
                    0.0,
                )
            ),
            "position_step_clipped_fraction": float(
                projection_metrics.get(
                    "position_step_clipped_fraction",
                    0.0,
                )
            ),
            "position_shift_abs_max_after_rebase": float(
                projection_metrics.get(
                    "position_shift_abs_max_after_rebase",
                    shifts_now.abs()
                    .max()
                    .cpu(),
                )
            ),
            "obja_thresh_applied": float(
                projection_metrics.get(
                    "obja_thresh_applied",
                    0.0,
                )
            ),
            "obja_thresh_lower": float(
                projection_metrics.get(
                    "obja_thresh_lower",
                    0.0,
                )
            ),
            "obja_thresh_upper": float(
                projection_metrics.get(
                    "obja_thresh_upper",
                    0.0,
                )
            ),
            "obja_thresh_relax": float(
                projection_metrics.get(
                    "obja_thresh_relax",
                    0.0,
                )
            ),
            "obja_thresh_fraction_below": float(
                projection_metrics.get(
                    "obja_thresh_fraction_below",
                    0.0,
                )
            ),
            "obja_thresh_fraction_above": float(
                projection_metrics.get(
                    "obja_thresh_fraction_above",
                    0.0,
                )
            ),
            "obja_min_before_thresh": float(
                projection_metrics.get(
                    "obja_min_before_thresh",
                    state.obja.detach()
                    .min()
                    .cpu(),
                )
            ),
            "obja_max_before_thresh": float(
                projection_metrics.get(
                    "obja_max_before_thresh",
                    state.obja.detach()
                    .max()
                    .cpu(),
                )
            ),
            "obja_min_after_thresh": float(
                projection_metrics.get(
                    "obja_min_after_thresh",
                    state.obja.detach()
                    .min()
                    .cpu(),
                )
            ),
            "obja_max_after_thresh": float(
                projection_metrics.get(
                    "obja_max_after_thresh",
                    state.obja.detach()
                    .max()
                    .cpu(),
                )
            ),
            "obja_min_before_positive_projection": float(
                projection_metrics.get(
                    "obja_min_before_positive_projection",
                    0.0,
                )
            ),
            "obja_max_before_positive_projection": float(
                projection_metrics.get(
                    "obja_max_before_positive_projection",
                    0.0,
                )
            ),
            "obja_negative_fraction_before_projection": float(
                projection_metrics.get(
                    "obja_negative_fraction_before_projection",
                    0.0,
                )
            ),
            "obja_min": float(
                state.obja.detach()
                .min()
                .cpu()
            ),
            "obja_max": float(
                state.obja.detach()
                .max()
                .cpu()
            ),
            "obja_mean": float(
                state.obja.detach()
                .mean()
                .cpu()
            ),
            "objp_min": float(
                state.objp.detach()
                .min()
                .cpu()
            ),
            "objp_max": float(
                state.objp.detach()
                .max()
                .cpu()
            ),
            "objp_mean": float(
                state.objp.detach()
                .mean()
                .cpu()
            ),
        }

        for index, fraction in enumerate(
            fractions
        ):
            row[
                f"mode_{index:02d}_fraction"
            ] = float(
                fraction
            )

        history.append(
            row
        )

        output.append_history(
            row
        )

        if (
            bool(
                selection_cfg.get(
                    "enabled",
                    True,
                )
            )
            and iteration
            >= int(
                selection_cfg.get(
                    "start_iter",
                    50,
                )
            )
        ):
            if score < best_score:
                best_score = (
                    score
                )

                best_iteration = (
                    iteration
                )

                best_snapshot = snapshot_state(
                    state,
                    crop_positions,
                )

                best_row = copy.deepcopy(
                    row
                )

        mode_text = " ".join(
            f"m{i}={fraction:.3f}"
            for i, fraction
            in enumerate(
                fractions
            )
        )

        thresh_text = ""

        if (
            row[
                "obja_thresh_applied"
            ]
            > 0.5
        ):
            outside_pct = (
                100.0
                * (
                    row[
                        "obja_thresh_fraction_below"
                    ]
                    + row[
                        "obja_thresh_fraction_above"
                    ]
                )
            )

            thresh_text = (
                f" obja_thresh=["
                f"{row['obja_thresh_lower']:.2f},"
                f"{row['obja_thresh_upper']:.2f}]"
                f" outside_before="
                f"{outside_pct:.2f}%"
            )

        print(
            f"[iter {iteration:04d}] "
            f"stage={stage.name:<22} "
            f"data={mean_data:.6e} "
            f"score={score:.6e} "
            f"probe_anchor="
            f"{quality['probe_anchor_nrmse']:.4f} "
            f"pos_drift="
            f"{quality['position_rms_drift_px']:.4f}px "
            f"A=[{row['obja_min']:.3f},"
            f"{row['obja_max']:.3f}] "
            f"phi=[{row['objp_min']:.3f},"
            f"{row['objp_max']:.3f}] "
            f"guard={guard_scale:.2f}"
            f"{thresh_text} "
            f"{mode_text}"
        )

        should_save = (
            iteration
            % int(
                output_cfg[
                    "save_every"
                ]
            )
            == 0
            or iteration
            == iterations
        )

        if should_save:
            output.save_state_figures(
                iteration,
                state.obja,
                state.objp,
                state.probe,
                crop_positions,
                state.probe_pos_shifts,
                history,
            )

            if (
                last_measured
                is not None
                and last_predicted
                is not None
            ):
                output.save_forward_summary(
                    iteration,
                    last_measured,
                    last_predicted,
                )

        if (
            iteration
            % int(
                output_cfg[
                    "checkpoint_every"
                ]
            )
            == 0
            or iteration
            == iterations
        ):
            output.save_checkpoint(
                iteration,
                state,
                optimizer,
                crop_positions,
                history,
            )

            output.save_hdf5(
                iteration,
                obja=state.obja,
                objp=state.objp,
                probe=state.probe,
                crop_positions=crop_positions,
                probe_pos_shifts=state.probe_pos_shifts,
                metadata=metadata,
                loss=mean_data,
                intensity_loss=mean_int,
                selection_score_value=score,
            )

    synchronize_if_cuda(
        device
    )

    reconstruction_wall_seconds = (
        time.perf_counter()
        - reconstruction_wall_start
    )

    mean_core_seconds_per_iteration = (
        reconstruction_core_seconds
        / max(
            iterations,
            1,
        )
    )

    mean_wall_seconds_per_iteration = (
        reconstruction_wall_seconds
        / max(
            iterations,
            1,
        )
    )

    timing_summary = {
        "reconstruction_version": PLATFORM_RECONSTRUCTION_VERSION,
        "device": str(
            device
        ),
        "iterations": iterations,
        "full_dataset_passes": iterations,
        "minibatches_per_iteration": n_batches,
        "optimizer_steps_per_iteration": 1,
        "reconstruction_core_seconds": float(
            reconstruction_core_seconds
        ),
        "reconstruction_wall_seconds": float(
            reconstruction_wall_seconds
        ),
        "mean_core_seconds_per_iteration": float(
            mean_core_seconds_per_iteration
        ),
        "mean_wall_seconds_per_iteration": float(
            mean_wall_seconds_per_iteration
        ),
        "primary_runtime_for_benchmark_seconds": float(
            reconstruction_core_seconds
        ),
        "primary_runtime_scope": (
            "Synchronized reconstruction compute accumulated across all "
            "iterations; excludes post-iteration plotting/checkpoint/HDF5 "
            "and final-summary export."
        ),
        "wall_runtime_scope": (
            "Synchronized wall-clock across the reconstruction loop, "
            "including plotting/checkpoint/HDF5 work triggered inside it."
        ),
    }

    timing_path = (
        Path(
            output.root
        )
        / "timing_summary.json"
    )

    timing_path.write_text(
        json.dumps(
            timing_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 80
    )
    print(
        "RECONSTRUCTION TIMING"
    )
    print(
        "=" * 80
    )
    print(
        f"Iterations / full data passes : {iterations}"
    )
    print(
        f"Core reconstruction time       : "
        f"{reconstruction_core_seconds:.3f} s "
        f"({reconstruction_core_seconds / 60.0:.3f} min)"
    )
    print(
        f"Mean core time / iteration     : "
        f"{mean_core_seconds_per_iteration:.3f} s"
    )
    print(
        f"Loop wall-clock time           : "
        f"{reconstruction_wall_seconds:.3f} s "
        f"({reconstruction_wall_seconds / 60.0:.3f} min)"
    )
    print(
        f"Mean wall time / iteration     : "
        f"{mean_wall_seconds_per_iteration:.3f} s"
    )
    print(
        f"Timing JSON                    : {timing_path}"
    )
    print(
        "=" * 80
    )
    print(
        "Use 'Core reconstruction time' as the primary runtime "
        "for the ePIE / LSQML comparison."
    )
    print(
        "=" * 80
    )

    if not history:
        raise RuntimeError(
            "Reconstruction finished without any history rows."
        )

    last_snapshot = snapshot_state(
        state,
        crop_positions,
    )

    last_row = copy.deepcopy(
        history[
            -1
        ]
    )

    if (
        bool(
            selection_cfg.get(
                "enabled",
                True,
            )
        )
        and bool(
            selection_cfg.get(
                "restore_best_at_end",
                True,
            )
        )
        and best_snapshot
        is not None
    ):
        restore_snapshot(
            state,
            crop_positions,
            best_snapshot,
        )

        selected_iteration = int(
            best_iteration
        )

        selected_row = (
            best_row
            if best_row
            is not None
            else last_row
        )

    else:
        selected_iteration = (
            iterations
        )

        selected_row = (
            last_row
        )

    output.save_final_summary(
        obja=state.obja,
        objp=state.objp,
        probe=state.probe,
        crop_positions=crop_positions,
        probe_pos_shifts=state.probe_pos_shifts,
        history=history,
        metadata=metadata,
        selected_iteration=selected_iteration,
        selected_score=float(
            selected_row[
                "selection_score"
            ]
        ),
        last_iteration=iterations,
        last_snapshot=last_snapshot,
    )

    return {
        "device": str(
            device
        ),
        "reconstruction_version": PLATFORM_RECONSTRUCTION_VERSION,
        "iterations_run": iterations,
        "optimizer_steps_per_iteration": 1,
        "minibatches_per_iteration": n_batches,
        "selected_iteration": selected_iteration,
        "selected_data_loss": float(
            selected_row[
                "data_loss"
            ]
        ),
        "selected_quality_score": float(
            selected_row[
                "selection_score"
            ]
        ),
        "last_data_loss": float(
            last_row[
                "data_loss"
            ]
        ),
        "last_quality_score": float(
            last_row[
                "selection_score"
            ]
        ),
        "object_shape": list(
            state.obja.shape
        ),
        "probe_shape": list(
            state.probe.shape
        ),
        "probe_mode_fractions": [
            float(
                value
            )
            for value
            in mode_fractions(
                state.probe
            )
        ],
        "probe_anchor_nrmse": float(
            selected_row[
                "probe_anchor_nrmse"
            ]
        ),
        "position_rms_drift_px": float(
            selected_row[
                "position_rms_drift_px"
            ]
        ),
        "object_amplitude_threshold_enabled": obja_thresh_enabled,
        "object_amplitude_min": float(
            state.obja.detach()
            .min()
            .cpu()
        ),
        "object_amplitude_max": float(
            state.obja.detach()
            .max()
            .cpu()
        ),
        "reconstruction_core_seconds": float(
            reconstruction_core_seconds
        ),
        "reconstruction_wall_seconds": float(
            reconstruction_wall_seconds
        ),
        "mean_core_seconds_per_iteration": float(
            mean_core_seconds_per_iteration
        ),
        "mean_wall_seconds_per_iteration": float(
            mean_wall_seconds_per_iteration
        ),
        "primary_runtime_for_benchmark_seconds": float(
            reconstruction_core_seconds
        ),
        "timing_summary_path": str(
            timing_path
        ),
        "output_dir": str(
            output.root
        ),
    }


# ============================================================================
# Demo / smoke test
# ============================================================================


def run_synthetic_demo(
    config: dict[str, Any],
) -> None:
    device = choose_device(
        config[
            "project"
        ].get(
            "device",
            "auto",
        )
    )

    print(
        f"Demo device: {device}"
    )

    h_obj, w_obj = (
        32,
        32,
    )

    h_probe, w_probe = (
        8,
        8,
    )

    obja = torch.ones(
        (
            h_obj,
            w_obj,
        ),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    objp = torch.zeros(
        (
            h_obj,
            w_obj,
        ),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    probe = torch.ones(
        (
            2,
            h_probe,
            w_probe,
        ),
        dtype=torch.complex64,
        device=device,
        requires_grad=True,
    )

    crop = torch.tensor(
        [
            [
                4,
                5,
            ],
            [
                10,
                11,
            ],
        ],
        dtype=torch.long,
        device=device,
    )

    shifts = torch.tensor(
        [
            [
                0.25,
                -0.20,
            ],
            [
                -0.15,
                0.30,
            ],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    object_complex = build_complex_object(
        obja,
        objp,
    )

    predicted = forward_from_crop_and_shift(
        object_tensor=object_complex,
        probe=probe,
        crop_positions=crop,
        probe_shifts=shifts,
    )

    measured = (
        predicted.detach()
        * 1.01
    )

    loss_cfg = (
        config.get(
            "reconstruction",
            {},
        )
        .get(
            "data_loss",
            {},
        )
    )

    loss, _ = hybrid_data_loss(
        predicted,
        measured,
        amplitude_weight=float(
            loss_cfg.get(
                "amplitude_weight",
                0.0,
            )
        ),
        intensity_weight=float(
            loss_cfg.get(
                "intensity_weight",
                1.0,
            )
        ),
        eps=float(
            loss_cfg.get(
                "eps",
                1.0e-8,
            )
        ),
        charbonnier_eps=float(
            loss_cfg.get(
                "charbonnier_eps",
                1.0e-3,
            )
        ),
    )

    loss.backward()

    print(
        "V0.8.4 loss finite:",
        bool(
            torch.isfinite(
                loss
            )
        ),
    )

    print(
        "obja grad finite:",
        bool(
            torch.isfinite(
                obja.grad
            ).all()
        ),
    )

    print(
        "objp grad finite:",
        bool(
            torch.isfinite(
                objp.grad
            ).all()
        ),
    )

    print(
        "probe grad finite:",
        bool(
            torch.isfinite(
                probe.grad
            ).all()
        ),
    )

    print(
        "position grad finite:",
        bool(
            torch.isfinite(
                shifts.grad
            ).all()
        ),
    )
