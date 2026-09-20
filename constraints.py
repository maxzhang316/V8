"""Stability-oriented projections for Ptychography Platform V0.8.2.1.

This version keeps the V8 fixed-position reconstruction strategy and adds a
PtyRAD-style object-amplitude threshold constraint (``obja_thresh``).

The amplitude threshold follows the PtyRAD relaxation convention:

    post = relax * pre + (1 - relax) * clip(pre, lower, upper)

Therefore:

- relax = 0.0 -> hard threshold / hard projection;
- relax = 1.0 -> no-op;
- 0 < relax < 1 -> relaxed thresholding.

Other V8 projections remain unchanged:

- optional object smoothing;
- positive object phase;
- fixed probe total power;
- mixed-mode orthogonalization/sorting;
- position re-basing/re-centering when position optimization is enabled.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _active(cfg: dict, iteration: int) -> bool:
    if not cfg.get("enabled", False):
        return False
    start = int(cfg.get("start_iter", 1))
    end = cfg.get("end_iter")
    step = int(cfg.get("step", 1))
    if iteration < start:
        return False
    if end is not None and iteration > int(end):
        return False
    return (iteration - start) % max(step, 1) == 0


def _gaussian_kernel_1d(
    kernel_size: int,
    std: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kernel_size % 2 != 1:
        raise ValueError("Gaussian kernel_size must be odd.")
    if std <= 0:
        raise ValueError("Gaussian std must be positive.")
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(std)).square())
    return kernel / kernel.sum().clamp_min(1e-12)


def gaussian_blur_2d(
    image: torch.Tensor,
    kernel_size: int,
    std: float,
) -> torch.Tensor:
    if std <= 0:
        return image
    kernel_1d = _gaussian_kernel_1d(
        int(kernel_size),
        float(std),
        device=image.device,
        dtype=image.dtype,
    )
    kernel_2d = torch.outer(kernel_1d, kernel_1d)[None, None]
    radius = int(kernel_size) // 2
    x = image[None, None]
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(x, kernel_2d)[0, 0]


def orthogonalize_and_sort_probe_modes(probe: torch.Tensor) -> torch.Tensor:
    """Rotate a mixed-mode basis into orthogonal power-sorted modes.

    A unitary mode rotation preserves the incoherent detector intensity, so this
    projection improves conditioning without changing the represented mixed
    state (up to numerical precision).
    """
    if probe.ndim != 3:
        raise ValueError("probe must be [M,H,W]")
    m, h, w = probe.shape
    if m <= 1:
        return probe

    flat = probe.reshape(m, -1)
    gram = flat @ torch.conj(flat).T
    gram = 0.5 * (gram + torch.conj(gram).T)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals.real, descending=True)
    eigvecs = eigvecs[:, order]
    transformed = torch.conj(eigvecs).T @ flat
    return transformed.reshape(m, h, w).to(torch.complex64)


@torch.no_grad()
def threshold_object_amplitude_(
    obja: torch.Tensor,
    *,
    lower: float,
    upper: float,
    relax: float,
) -> dict[str, float]:
    """Apply a relaxed threshold around unity to object amplitude in-place.

    The thresholded target is ``clip(obja, lower, upper)``.  Relaxation uses a
    weighted sum between the pre-threshold and post-threshold values:

        obja <- relax * obja + (1 - relax) * clipped

    Hence ``relax=0`` is a hard threshold and ``relax=1`` leaves the amplitude
    unchanged.  This matches the semantics used by PtyRAD's ``obja_thresh``.
    """
    lower = float(lower)
    upper = float(upper)
    relax = float(relax)

    if not lower <= upper:
        raise ValueError(
            f"Invalid obja_thresh: lower ({lower}) must be <= upper ({upper})."
        )
    if not 0.0 <= relax <= 1.0:
        raise ValueError(
            f"Invalid obja_thresh relax={relax}; expected a value in [0, 1]."
        )

    before_min = float(obja.min().detach().cpu())
    before_max = float(obja.max().detach().cpu())
    below_fraction = float((obja < lower).float().mean().detach().cpu())
    above_fraction = float((obja > upper).float().mean().detach().cpu())

    clipped = torch.clamp(obja, min=lower, max=upper)
    if relax <= 0.0:
        obja.copy_(clipped)
    elif relax < 1.0:
        obja.mul_(relax).add_(clipped, alpha=1.0 - relax)
    # relax == 1.0 is intentionally a no-op.

    return {
        "obja_thresh_applied": 1.0,
        "obja_thresh_lower": lower,
        "obja_thresh_upper": upper,
        "obja_thresh_relax": relax,
        "obja_thresh_fraction_below": below_fraction,
        "obja_thresh_fraction_above": above_fraction,
        "obja_min_before_thresh": before_min,
        "obja_max_before_thresh": before_max,
        "obja_min_after_thresh": float(obja.min().detach().cpu()),
        "obja_max_after_thresh": float(obja.max().detach().cpu()),
        "obja_mean_after_thresh": float(obja.mean().detach().cpu()),
    }


@torch.no_grad()
def limit_position_iteration_step_(
    crop_positions: torch.Tensor,
    probe_pos_shifts: torch.Tensor,
    previous_continuous_positions: torch.Tensor,
    max_step_px: float,
) -> dict[str, float]:
    """Limit the per-scan continuous-position move during one full iteration."""
    if max_step_px <= 0:
        return {
            "position_step_max_before_clip": 0.0,
            "position_step_clipped_fraction": 0.0,
        }

    current = crop_positions.to(torch.float32) + probe_pos_shifts
    delta = current - previous_continuous_positions
    norms = torch.linalg.vector_norm(delta, dim=1)
    scale = torch.clamp(float(max_step_px) / norms.clamp_min(1e-12), max=1.0)
    clipped = previous_continuous_positions + delta * scale[:, None]
    probe_pos_shifts.copy_(clipped - crop_positions.to(torch.float32))

    return {
        "position_step_max_before_clip": float(norms.max().cpu()),
        "position_step_clipped_fraction": float(
            (scale < 0.999999).float().mean().cpu()
        ),
    }


@torch.no_grad()
def rebase_positions_(
    crop_positions: torch.Tensor,
    probe_pos_shifts: torch.Tensor,
    *,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
) -> dict[str, float]:
    """Re-represent the same continuous coordinates with nearby integer crops.

    continuous = crop + shift is preserved exactly (apart from float rounding).
    The integer crop is updated to round(continuous), while the remaining shift
    becomes fractional.  This keeps Fourier translations well conditioned and
    prevents a handful of scan points from accumulating multi-pixel shift
    parameters merely because their integer crop was never re-based.
    """
    continuous = crop_positions.to(torch.float32) + probe_pos_shifts
    new_crop = torch.round(continuous).to(torch.long)

    max_y = int(object_shape[0]) - int(probe_shape[0])
    max_x = int(object_shape[1]) - int(probe_shape[1])
    new_crop[:, 0].clamp_(0, max_y)
    new_crop[:, 1].clamp_(0, max_x)

    new_shift = continuous - new_crop.to(torch.float32)
    crop_positions.copy_(new_crop)
    probe_pos_shifts.copy_(new_shift)

    return {
        "position_shift_abs_max_after_rebase": float(new_shift.abs().max().cpu()),
        "position_shift_std_y": float(new_shift[:, 0].std().cpu()),
        "position_shift_std_x": float(new_shift[:, 1].std().cpu()),
    }


@torch.no_grad()
def apply_v8_projections(
    *,
    obja: torch.Tensor,
    objp: torch.Tensor,
    probe: torch.Tensor,
    crop_positions: torch.Tensor,
    probe_pos_shifts: torch.Tensor,
    fixed_probe_power: float,
    constraints_cfg: dict,
    iteration: int,
    probe_active: bool,
    positions_active: bool,
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    # ---------------------------------------------------------------------
    # Object blur
    # ---------------------------------------------------------------------
    blur_cfg = constraints_cfg.get("object_blur", {})
    if _active(blur_cfg, iteration):
        kernel_size = int(blur_cfg.get("kernel_size", 5))
        std = float(blur_cfg.get("std", 0.25))
        obj_type = str(blur_cfg.get("obj_type", "both")).lower()
        if obj_type in ("both", "amp", "amplitude", "obja"):
            obja.copy_(gaussian_blur_2d(obja, kernel_size, std))
        if obj_type in ("both", "phase", "objp"):
            objp.copy_(gaussian_blur_2d(objp, kernel_size, std))
        metrics["object_blur_applied"] = 1.0
    else:
        metrics["object_blur_applied"] = 0.0

    # ---------------------------------------------------------------------
    # PtyRAD-style object-amplitude threshold around unity.
    # Recommended initial setting for the current 2D MoS2 benchmark:
    # thresh=[0.85, 1.15], relax=0.02.
    # ---------------------------------------------------------------------
    obja_thresh_cfg = constraints_cfg.get("obja_thresh", {})
    if _active(obja_thresh_cfg, iteration):
        thresh = obja_thresh_cfg.get("thresh", [0.85, 1.15])
        if not isinstance(thresh, (list, tuple)) or len(thresh) != 2:
            raise ValueError(
                "constraints.obja_thresh.thresh must be [lower, upper]."
            )
        metrics.update(
            threshold_object_amplitude_(
                obja,
                lower=float(thresh[0]),
                upper=float(thresh[1]),
                relax=float(obja_thresh_cfg.get("relax", 0.0)),
            )
        )
    else:
        metrics["obja_thresh_applied"] = 0.0

    # ---------------------------------------------------------------------
    # Positive object phase
    # ---------------------------------------------------------------------
    positive_cfg = constraints_cfg.get("objp_positive", {})
    if _active(positive_cfg, iteration):
        relax = float(positive_cfg.get("relax", 0.0))
        objp.clamp_(min=-relax)

    # ---------------------------------------------------------------------
    # Probe constraints
    # ---------------------------------------------------------------------
    if probe_active:
        ortho_cfg = constraints_cfg.get("ortho_pmode", {})
        if _active(ortho_cfg, iteration):
            probe.copy_(orthogonalize_and_sort_probe_modes(probe))
            metrics["probe_ortho_applied"] = 1.0
        else:
            metrics["probe_ortho_applied"] = 0.0

        fix_cfg = constraints_cfg.get("fix_probe_int", {})
        if _active(fix_cfg, iteration):
            current = torch.sum(torch.abs(probe).square()).real.clamp_min(1e-12)
            target = torch.as_tensor(
                fixed_probe_power,
                device=probe.device,
                dtype=current.dtype,
            )
            probe.mul_(torch.sqrt(target / current))

    # ---------------------------------------------------------------------
    # Position constraints
    # ---------------------------------------------------------------------
    if positions_active:
        rebase_cfg = constraints_cfg.get("position_rebase", {})
        if _active(rebase_cfg, iteration):
            metrics.update(
                rebase_positions_(
                    crop_positions,
                    probe_pos_shifts,
                    object_shape=object_shape,
                    probe_shape=probe_shape,
                )
            )

        recenter_cfg = constraints_cfg.get("pos_recenter", {})
        if _active(recenter_cfg, iteration):
            relax = float(recenter_cfg.get("relax", 0.0))
            mean_shift = probe_pos_shifts.mean(dim=0, keepdim=True)
            probe_pos_shifts.sub_((1.0 - relax) * mean_shift)
            metrics["position_mean_y_after_recenter"] = float(
                probe_pos_shifts[:, 0].mean().cpu()
            )
            metrics["position_mean_x_after_recenter"] = float(
                probe_pos_shifts[:, 1].mean().cpu()
            )

    return metrics
