from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ptychi
import ptychi.api as api


def electron_wavelength_angstrom(kv: float) -> float:
    """Relativistic electron wavelength in Angstrom."""
    v = float(kv) * 1.0e3
    h = 6.62607015e-34
    m = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0
    lam_m = h / math.sqrt(2.0 * m * e * v * (1.0 + e * v / (2.0 * m * c * c)))
    return lam_m * 1.0e10


def load_metadata(raw):
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def simulate_stem_probe(
    *,
    kv: float,
    conv_angle_mrad: float,
    npix: int,
    dx_angstrom: float,
    c10_angstrom: float = 0.0,
) -> np.ndarray:
    """Match the physical single-mode STEM probe used by the in-house V0.8.x code."""
    wavelength = electron_wavelength_angstrom(kv)
    k_aperture = (float(conv_angle_mrad) / 1.0e3) / wavelength

    f = np.fft.fftshift(np.fft.fftfreq(npix, d=float(dx_angstrom))).astype(np.float64)
    ky, kx = np.meshgrid(f, f, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)

    aperture = (kr <= k_aperture).astype(np.float64)

    alpha_r = kr * wavelength
    chi = math.pi * float(c10_angstrom) * alpha_r**2 / wavelength

    pupil = aperture * np.exp(-1j * chi)
    probe = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(pupil)
        )
    )

    norm = np.sqrt(np.sum(np.abs(probe) ** 2))
    probe = probe / max(float(norm), 1.0e-12)
    return probe.astype(np.complex64)


def orthogonalize_candidate(
    candidate: np.ndarray,
    basis: list[np.ndarray],
) -> np.ndarray:
    flat = candidate.reshape(-1).astype(np.complex128)

    for existing in basis:
        b = existing.reshape(-1).astype(np.complex128)
        denom = np.vdot(b, b).real
        coeff = np.vdot(b, flat) / max(float(denom), 1.0e-12)
        flat = flat - coeff * b

    norm = np.linalg.norm(flat)
    flat = flat / max(float(norm), 1.0e-12)
    return flat.reshape(candidate.shape).astype(np.complex64)


def initialize_mixed_probe(
    primary_probe: np.ndarray,
    n_modes: int,
    new_mode_power: float = 0.02,
) -> np.ndarray:
    """
    Same basic deterministic Hermite-like initialization used by the
    in-house V0.8.x baseline.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    if n_modes == 1:
        return primary_probe[None]

    h, w = primary_probe.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(yy, xx, indexing="ij")

    modifiers = [
        gx,
        gy,
        gx * gy,
        gx**2 - gy**2,
        2.0 * gx**2 - 1.0,
        2.0 * gy**2 - 1.0,
    ]

    primary_norm = np.linalg.norm(primary_probe.reshape(-1))
    modes = [primary_probe.astype(np.complex64)]

    for i in range(1, n_modes):
        modifier = modifiers[(i - 1) % len(modifiers)]
        candidate = primary_probe * modifier.astype(np.float32)
        candidate = orthogonalize_candidate(candidate, modes)
        candidate = candidate * primary_norm * math.sqrt(float(new_mode_power))
        modes.append(candidate.astype(np.complex64))

    probe = np.stack(modes, axis=0)
    power = np.sum(np.abs(probe.astype(np.complex128)) ** 2)
    probe = probe / math.sqrt(max(float(power), 1.0e-12))
    return probe.astype(np.complex64)


def center_positions(positions_yx: np.ndarray) -> np.ndarray:
    """
    Absolute scan origin is arbitrary in Pty-Chi. Preserve all relative/subpixel
    positions while centering the scan range around zero.
    """
    p = np.asarray(positions_yx, dtype=np.float32).copy()
    center = 0.5 * (p.max(axis=0) + p.min(axis=0))
    p -= center[None, :]
    return p


def save_summary_figure(
    obj: np.ndarray,
    probe: np.ndarray,
    output_path: Path,
) -> None:
    # Object shape: (n_slices, H, W)
    obj2d = np.prod(obj, axis=0) if obj.ndim == 3 else np.asarray(obj)

    # Probe shape: (n_opr_modes, n_incoherent_modes, H, W)
    p = np.asarray(probe)
    if p.ndim == 4:
        total_probe_intensity = np.sum(np.abs(p) ** 2, axis=(0, 1))
    elif p.ndim == 3:
        total_probe_intensity = np.sum(np.abs(p) ** 2, axis=0)
    else:
        total_probe_intensity = np.abs(p) ** 2

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    im = axs[0].imshow(np.abs(obj2d))
    axs[0].set_title("ePIE object amplitude")
    axs[0].set_axis_off()
    fig.colorbar(im, ax=axs[0], shrink=0.8)

    im = axs[1].imshow(np.angle(obj2d))
    axs[1].set_title("ePIE object phase")
    axs[1].set_axis_off()
    fig.colorbar(im, ax=axs[1], shrink=0.8)

    im = axs[2].imshow(total_probe_intensity)
    axs[2].set_title("ePIE total probe intensity")
    axs[2].set_axis_off()
    fig.colorbar(im, ax=axs[2], shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Pty-Chi ePIE on a V8-prepared abTEM dataset. "
            "The measurement values are passed unchanged."
        )
    )
    parser.add_argument("--prepared", required=True, help="V8 prepared .npz")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--modes", type=int, default=1, help="Number of incoherent probe modes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 for a classical sequential ePIE-style baseline.",
    )
    parser.add_argument("--object-alpha", type=float, default=0.1)
    parser.add_argument("--probe-alpha", type=float, default=0.1)
    parser.add_argument("--probe-mode-init-power", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-c10-A", type=float, default=0.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prepared_path = Path(args.prepared)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(prepared_path, allow_pickle=False) as data:
        required = {
            "measured_intensity",
            "scan_positions",
            "ground_truth_object",
            "metadata_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Prepared dataset missing keys: {sorted(missing)}")

        measured = np.asarray(data["measured_intensity"], dtype=np.float32)
        positions_yx = np.asarray(data["scan_positions"], dtype=np.float32)
        gt_object = np.asarray(data["ground_truth_object"], dtype=np.complex64)
        metadata = load_metadata(data["metadata_json"])

    if measured.ndim != 3:
        raise ValueError(f"Expected diffraction shape (N,H,W), got {measured.shape}")
    if positions_yx.shape != (measured.shape[0], 2):
        raise ValueError(
            f"Position shape mismatch: measured={measured.shape}, positions={positions_yx.shape}"
        )

    detector_h, detector_w = measured.shape[-2:]
    if detector_h != detector_w:
        raise ValueError("This benchmark script currently expects a square detector.")

    dx_A = float(metadata["dx_angstrom_per_object_pixel"])
    kv = float(metadata.get("probe_kv", 80.0))
    conv_mrad = float(metadata.get("probe_conv_angle_mrad", 24.9))
    object_shape = tuple(int(v) for v in metadata["recommended_object_shape"])

    wavelength_A = electron_wavelength_angstrom(kv)
    wavelength_m = wavelength_A * 1.0e-10
    pixel_size_m = dx_A * 1.0e-10

    # Preserve the exact V8 relative/subpixel scan geometry.
    positions_centered_yx = center_positions(positions_yx)

    primary_probe = simulate_stem_probe(
        kv=kv,
        conv_angle_mrad=conv_mrad,
        npix=detector_h,
        dx_angstrom=dx_A,
        c10_angstrom=float(args.probe_c10_A),
    )

    modes = initialize_mixed_probe(
        primary_probe,
        n_modes=int(args.modes),
        new_mode_power=float(args.probe_mode_init_power),
    )

    # Pty-Chi probe shape: (n_opr_modes, n_incoherent_modes, H, W).
    probe_guess = modes[None, ...].astype(np.complex64)

    # Single-slice object: (1,H,W).
    object_guess = np.ones(
        (1, object_shape[0], object_shape[1]),
        dtype=np.complex64,
    )
    object_guess *= np.exp(
        1j
        * (
            1.0e-8
            * np.random.default_rng(args.seed).random(object_guess.shape)
        )
    ).astype(np.complex64)

    options = api.EPIEOptions()

    # Our V8 prepared diffraction patterns have DC at the detector center.
    # Pty-Chi's far-field forward has DC at the top-left, so its documented
    # fft_shift=True preprocessing is the correct setting here.
    options.data_options.fft_shift = True
    options.data_options.wavelength_m = wavelength_m

    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.alpha = float(args.object_alpha)

    options.probe_options.pixel_size_m = pixel_size_m
    options.probe_options.optimizable = True
    options.probe_options.alpha = float(args.probe_alpha)

    # Pty-Chi defaults to SVD orthogonalization for incoherent modes.
    # Keep it enabled for the mixed-state ePIE benchmark.
    options.probe_options.orthogonalize_incoherent_modes.enabled = (
        int(args.modes) > 1
    )
    options.probe_options.orthogonalize_incoherent_modes.sort_by_occupancy = True

    options.probe_position_options.optimizable = False

    options.reconstructor_options.num_epochs = int(args.epochs)
    options.reconstructor_options.batch_size = int(args.batch_size)
    options.reconstructor_options.random_seed = int(args.seed)

    print("=" * 100)
    print("PTY-CHI ePIE V8 BENCHMARK")
    print("=" * 100)
    print("prepared              :", prepared_path)
    print("dataset               :", metadata.get("dataset_name", ""))
    print("torch                 :", torch.__version__)
    print("torch CUDA            :", torch.version.cuda)
    print("GPU                   :", torch.cuda.get_device_name(0))
    print("ptychi module         :", getattr(ptychi, "__version__", "installed"))
    print("diffraction shape     :", measured.shape)
    print("object guess shape    :", object_guess.shape)
    print("probe guess shape     :", probe_guess.shape)
    print("scan position range y :", float(positions_centered_yx[:, 0].min()),
          "to", float(positions_centered_yx[:, 0].max()), "px")
    print("scan position range x :", float(positions_centered_yx[:, 1].min()),
          "to", float(positions_centered_yx[:, 1].max()), "px")
    print("dx                    :", dx_A, "A/px")
    print("wavelength            :", wavelength_A, "A")
    print("modes                 :", args.modes)
    print("epochs                :", args.epochs)
    print("batch size            :", args.batch_size)
    print("object alpha          :", args.object_alpha)
    print("probe alpha           :", args.probe_alpha)
    print("=" * 100)

    task = api.PtychographyTask(
        options,
        diffraction_data=measured,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_centered_yx[:, 1],
        probe_position_y_px=positions_centered_yx[:, 0],
    )

    start = time.perf_counter()
    task.run()
    elapsed = time.perf_counter() - start

    object_final = task.get_data_to_cpu("object", as_numpy=True)
    probe_final = task.get_data_to_cpu("probe", as_numpy=True)
    positions_final = task.get_data_to_cpu("probe_positions", as_numpy=True)

    np.save(output_dir / "object_final.npy", object_final, allow_pickle=False)
    np.save(output_dir / "probe_final.npy", probe_final, allow_pickle=False)
    np.save(output_dir / "probe_positions_final.npy", positions_final, allow_pickle=False)
    np.save(output_dir / "ground_truth_object.npy", gt_object, allow_pickle=False)

    with (output_dir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(task.get_options_as_dict(), f, indent=2, default=str)

    summary = {
        "prepared": str(prepared_path),
        "dataset_name": metadata.get("dataset_name", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "modes": int(args.modes),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "object_alpha": float(args.object_alpha),
        "probe_alpha": float(args.probe_alpha),
        "dx_angstrom_per_pixel": dx_A,
        "wavelength_angstrom": wavelength_A,
        "elapsed_seconds": elapsed,
        "object_final_shape": list(object_final.shape),
        "probe_final_shape": list(probe_final.shape),
    }

    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    save_summary_figure(
        object_final,
        probe_final,
        output_dir / "epie_final_summary.png",
    )

    print()
    print("=" * 100)
    print("ePIE COMPLETE")
    print("=" * 100)
    print("elapsed seconds       :", f"{elapsed:.3f}")
    print("object final shape    :", object_final.shape)
    print("probe final shape     :", probe_final.shape)
    print("saved to              :", output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
