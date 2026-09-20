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
import ptychi.utils as putils
from ptychi.api.task import PtychographyTask


def electron_wavelength_angstrom(kv: float) -> float:
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
    kv: float,
    conv_angle_mrad: float,
    npix: int,
    dx_angstrom: float,
    c10_angstrom: float = 0.0,
) -> np.ndarray:
    wavelength = electron_wavelength_angstrom(kv)
    k_aperture = (float(conv_angle_mrad) / 1.0e3) / wavelength
    f = np.fft.fftshift(np.fft.fftfreq(npix, d=float(dx_angstrom))).astype(np.float64)
    ky, kx = np.meshgrid(f, f, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)
    aperture = (kr <= k_aperture).astype(np.float64)
    alpha_r = kr * wavelength
    chi = math.pi * float(c10_angstrom) * alpha_r**2 / wavelength
    pupil = aperture * np.exp(-1j * chi)
    probe = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil)))
    norm = np.sqrt(np.sum(np.abs(probe) ** 2))
    probe = probe / max(float(norm), 1.0e-12)
    return probe.astype(np.complex64)


def center_positions(positions_yx: np.ndarray) -> np.ndarray:
    p = np.asarray(positions_yx, dtype=np.float32).copy()
    center = 0.5 * (p.max(axis=0) + p.min(axis=0))
    p -= center[None, :]
    return p


def save_summary_figure(obj: np.ndarray, probe: np.ndarray, output_path: Path) -> None:
    obj2d = np.prod(obj, axis=0) if obj.ndim == 3 else np.asarray(obj)
    p = np.asarray(probe)
    if p.ndim == 4:
        total_probe_intensity = np.sum(np.abs(p) ** 2, axis=(0, 1))
    elif p.ndim == 3:
        total_probe_intensity = np.sum(np.abs(p) ** 2, axis=0)
    else:
        total_probe_intensity = np.abs(p) ** 2

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    im = axs[0].imshow(np.abs(obj2d))
    axs[0].set_title("LSQML object amplitude")
    axs[0].set_axis_off()
    fig.colorbar(im, ax=axs[0], shrink=0.8)

    im = axs[1].imshow(np.angle(obj2d))
    axs[1].set_title("LSQML object phase")
    axs[1].set_axis_off()
    fig.colorbar(im, ax=axs[1], shrink=0.8)

    im = axs[2].imshow(total_probe_intensity)
    axs[2].set_title("LSQML total probe intensity")
    axs[2].set_axis_off()
    fig.colorbar(im, ax=axs[2], shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Pty-Chi LSQML on V8-prepared abTEM Dataset B/C."
    )
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--probe-mode-init-power", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-c10-A", type=float, default=0.0)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.5)
    parser.add_argument("--optimal-step-size-scaler", type=float, default=0.9)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prepared_path = Path(args.prepared)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(prepared_path, allow_pickle=False) as data:
        required = {"measured_intensity", "scan_positions", "ground_truth_object", "metadata_json"}
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
        raise ValueError("Square detector required.")

    dx_A = float(metadata["dx_angstrom_per_object_pixel"])
    kv = float(metadata.get("probe_kv", 80.0))
    conv_mrad = float(metadata.get("probe_conv_angle_mrad", 24.9))
    object_shape = tuple(int(v) for v in metadata["recommended_object_shape"])

    wavelength_A = electron_wavelength_angstrom(kv)
    wavelength_m = wavelength_A * 1.0e-10
    pixel_size_m = dx_A * 1.0e-10

    positions_centered_yx = center_positions(positions_yx)

    primary_probe = simulate_stem_probe(
        kv=kv,
        conv_angle_mrad=conv_mrad,
        npix=detector_h,
        dx_angstrom=dx_A,
        c10_angstrom=float(args.probe_c10_A),
    )

    probe_guess = np.zeros(
        (1, int(args.modes), detector_h, detector_w),
        dtype=np.complex64,
    )
    probe_guess[0, 0] = primary_probe

    if int(args.modes) > 1:
        probe_t = torch.from_numpy(probe_guess.copy())
        probe_t = putils.orthogonalize_initial_probe(
            probe_t,
            secondary_mode_energy=float(args.probe_mode_init_power),
        )
        probe_guess = probe_t.detach().cpu().numpy().astype(np.complex64)

    probe_power_before_scale = float(np.sum(np.abs(probe_guess) ** 2))
    probe_guess = putils.rescale_probe(probe_guess, measured).astype(np.complex64)
    probe_power_after_scale = float(np.sum(np.abs(probe_guess) ** 2))

    rng = np.random.default_rng(args.seed)
    object_guess = np.ones((1, object_shape[0], object_shape[1]), dtype=np.complex64)
    object_guess *= np.exp(
        1j * (1.0e-8 * rng.random(object_guess.shape))
    ).astype(np.complex64)

    options = api.LSQMLOptions()

    options.data_options.fft_shift = True
    options.data_options.wavelength_m = wavelength_m
    options.data_options.save_data_on_device = True

    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = 1.0
    options.object_options.optimal_step_size_scaler = float(args.optimal_step_size_scaler)
    options.object_options.multimodal_update = True
    options.object_options.remove_object_probe_ambiguity.enabled = True

    options.probe_options.pixel_size_m = pixel_size_m
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = 1.0
    options.probe_options.optimal_step_size_scaler = float(args.optimal_step_size_scaler)
    options.probe_options.orthogonalize_incoherent_modes.enabled = int(args.modes) > 1
    options.probe_options.orthogonalize_incoherent_modes.sort_by_occupancy = True

    options.probe_position_options.optimizable = False

    ro = options.reconstructor_options
    ro.num_epochs = int(args.epochs)
    ro.batch_size = int(args.batch_size)
    ro.random_seed = int(args.seed)
    ro.gaussian_noise_std = float(args.gaussian_noise_std)
    ro.single_slice_solve_obj_prb_step_size_jointly = True
    ro.solve_step_sizes_only_using_first_probe_mode = True
    ro.momentum_acceleration_gain = 0.0
    ro.preconditioning_damping_factor = 0.1

    # Already power-matched above, matching the corrected ePIE initialization.
    ro.rescale_probe_intensity_in_first_epoch = False

    n_batches = math.ceil(measured.shape[0] / int(args.batch_size))

    print("=" * 104)
    print("PTY-CHI LSQML — V8 DATASET B/C BENCHMARK")
    print("=" * 104)
    print("prepared                 :", prepared_path)
    print("dataset                  :", metadata.get("dataset_name", ""))
    print("torch                    :", torch.__version__)
    print("torch CUDA               :", torch.version.cuda)
    print("GPU                      :", torch.cuda.get_device_name(0))
    print("ptychi                   :", getattr(ptychi, "__version__", "installed"))
    print("diffraction shape        :", measured.shape)
    print("object guess shape       :", object_guess.shape)
    print("probe guess shape        :", probe_guess.shape)
    print("dx                       :", dx_A, "A/px")
    print("wavelength               :", wavelength_A, "A")
    print("modes                    :", args.modes)
    print("epochs                   :", args.epochs)
    print("batch size               :", args.batch_size)
    print("batches / epoch          :", n_batches)
    print("total minibatches        :", n_batches * int(args.epochs))
    print("Gaussian noise std       :", args.gaussian_noise_std)
    print("optimal step scaler      :", args.optimal_step_size_scaler)
    print("probe power pre-scale    :", probe_power_before_scale)
    print("probe power post-scale   :", probe_power_after_scale)
    print("positions optimized      :", False)
    print("=" * 104)

    task = PtychographyTask(
        options,
        diffraction_data=measured,
        object_data=object_guess,
        probe_data=probe_guess,
        probe_position_x_px=positions_centered_yx[:, 1],
        probe_position_y_px=positions_centered_yx[:, 0],
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    print("Starting Pty-Chi LSQML...")
    task.run()
    torch.cuda.synchronize()
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

    p = np.asarray(probe_final)
    if p.ndim == 4:
        mode_power = np.sum(np.abs(p) ** 2, axis=(0, 2, 3))
    elif p.ndim == 3:
        mode_power = np.sum(np.abs(p) ** 2, axis=(1, 2))
    else:
        mode_power = np.array([np.sum(np.abs(p) ** 2)])

    mode_frac = mode_power / max(float(mode_power.sum()), 1e-20)

    summary = {
        "method": "Pty-Chi LSQML",
        "prepared": str(prepared_path),
        "dataset_name": metadata.get("dataset_name", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ptychi_version": getattr(ptychi, "__version__", "installed"),
        "modes": int(args.modes),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "gaussian_noise_std": float(args.gaussian_noise_std),
        "optimal_step_size_scaler": float(args.optimal_step_size_scaler),
        "elapsed_seconds": elapsed,
        "probe_mode_power_fractions": [float(x) for x in mode_frac],
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
        output_dir / "lsqml_final_summary.png",
    )

    print()
    print("=" * 104)
    print("LSQML COMPLETE")
    print("=" * 104)
    print("elapsed seconds         :", f"{elapsed:.3f}")
    print("object final shape      :", object_final.shape)
    print("probe final shape       :", probe_final.shape)
    print("probe mode fractions    :", [float(x) for x in mode_frac])
    print("saved to                :", output_dir)
    print("=" * 104)


if __name__ == "__main__":
    main()
