"""Unified V0.8 platform entry point: Adam (default), Pty-Chi ePIE, Pty-Chi LSQML.

The existing Adam implementation and its YAML schema are left unchanged. The
Pty-Chi baselines are launched as separate Python processes so an existing
validated Pty-Chi environment may be used without changing the V8 environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml


METHODS = ("adam", "epie", "lsqml")
BENCHMARK_RUNNERS = {
    "epie": "run_ptychi_epie_v8.py",
    "lsqml": "run_ptychi_lsqml_v8.py",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    """Resolve legacy cwd-relative paths first, then relative to YAML location."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    yaml_path = (config_path.parent / path).resolve()
    return cwd_path if cwd_path.exists() or not yaml_path.exists() else yaml_path


def _existing_path(value: str | Path, config_path: Path, purpose: str) -> Path:
    path = _resolve_path(value, config_path)
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} not found: {path}")
    return path


def _optional_ptychi_settings(config: dict) -> dict:
    integrations = config.get("integrations", {})
    if not isinstance(integrations, dict):
        raise ValueError("'integrations' must be a YAML mapping")
    settings = integrations.get("ptychi", {})
    if not isinstance(settings, dict):
        raise ValueError("'integrations.ptychi' must be a YAML mapping")
    return settings


def _select_method(config: dict, specified_method: str | None, interactive: bool) -> str:
    # Default is always Adam when --method is not provided, regardless of YAML.
    chosen = (specified_method or "adam").lower().strip()
    if chosen not in METHODS:
        raise ValueError(f"Unknown method {chosen!r}; use one of: {', '.join(METHODS)}")
    if not interactive:
        return chosen

    print("\nChoose reconstruction method:")
    print("  1. Adam (V8, default)")
    print("  2. ePIE (Pty-Chi)")
    print("  3. LSQML (Pty-Chi)")
    answer = input("Method [Enter=Adam]: ").strip().lower()
    result = {"": "adam", "1": "adam", "2": "epie", "3": "lsqml"}.get(answer, answer)
    if result not in METHODS:
        raise ValueError(f"Unknown selection {answer!r}; choose 1, 2, 3 or press Enter")
    return result


def _validate_benchmark_npz(prepared: Path) -> None:
    """These *existing* Pty-Chi runners require synthetic GT: fail clearly here."""
    required = {"measured_intensity", "scan_positions", "ground_truth_object", "metadata_json"}
    if not zipfile.is_zipfile(prepared):
        raise ValueError(f"Prepared dataset is not a valid .npz archive: {prepared}")
    with zipfile.ZipFile(prepared) as archive:
        fields = {Path(name).stem for name in archive.namelist() if name.endswith(".npy")}
    missing = required - fields
    if missing:
        raise ValueError(
            f"The existing Pty-Chi synthetic benchmark runners require {sorted(missing)} "
            f"in {prepared.name}. For real datasets without GT, the runners must be "
            "extended to support optional ground_truth_object first."
        )


def _runner_script(
    method: str, args: argparse.Namespace, config: dict, config_path: Path
) -> Path:
    settings = _optional_ptychi_settings(config)
    override = getattr(args, f"{method}_runner")
    configured = settings.get(f"{method}_script")
    if override or configured:
        return _existing_path(override or configured, config_path, f"{method} runner")

    dirname = args.ptychi_runner_dir or settings.get("runner_dir")
    if dirname:
        directory = _resolve_path(dirname, config_path)
        return _existing_path(directory / BENCHMARK_RUNNERS[method], config_path, f"{method} runner")

    here = Path(__file__).resolve().parent
    candidates = [
        here / "engines" / "ptychi" / BENCHMARK_RUNNERS[method],
        here / BENCHMARK_RUNNERS[method],  # backward-compatible legacy location
        here.parent / "ptychi_benchmark" / BENCHMARK_RUNNERS[method],
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {BENCHMARK_RUNNERS[method]}. Place it under "
        "V08/engines/ptychi/, or specify --epie-runner/--lsqml-runner."
    )


def _ptychi_python(args: argparse.Namespace, config: dict, config_path: Path) -> Path:
    settings = _optional_ptychi_settings(config)
    value = (
        args.ptychi_python
        or settings.get("python")
        or os.environ.get("PTYCHI_PYTHON")
        or sys.executable
    )
    return _existing_path(value, config_path, "Pty-Chi Python interpreter")


def _positive_integer(value: object, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0; got {result}")
    return result


def _ptychi_args(method: str, config: dict, args: argparse.Namespace) -> list[str]:
    recon = config.get("reconstruction", {})
    project = config.get("project", {})
    if not isinstance(recon, dict):
        raise ValueError("reconstruction must be a YAML mapping")
    if not isinstance(project, dict):
        raise ValueError("project must be a YAML mapping")
    per_method = config.get("methods", {}).get(method, {})
    if not isinstance(per_method, dict):
        raise ValueError(f"methods.{method} must be a YAML mapping")

    def param(key: str, fallback: object) -> object:
        # Optional CLI overrides take priority over method-specific YAML.
        cli = getattr(args, key, None)
        return cli if cli is not None else per_method.get(key, fallback)

    epochs = _positive_integer(param("epochs", recon.get("iterations", 200)), "epochs")
    # Do NOT blindly use Adam's batch size: historic Pty-Chi benchmark used 100.
    batch_size = _positive_integer(param("batch_size", 100), "batch_size")
    modes = _positive_integer(param("modes", recon.get("n_modes", 1)), "modes")
    seed = int(param("seed", project.get("seed", 0)))
    init_power = float(param("probe_mode_init_power", recon.get("probe_mode_init_power", 0.02)))
    c10 = float(param("probe_c10_a", recon.get("probe_C10_angstrom", 0.0)))
    result = [
        "--modes", str(modes),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--probe-mode-init-power", str(init_power),
        "--seed", str(seed),
        "--probe-c10-A", str(c10),
    ]
    if method == "epie":
        result += [
            "--object-alpha", str(float(param("object_alpha", 0.1))),
            "--probe-alpha", str(float(param("probe_alpha", 0.1))),
        ]
    elif method == "lsqml":
        result += [
            "--gaussian-noise-std", str(float(param("gaussian_noise_std", 0.5))),
            "--optimal-step-size-scaler", str(float(param("optimal_step_size_scaler", 0.9))),
        ]
    return result


def _ptychi_output(config: dict, config_path: Path, method: str, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return _resolve_path(args.output_dir, config_path)
    cfg = config.get("output", {})
    if not isinstance(cfg, dict):
        raise ValueError("output must be a YAML mapping")
    baseline = cfg.get("reconstruction_dir", "outputs/reconstruction")
    original = Path(baseline).expanduser()
    if not original.is_absolute():
        original = Path(__file__).resolve().parent / original
    original = original.resolve()
    return original.with_name(original.name + "_" + method)


def run_ptychi_method(
    method: str, config: dict, args: argparse.Namespace, config_path: Path
) -> dict:
    data = config.get("data", {})
    if not isinstance(data, dict) or not data.get("prepared_npz_path"):
        raise ValueError("Pty-Chi requires data.prepared_npz_path in the V8 YAML")

    prepared = _existing_path(data["prepared_npz_path"], config_path, "Prepared dataset")
    _validate_benchmark_npz(prepared)
    output_dir = _ptychi_output(config, config_path, method, args)
    runner = _runner_script(method, args, config, config_path)
    python = _ptychi_python(args, config, config_path)

    command = [
        str(python), str(runner),
        "--prepared", str(prepared),
        "--output", str(output_dir),
        *_ptychi_args(method, config, args),
    ]

    print(f"[V8] Method: {method.upper()} via Pty-Chi", flush=True)
    print(f"[V8] Python: {python}", flush=True)
    print(f"[V8] Runner: {runner}", flush=True)
    print(f"[V8] Dataset: {prepared}", flush=True)
    print(f"[V8] Output: {output_dir}", flush=True)

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"  # Avoid contamination by Windows user-site packages.
    # check=True propagates runner failure rather than falsely announcing success.
    subprocess.run(command, cwd=str(runner.parent), env=env, check=True)

    summary_path = output_dir / "run_summary.json"
    summary = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    return {
        "method": method,
        "prepared_path": str(prepared),
        "output_dir": str(output_dir),
        "run_summary": str(summary_path) if summary_path.exists() else None,
        "reconstruction_seconds": summary.get("elapsed_seconds"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ptychography Platform V0.8: Adam / ePIE / LSQML")
    parser.add_argument("command", choices=["prepare", "demo", "reconstruct"])
    parser.add_argument("--config", default="config_mos.yaml")
    parser.add_argument(
        "--method", choices=METHODS, default=None,
        help="Reconstruction method (default: Adam).",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Show a method selection menu; Enter chooses Adam.",
    )
    parser.add_argument("--ptychi-python", help="Python executable with installed Pty-Chi")
    parser.add_argument("--ptychi-runner-dir", help="Directory containing existing Pty-Chi runner scripts")
    parser.add_argument("--epie-runner", help="Explicit path to existing ePIE runner")
    parser.add_argument("--lsqml-runner", help="Explicit path to existing LSQML runner")
    parser.add_argument("--output-dir", help="Override output directory (ePIE / LSQML)")
    parser.add_argument("--epochs", type=int, help="Override full passes for ePIE / LSQML")
    parser.add_argument("--batch-size", type=int, help="Override batch size for ePIE / LSQML")
    parser.add_argument("--modes", type=int, help="Override probe modes for ePIE / LSQML")
    parser.add_argument("--seed", type=int, help="Override random seed for ePIE / LSQML")
    parser.add_argument("--probe-mode-init-power", type=float, help="ePIE / LSQML secondary-mode initial power")
    parser.add_argument("--probe-c10-a", type=float, help="ePIE / LSQML probe defocus in angstroms")
    parser.add_argument("--object-alpha", type=float, help="ePIE object update alpha")
    parser.add_argument("--probe-alpha", type=float, help="ePIE probe update alpha")
    parser.add_argument("--gaussian-noise-std", type=float, help="LSQML Gaussian noise std")
    parser.add_argument("--optimal-step-size-scaler", type=float, help="LSQML step scaler")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    if args.command == "prepare":
        from experimental_data import prepare_experimental_data

        result = prepare_experimental_data(config)
        print("V0.8 data preparation complete.")
        print(f"fitRBF: {result['fitRBF']:.6f} px")
        print(f"dx: {result['dx']:.9f} Å/object-pixel")
        print(f"scan step: {result['scan_step_px']:.9f} object pixels")
        print(f"object shape: {result['object_shape']}")
        print(f"mean total intensity: {result['mean_total_intensity']:.8f}")
        print(f"prepared dataset: {result['prepared_path']}")
        return

    if args.command == "demo":
        from reconstruction import run_synthetic_demo

        run_synthetic_demo(config)
        return

    method = _select_method(config, args.method, args.interactive)
    if method == "adam":
        from reconstruction import run_reconstruction

        print("[V8] Method: Adam (default)", flush=True)
        result = run_reconstruction(config)
    else:
        result = run_ptychi_method(method, config, args, config_path)

    print(f"V0.8 {method.upper()} reconstruction complete.")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(result)


if __name__ == "__main__":
    main()
