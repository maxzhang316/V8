"""Experimental-data preparation for Ptychography Platform V0.8.2.1.

Supported input formats
-----------------------
1. MATLAB .mat
   - Backward compatible with the previous MoS pipeline.
   - Supported layouts:
       det_y,det_x,scan_y,scan_x
       scan_y,scan_x,det_y,det_x

2. EMPAD .raw
   - Designed for the 1st-generation EMPAD raw layout used by the WSe2 data.
   - A stored frame may contain detector pixels plus metadata rows.
   - Example:
       scan grid      : 128 x 128
       stored frame   : 130 x 128 float32
       detector region: first 128 x 128 pixels
   - The raw file is memory-mapped so the full stored 1.09 GB file is not
     duplicated in RAM before the detector region is extracted.

The output contract remains the same as the previous platform:
    measured_intensity : [N_scan, det_y, det_x] float32
    scan_positions     : [N_scan, 2] float32 [y, x]
    crop_positions     : [N_scan, 2] int32/int64
    probe_pos_shifts   : [N_scan, 2] float32
    metadata_json      : JSON string

This keeps the reconstruction and forward-model interfaces unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# ============================================================================
# Electron / reciprocal-space calibration
# ============================================================================


def electron_wavelength_angstrom(voltage_kv: float) -> float:
    """Relativistic electron wavelength in Angstrom."""
    h = 6.62607015e-34
    m = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0

    voltage_v = float(voltage_kv) * 1000.0
    energy = e * voltage_v

    wavelength_m = h / math.sqrt(
        2.0
        * m
        * energy
        * (
            1.0
            + energy
            / (
                2.0
                * m
                * c
                * c
            )
        )
    )

    return wavelength_m * 1e10


def guess_radius_of_bright_field_disk(
    image: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Estimate BF-disk radius from thresholded area."""
    max_value = float(
        np.max(
            image
        )
    )

    if max_value <= 0:
        raise ValueError(
            "Cannot estimate bright-field radius from a non-positive image."
        )

    binary = (
        image
        > (
            max_value
            * float(
                threshold
            )
        )
    )

    area = int(
        np.sum(
            binary
        )
    )

    if area <= 0:
        raise ValueError(
            "Bright-field threshold produced an empty mask."
        )

    return float(
        np.sqrt(
            area
            / np.pi
        )
    )


def infer_dx_from_rbf(
    rbf_px: float,
    conv_angle_mrad: float,
    wavelength_angstrom: float,
    npix: int,
) -> float:
    """Infer object-plane pixel size using PtyRAD-style BF radius calibration."""
    da_rad = (
        float(
            conv_angle_mrad
        )
        / float(
            rbf_px
        )
        / 1e3
    )

    dk = (
        da_rad
        / float(
            wavelength_angstrom
        )
    )

    return float(
        1.0
        / (
            float(
                npix
            )
            * dk
        )
    )


def infer_dx_from_kmax(
    kmax_inv_angstrom: float,
) -> float:
    """Infer real-space sampling from detector kmax.

    If:
        kmax = (N / 2) * dk
        dx   = 1 / (N * dk)

    then:
        dx = 1 / (2 * kmax)
    """
    kmax = float(
        kmax_inv_angstrom
    )

    if kmax <= 0:
        raise ValueError(
            "kmax_inv_angstrom must be positive."
        )

    return float(
        1.0
        / (
            2.0
            * kmax
        )
    )


# ============================================================================
# Source-path / dtype helpers
# ============================================================================


def _resolve_source_path(
    cfg: dict[str, Any],
) -> Path:
    """Resolve new and legacy config keys."""
    for key in (
        "source_path",
        "raw_path",
        "raw_mat_path",
    ):
        value = cfg.get(
            key
        )

        if value:
            return Path(
                value
            )

    raise KeyError(
        "Data config must contain one of: "
        "source_path, raw_path, raw_mat_path."
    )


def _resolve_source_type(
    path: Path,
    cfg: dict[str, Any],
) -> str:
    requested = str(
        cfg.get(
            "source_type",
            "auto",
        )
    ).lower()

    if requested != "auto":
        return requested

    suffix = path.suffix.lower()

    if suffix == ".mat":
        return "mat"

    if suffix == ".raw":
        return "empad_raw"

    raise ValueError(
        f"Cannot infer source_type from extension {suffix!r}. "
        "Set data.source_type explicitly."
    )


def _numpy_dtype(
    name: str,
    endianness: str,
) -> np.dtype:
    dtype = np.dtype(
        name
    )

    endian = str(
        endianness
    ).lower()

    if dtype.itemsize == 1:
        return dtype

    if endian in (
        "little",
        "<",
    ):
        return dtype.newbyteorder(
            "<"
        )

    if endian in (
        "big",
        ">",
    ):
        return dtype.newbyteorder(
            ">"
        )

    if endian in (
        "native",
        "=",
    ):
        return dtype.newbyteorder(
            "="
        )

    raise ValueError(
        "raw_endianness must be little, big, or native."
    )


# ============================================================================
# MATLAB loading
# ============================================================================


def _load_mat_source(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[
    np.ndarray,
    tuple[int, int],
    dict[str, Any],
]:
    contents = loadmat(
        path
    )

    key = str(
        cfg[
            "intensity_key"
        ]
    )

    if key not in contents:
        public_keys = sorted(
            key_name
            for key_name
            in contents
            if not key_name.startswith(
                "__"
            )
        )

        raise KeyError(
            f"MAT variable {key!r} was not found. "
            f"Available variables: {public_keys}"
        )

    raw = np.asarray(
        contents[
            key
        ],
        dtype=np.float32,
    )

    if raw.ndim != 4:
        raise ValueError(
            f"Expected 4D MATLAB diffraction data, got {raw.shape}."
        )

    layout = str(
        cfg.get(
            "intensity_layout",
            "det_y,det_x,scan_y,scan_x",
        )
    ).replace(
        " ",
        "",
    ).lower()

    if layout == "det_y,det_x,scan_y,scan_x":
        det_y, det_x, scan_y, scan_x = raw.shape

        measurement = np.transpose(
            raw,
            (
                2,
                3,
                0,
                1,
            ),
        ).reshape(
            scan_y
            * scan_x,
            det_y,
            det_x,
        )

    elif layout == "scan_y,scan_x,det_y,det_x":
        scan_y, scan_x, det_y, det_x = raw.shape

        measurement = raw.reshape(
            scan_y
            * scan_x,
            det_y,
            det_x,
        )

    else:
        raise ValueError(
            "Unsupported intensity_layout for .mat input: "
            f"{layout!r}."
        )

    measurement = np.ascontiguousarray(
        measurement,
        dtype=np.float32,
    )

    metadata = {
        "source_type": "mat",
        "source_file_shape": list(
            raw.shape
        ),
        "stored_frame_shape": [
            int(
                det_y
            ),
            int(
                det_x
            ),
        ],
        "detector_shape": [
            int(
                det_y
            ),
            int(
                det_x
            ),
        ],
        "mat_intensity_key": key,
        "intensity_layout": layout,
    }

    return (
        measurement,
        (
            int(
                scan_y
            ),
            int(
                scan_x
            ),
        ),
        metadata,
    )


# ============================================================================
# EMPAD raw loading
# ============================================================================


def _load_empad_raw_source(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[
    np.ndarray,
    tuple[int, int],
    dict[str, Any],
]:
    """Load a headerless EMPAD raw stream.

    The loader validates the exact file size before reading detector data.

    Config fields:
        pos_N_scan_slow
        pos_N_scan_fast

        raw_dtype
        raw_endianness
        raw_header_bytes

        raw_frame_shape: [stored_rows, stored_cols]
        detector_shape:  [detector_rows, detector_cols]

        detector_row_start
        detector_col_start

        raw_scan_axis_order:
            slow_fast   (default)
            fast_slow

    For the WSe2 EMPAD dataset used here:
        scan grid       = 128 x 128
        raw_frame_shape = 130 x 128
        detector_shape  = 128 x 128
        dtype           = float32
        header          = 0 bytes
    """
    n_slow = int(
        cfg[
            "pos_N_scan_slow"
        ]
    )

    n_fast = int(
        cfg[
            "pos_N_scan_fast"
        ]
    )

    frame_shape = tuple(
        int(
            value
        )
        for value
        in cfg.get(
            "raw_frame_shape",
            [
                130,
                128,
            ],
        )
    )

    if len(
        frame_shape
    ) != 2:
        raise ValueError(
            "raw_frame_shape must contain [rows, cols]."
        )

    detector_shape = tuple(
        int(
            value
        )
        for value
        in cfg.get(
            "detector_shape",
            [
                128,
                128,
            ],
        )
    )

    if len(
        detector_shape
    ) != 2:
        raise ValueError(
            "data.detector_shape must contain [rows, cols]."
        )

    frame_rows, frame_cols = (
        frame_shape
    )

    detector_rows, detector_cols = (
        detector_shape
    )

    row_start = int(
        cfg.get(
            "detector_row_start",
            0,
        )
    )

    col_start = int(
        cfg.get(
            "detector_col_start",
            0,
        )
    )

    row_end = (
        row_start
        + detector_rows
    )

    col_end = (
        col_start
        + detector_cols
    )

    if (
        row_start < 0
        or col_start < 0
        or row_end > frame_rows
        or col_end > frame_cols
    ):
        raise ValueError(
            "Configured detector region does not fit inside raw_frame_shape."
        )

    dtype = _numpy_dtype(
        str(
            cfg.get(
                "raw_dtype",
                "float32",
            )
        ),
        str(
            cfg.get(
                "raw_endianness",
                "little",
            )
        ),
    )

    header_bytes = int(
        cfg.get(
            "raw_header_bytes",
            0,
        )
    )

    expected_elements = (
        n_slow
        * n_fast
        * frame_rows
        * frame_cols
    )

    expected_payload_bytes = (
        expected_elements
        * dtype.itemsize
    )

    expected_file_bytes = (
        header_bytes
        + expected_payload_bytes
    )

    actual_file_bytes = int(
        path.stat().st_size
    )

    allow_trailing = bool(
        cfg.get(
            "allow_trailing_bytes",
            False,
        )
    )

    if allow_trailing:
        if actual_file_bytes < expected_file_bytes:
            raise ValueError(
                "EMPAD raw file is smaller than expected. "
                f"Expected at least {expected_file_bytes:,} bytes, "
                f"got {actual_file_bytes:,}."
            )
    elif actual_file_bytes != expected_file_bytes:
        raise ValueError(
            "EMPAD raw file size does not match the configured geometry.\n"
            f"  actual bytes   : {actual_file_bytes:,}\n"
            f"  expected bytes : {expected_file_bytes:,}\n"
            f"  scan grid      : {n_slow} x {n_fast}\n"
            f"  stored frame   : {frame_rows} x {frame_cols}\n"
            f"  dtype          : {dtype} ({dtype.itemsize} bytes/value)\n"
            "Check raw_dtype, raw_frame_shape, scan dimensions, or header size."
        )

    scan_order = str(
        cfg.get(
            "raw_scan_axis_order",
            "slow_fast",
        )
    ).lower()

    if scan_order == "slow_fast":
        stored_shape = (
            n_slow,
            n_fast,
            frame_rows,
            frame_cols,
        )

    elif scan_order == "fast_slow":
        stored_shape = (
            n_fast,
            n_slow,
            frame_rows,
            frame_cols,
        )

    else:
        raise ValueError(
            "raw_scan_axis_order must be slow_fast or fast_slow."
        )

    mapped = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header_bytes,
        shape=stored_shape,
        order="C",
    )

    detector_view = mapped[
        ...,
        row_start:row_end,
        col_start:col_end,
    ]

    if scan_order == "fast_slow":
        detector_view = np.transpose(
            detector_view,
            (
                1,
                0,
                2,
                3,
            ),
        )

    # One intentional allocation: create the canonical float32 detector data.
    measurement = np.array(
        detector_view,
        dtype=np.float32,
        copy=True,
        order="C",
    ).reshape(
        n_slow
        * n_fast,
        detector_rows,
        detector_cols,
    )

    metadata_rows = (
        frame_rows
        - detector_rows
    )

    metadata_cols = (
        frame_cols
        - detector_cols
    )

    metadata = {
        "source_type": "empad_raw",
        "source_file_shape": [
            int(
                n_slow
            ),
            int(
                n_fast
            ),
            int(
                frame_rows
            ),
            int(
                frame_cols
            ),
        ],
        "stored_frame_shape": [
            int(
                frame_rows
            ),
            int(
                frame_cols
            ),
        ],
        "detector_shape": [
            int(
                detector_rows
            ),
            int(
                detector_cols
            ),
        ],
        "raw_dtype": str(
            dtype
        ),
        "raw_endianness": str(
            cfg.get(
                "raw_endianness",
                "little",
            )
        ),
        "raw_header_bytes": int(
            header_bytes
        ),
        "raw_scan_axis_order": scan_order,
        "detector_region": {
            "row_start": int(
                row_start
            ),
            "row_end": int(
                row_end
            ),
            "col_start": int(
                col_start
            ),
            "col_end": int(
                col_end
            ),
        },
        "ignored_rows_per_frame": int(
            metadata_rows
        ),
        "ignored_cols_per_frame": int(
            metadata_cols
        ),
        "source_file_bytes": actual_file_bytes,
        "expected_file_bytes": expected_file_bytes,
    }

    return (
        measurement,
        (
            n_slow,
            n_fast,
        ),
        metadata,
    )


# ============================================================================
# Unified source loader
# ============================================================================


def _load_experimental_source(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[
    np.ndarray,
    tuple[int, int],
    dict[str, Any],
]:
    source_type = _resolve_source_type(
        path,
        cfg,
    )

    if source_type == "mat":
        return _load_mat_source(
            path,
            cfg,
        )

    if source_type in (
        "raw",
        "empad_raw",
        "empad",
    ):
        return _load_empad_raw_source(
            path,
            cfg,
        )

    raise ValueError(
        f"Unsupported source_type: {source_type!r}."
    )


# ============================================================================
# Geometry / preprocessing
# ============================================================================


def _apply_flipT(
    measurement: np.ndarray,
    flipT: list[int] | None,
) -> np.ndarray:
    if flipT is None:
        return measurement

    if len(
        flipT
    ) != 3:
        raise ValueError(
            "meas_flipT must contain [flipud, fliplr, transpose]."
        )

    out = measurement

    if int(
        flipT[
            0
        ]
    ):
        out = np.flip(
            out,
            axis=1,
        )

    if int(
        flipT[
            1
        ]
    ):
        out = np.flip(
            out,
            axis=2,
        )

    if int(
        flipT[
            2
        ]
    ):
        out = np.transpose(
            out,
            (
                0,
                2,
                1,
            ),
        )

    return np.ascontiguousarray(
        out,
        dtype=np.float32,
    )


def _remove_negative_inplace(
    measurement: np.ndarray,
    mode: str,
) -> np.ndarray:
    mode = str(
        mode
    )

    if mode == "clip_neg":
        np.maximum(
            measurement,
            np.float32(
                0.0
            ),
            out=measurement,
        )

        return measurement

    if mode == "subtract_min":
        minimum = float(
            measurement.min()
        )

        measurement -= np.float32(
            minimum
        )

        return measurement

    raise ValueError(
        f"Unsupported negative mode: {mode}"
    )


def _normalize_measurement_inplace(
    measurement: np.ndarray,
    mode: str,
) -> tuple[
    np.ndarray,
    float,
]:
    if mode == "max_at_one":
        constant = float(
            np.mean(
                measurement,
                axis=0,
                dtype=np.float32,
            ).max()
        )

    elif mode == "mean_at_one":
        constant = float(
            np.mean(
                measurement,
                axis=0,
                dtype=np.float32,
            ).mean()
        )

    elif mode == "sum_to_one":
        constant = float(
            np.mean(
                measurement,
                axis=0,
                dtype=np.float32,
            ).sum()
        )

    else:
        raise ValueError(
            f"Unsupported normalization mode: {mode}"
        )

    if constant <= 0:
        raise ValueError(
            "Normalization constant must be positive."
        )

    measurement /= np.float32(
        constant
    )

    return (
        measurement,
        constant,
    )


# ============================================================================
# Scan positions
# ============================================================================


def _ptyrad_positions(
    dx_angstrom: float,
    scan_step_angstrom: float,
    n_slow: int,
    n_fast: int,
    probe_shape: tuple[int, int],
    random_std_px: float,
    seed: int,
    object_shape_override: tuple[int, int] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """Generate centered regular raster positions.

    Optional deterministic Gaussian jitter is retained for compatibility, but
    V8.2.1 normally uses random_std_px = 0.
    """
    step_px = (
        float(
            scan_step_angstrom
        )
        / float(
            dx_angstrom
        )
    )

    positions = (
        step_px
        * np.asarray(
            [
                (
                    y,
                    x,
                )
                for y
                in range(
                    n_slow
                )
                for x
                in range(
                    n_fast
                )
            ],
            dtype=np.float64,
        )
    )

    positions -= positions.mean(
        axis=0
    )

    probe_shape_arr = np.asarray(
        probe_shape,
        dtype=np.float64,
    )

    initial_obj_shape = (
        1.2
        * np.ceil(
            positions.max(
                axis=0
            )
            - positions.min(
                axis=0
            )
            + probe_shape_arr
        )
    )

    positions += np.ceil(
        initial_obj_shape
        / 2.0
        - probe_shape_arr
        / 2.0
    )

    if random_std_px > 0:
        rng = np.random.RandomState(
            int(
                seed
            )
        )

        positions = (
            positions
            + float(
                random_std_px
            )
            * rng.randn(
                *positions.shape
            )
        )

    crop_positions = np.round(
        positions
    ).astype(
        np.int32
    )

    probe_pos_shifts = (
        positions
        - crop_positions
    ).astype(
        np.float32
    )

    if object_shape_override is None:
        extent = (
            1.2
            * np.ceil(
                positions.max(
                    axis=0
                )
                - positions.min(
                    axis=0
                )
                + probe_shape_arr
            )
        ).astype(
            int
        )

        object_shape = (
            int(
                extent[
                    0
                ]
            ),
            int(
                extent[
                    1
                ]
            ),
        )

    else:
        object_shape = (
            int(
                object_shape_override[
                    0
                ]
            ),
            int(
                object_shape_override[
                    1
                ]
            ),
        )

    h_probe, w_probe = (
        probe_shape
    )

    if (
        crop_positions[
            :,
            0,
        ].min()
        < 0
        or crop_positions[
            :,
            1,
        ].min()
        < 0
        or (
            crop_positions[
                :,
                0,
            ]
            + h_probe
        ).max()
        > object_shape[
            0
        ]
        or (
            crop_positions[
                :,
                1,
            ]
            + w_probe
        ).max()
        > object_shape[
            1
        ]
    ):
        raise ValueError(
            "Generated crop positions do not fit the configured object shape. "
            "Remove object_shape_override or increase it."
        )

    return (
        positions.astype(
            np.float32
        ),
        crop_positions,
        probe_pos_shifts,
        object_shape,
    )


# ============================================================================
# Diagnostics
# ============================================================================


def _save_diagnostics(
    raw_mean: np.ndarray,
    processed_mean: np.ndarray,
    positions: np.ndarray,
    crop_positions: np.ndarray,
    probe_pos_shifts: np.ndarray,
    out_dir: Path,
    source_metadata: dict[str, Any],
) -> list[str]:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated: list[
        str
    ] = []

    for image, filename, title in [
        (
            raw_mean,
            "mean_diffraction_geometry_only.png",
            "Mean diffraction after geometry operations",
        ),
        (
            processed_mean,
            "mean_diffraction_prepared.png",
            "Prepared mean diffraction (log1p)",
        ),
    ]:
        fig, ax = plt.subplots(
            figsize=(
                7,
                6,
            )
        )

        if "prepared" in filename:
            display = np.log1p(
                np.clip(
                    image,
                    0,
                    None,
                )
            )
        else:
            display = image

        im = ax.imshow(
            display
        )

        ax.set_title(
            title
        )

        fig.colorbar(
            im,
            ax=ax,
        )

        fig.tight_layout()

        path = (
            out_dir
            / filename
        )

        fig.savefig(
            path,
            dpi=170,
        )

        plt.close(
            fig
        )

        generated.append(
            str(
                path
            )
        )

    fig, ax = plt.subplots(
        figsize=(
            7,
            7,
        )
    )

    ax.scatter(
        positions[
            :,
            1,
        ],
        positions[
            :,
            0,
        ],
        s=3,
    )

    ax.invert_yaxis()

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        "Initial continuous scan positions"
    )

    ax.set_xlabel(
        "x (object pixel)"
    )

    ax.set_ylabel(
        "y (object pixel)"
    )

    fig.tight_layout()

    path = (
        out_dir
        / "initial_scan_positions.png"
    )

    fig.savefig(
        path,
        dpi=170,
    )

    plt.close(
        fig
    )

    generated.append(
        str(
            path
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    ax.hist(
        probe_pos_shifts[
            :,
            0,
        ],
        bins=50,
        alpha=0.6,
        label="y",
    )

    ax.hist(
        probe_pos_shifts[
            :,
            1,
        ],
        bins=50,
        alpha=0.6,
        label="x",
    )

    ax.set_title(
        "Initial fractional probe-position shifts"
    )

    ax.set_xlabel(
        "pixels"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        out_dir
        / "initial_probe_position_shift_hist.png"
    )

    fig.savefig(
        path,
        dpi=170,
    )

    plt.close(
        fig
    )

    generated.append(
        str(
            path
        )
    )

    return generated


# ============================================================================
# Main preparation entry point
# ============================================================================


def prepare_experimental_data(
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config[
        "data"
    ]

    project_cfg = config[
        "project"
    ]

    raw_path = _resolve_source_path(
        cfg
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            raw_path
        )

    (
        measurement,
        scan_grid,
        source_metadata,
    ) = _load_experimental_source(
        raw_path,
        cfg,
    )

    configured_scan_grid = (
        int(
            cfg[
                "pos_N_scan_slow"
            ]
        ),
        int(
            cfg[
                "pos_N_scan_fast"
            ]
        ),
    )

    if scan_grid != configured_scan_grid:
        raise ValueError(
            "Loaded scan grid does not match config. "
            f"Loaded={scan_grid}, configured={configured_scan_grid}."
        )

    measurement = _apply_flipT(
        measurement,
        cfg.get(
            "meas_flipT"
        ),
    )

    # Geometry-only mean is retained before negative clipping/normalization.
    raw_mean = measurement.mean(
        axis=0,
        dtype=np.float32,
    )

    fit_rbf = guess_radius_of_bright_field_disk(
        raw_mean,
        threshold=float(
            cfg.get(
                "fitRBF_threshold",
                0.5,
            )
        ),
    )

    wavelength = electron_wavelength_angstrom(
        float(
            cfg[
                "probe_kv"
            ]
        )
    )

    dx_fit_rbf = infer_dx_from_rbf(
        fit_rbf,
        float(
            cfg[
                "probe_conv_angle_mrad"
            ]
        ),
        wavelength,
        int(
            measurement.shape[
                -1
            ]
        ),
    )

    calibration_mode = str(
        cfg.get(
            "calibration_mode",
            "fitRBF",
        )
    ).lower()

    kmax_value: float | None = None

    if calibration_mode in (
        "fitrbf",
        "fit_rbf",
    ):
        dx = float(
            dx_fit_rbf
        )

    elif calibration_mode in (
        "direct_dx",
        "dx",
    ):
        dx = float(
            cfg[
                "dx_angstrom_per_object_pixel"
            ]
        )

    elif calibration_mode in (
        "direct_kmax",
        "kmax",
    ):
        kmax_value = float(
            cfg[
                "kmax_inv_angstrom"
            ]
        )

        dx = infer_dx_from_kmax(
            kmax_value
        )

    else:
        raise ValueError(
            "calibration_mode must be fitRBF, direct_dx, or direct_kmax."
        )

    if dx <= 0:
        raise ValueError(
            "Chosen dx must be positive."
        )

    object_override = cfg.get(
        "object_shape_override"
    )

    object_override_tuple = (
        tuple(
            int(
                value
            )
            for value
            in object_override
        )
        if object_override is not None
        else None
    )

    (
        positions,
        crop_positions,
        initial_shifts,
        object_shape,
    ) = _ptyrad_positions(
        dx_angstrom=dx,
        scan_step_angstrom=float(
            cfg[
                "pos_scan_step_size_angstrom"
            ]
        ),
        n_slow=int(
            cfg[
                "pos_N_scan_slow"
            ]
        ),
        n_fast=int(
            cfg[
                "pos_N_scan_fast"
            ]
        ),
        probe_shape=(
            int(
                measurement.shape[
                    -2
                ]
            ),
            int(
                measurement.shape[
                    -1
                ]
            ),
        ),
        random_std_px=float(
            cfg.get(
                "pos_scan_rand_std_px",
                0.0,
            )
        ),
        seed=int(
            project_cfg.get(
                "seed",
                0,
            )
        ),
        object_shape_override=object_override_tuple,
    )

    negative_before = int(
        np.count_nonzero(
            measurement
            < 0
        )
    )

    min_before = float(
        measurement.min()
    )

    max_before = float(
        measurement.max()
    )

    _remove_negative_inplace(
        measurement,
        str(
            cfg.get(
                "remove_negative_mode",
                "clip_neg",
            )
        ),
    )

    (
        measurement,
        norm_const,
    ) = _normalize_measurement_inplace(
        measurement,
        str(
            cfg.get(
                "normalization",
                "max_at_one",
            )
        ),
    )

    processed_mean = measurement.mean(
        axis=0,
        dtype=np.float32,
    )

    mean_total_intensity = float(
        measurement.sum(
            axis=(
                1,
                2,
            ),
            dtype=np.float32,
        ).mean()
    )

    scan_step_px = float(
        cfg[
            "pos_scan_step_size_angstrom"
        ]
        / dx
    )

    metadata = {
        "version": "0.8.2.1",
        "source_path": str(
            raw_path
        ),
        **source_metadata,
        "canonical_shape": list(
            measurement.shape
        ),
        "scan_grid": list(
            scan_grid
        ),
        "meas_flipT": cfg.get(
            "meas_flipT"
        ),
        "fitRBF_px": float(
            fit_rbf
        ),
        "fitRBF_dx_angstrom_per_object_pixel": float(
            dx_fit_rbf
        ),
        "calibration_mode": calibration_mode,
        "kmax_inv_angstrom": (
            float(
                kmax_value
            )
            if kmax_value is not None
            else None
        ),
        "electron_wavelength_angstrom": float(
            wavelength
        ),
        "dx_angstrom_per_object_pixel": float(
            dx
        ),
        "probe_kv": float(
            cfg[
                "probe_kv"
            ]
        ),
        "probe_conv_angle_mrad": float(
            cfg[
                "probe_conv_angle_mrad"
            ]
        ),
        "pos_scan_step_size_angstrom": float(
            cfg[
                "pos_scan_step_size_angstrom"
            ]
        ),
        "scan_step_object_pixels": scan_step_px,
        "pos_scan_rand_std_px": float(
            cfg.get(
                "pos_scan_rand_std_px",
                0.0,
            )
        ),
        "recommended_object_shape": list(
            object_shape
        ),
        "normalization_mode": str(
            cfg.get(
                "normalization",
                "max_at_one",
            )
        ),
        "normalization_constant": float(
            norm_const
        ),
        "mean_total_intensity": mean_total_intensity,
        "raw_min_before_processing": min_before,
        "raw_max_before_processing": max_before,
        "negative_values_before_processing": negative_before,
        "crop_position_min_yx": crop_positions.min(
            axis=0
        ).tolist(),
        "crop_position_max_yx": crop_positions.max(
            axis=0
        ).tolist(),
        "initial_probe_shift_std_yx": initial_shifts.std(
            axis=0
        ).tolist(),
    }

    prepared_path = Path(
        cfg[
            "prepared_npz_path"
        ]
    )

    prepared_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    compress = bool(
        cfg.get(
            "compress_prepared",
            False,
        )
    )

    save_fn = (
        np.savez_compressed
        if compress
        else np.savez
    )

    save_fn(
        prepared_path,
        measured_intensity=measurement,
        scan_positions=positions,
        crop_positions=crop_positions,
        probe_pos_shifts=initial_shifts,
        metadata_json=json.dumps(
            metadata
        ),
    )

    out_dir = Path(
        cfg[
            "inspection_output_dir"
        ]
    )

    images = _save_diagnostics(
        raw_mean,
        processed_mean,
        positions,
        crop_positions,
        initial_shifts,
        out_dir,
        source_metadata,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out_dir
        / "dataset_report.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "Ptychography Platform V0.8.2.1 - dataset report",
        "="
        * 76,
        f"source: {raw_path}",
        f"source type: {source_metadata['source_type']}",
        f"source file shape: {source_metadata['source_file_shape']}",
        f"canonical measurement shape: {measurement.shape}",
        f"scan grid: {scan_grid}",
        f"detector shape: {measurement.shape[-2:]}",
        f"fitRBF: {fit_rbf:.6f} px",
        f"fitRBF-derived dx: {dx_fit_rbf:.9f} Ang/object-pixel",
        f"calibration mode: {calibration_mode}",
        f"chosen dx: {dx:.9f} Ang/object-pixel",
        f"scan step: {scan_step_px:.9f} object pixels",
        f"random position std: {metadata['pos_scan_rand_std_px']:.6f} px",
        f"object shape: {object_shape}",
        (
            "crop min/max: "
            f"{metadata['crop_position_min_yx']} / "
            f"{metadata['crop_position_max_yx']}"
        ),
        (
            "initial shift std: "
            f"{metadata['initial_probe_shift_std_yx']}"
        ),
        (
            "raw min/max before processing: "
            f"{min_before:.8g} / {max_before:.8g}"
        ),
        (
            "negative values before processing: "
            f"{negative_before}"
        ),
        (
            "mean total prepared intensity: "
            f"{mean_total_intensity:.8f}"
        ),
        f"prepared file: {prepared_path}",
        f"prepared compression: {compress}",
    ]

    (
        out_dir
        / "dataset_report.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("EXPERIMENTAL DATA PREPARATION")
    print("=" * 76)

    for line in lines[
        2:
    ]:
        print(
            line
        )

    print("=" * 76)

    return {
        "prepared_path": str(
            prepared_path
        ),
        "fitRBF": float(
            fit_rbf
        ),
        "fitRBF_dx": float(
            dx_fit_rbf
        ),
        "calibration_mode": calibration_mode,
        "dx": float(
            dx
        ),
        "scan_step_px": scan_step_px,
        "object_shape": object_shape,
        "mean_total_intensity": mean_total_intensity,
        "diagnostic_images": images,
    }


# ============================================================================
# Prepared dataset loader
# ============================================================================


def load_prepared_dataset(
    path: str | Path,
):
    with np.load(
        Path(
            path
        )
    ) as data:
        measurement = np.asarray(
            data[
                "measured_intensity"
            ],
            dtype=np.float32,
        )

        positions = np.asarray(
            data[
                "scan_positions"
            ],
            dtype=np.float32,
        )

        crop_positions = np.asarray(
            data[
                "crop_positions"
            ],
            dtype=np.int64,
        )

        initial_shifts = np.asarray(
            data[
                "probe_pos_shifts"
            ],
            dtype=np.float32,
        )

        metadata_raw = data[
            "metadata_json"
        ].item()

    metadata = json.loads(
        str(
            metadata_raw
        )
    )

    return (
        measurement,
        positions,
        crop_positions,
        initial_shifts,
        metadata,
    )