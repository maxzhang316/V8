"""Output manager for Ptychography Platform V0.8.2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def mode_powers(probe: torch.Tensor) -> np.ndarray:
    return (
        torch.sum(torch.abs(probe).square(), dim=(-2, -1))
        .real.detach().cpu().numpy()
    )


def mode_fractions(probe: torch.Tensor) -> np.ndarray:
    powers = mode_powers(probe)
    return powers / max(float(powers.sum()), 1e-30)


class OutputManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root = Path(config["output"]["reconstruction_dir"])
        self.object_dir = self.root / "object"
        self.probe_dir = self.root / "probe"
        self.loss_dir = self.root / "loss"
        self.forward_dir = self.root / "forward"
        self.position_dir = self.root / "positions"
        self.checkpoint_dir = self.root / "checkpoints"
        self.final_dir = self.root / "final"
        for directory in [
            self.root,
            self.object_dir,
            self.probe_dir,
            self.loss_dir,
            self.forward_dir,
            self.position_dir,
            self.checkpoint_dir,
            self.final_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.history_path = self.root / "history.csv"
        if self.history_path.exists():
            self.history_path.unlink()
        self._history_fieldnames: list[str] | None = None

    def append_history(self, row: dict[str, Any]) -> None:
        if self._history_fieldnames is None:
            self._history_fieldnames = list(row.keys())
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._history_fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def save_state_figures(
        self,
        iteration: int,
        obja: torch.Tensor,
        objp: torch.Tensor,
        probe: torch.Tensor,
        crop_positions: torch.Tensor,
        probe_pos_shifts: torch.Tensor,
        history: list[dict[str, Any]],
    ) -> None:
        amp = obja.detach().cpu().numpy()
        phase = objp.detach().cpu().numpy()
        probe_np = probe.detach().cpu().numpy()
        fractions = mode_fractions(probe)
        continuous = (
            crop_positions.to(torch.float32) + probe_pos_shifts
        ).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, image, title in [
            (axes[0], amp, "Object amplitude"),
            (axes[1], phase, "Object phase"),
        ]:
            im = ax.imshow(image)
            ax.set_title(f"{title} - iter {iteration:04d}")
            ax.set_axis_off()
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.object_dir / f"object_iter{iteration:04d}.png", dpi=180)
        plt.close(fig)

        n_modes = probe_np.shape[0]
        fig, axes = plt.subplots(2, n_modes, figsize=(4 * n_modes, 8), squeeze=False)
        for i in range(n_modes):
            im = axes[0, i].imshow(np.abs(probe_np[i]))
            axes[0, i].set_title(f"Mode {i} amplitude\n{fractions[i]*100:.2f}%")
            axes[0, i].set_axis_off()
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            reciprocal = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(probe_np[i]))
            )
            im = axes[1, i].imshow(np.log1p(np.abs(reciprocal)))
            axes[1, i].set_title(f"Mode {i} reciprocal")
            axes[1, i].set_axis_off()
            fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        fig.suptitle(f"V0.8.2 probe modes - iter {iteration:04d}")
        fig.tight_layout()
        fig.savefig(self.probe_dir / f"probe_modes_iter{iteration:04d}.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        iters = [row["iteration"] for row in history]
        axes[0].plot(iters, [row["data_loss"] for row in history], label="data loss")
        axes[0].plot(
            iters,
            [row["selection_score"] for row in history],
            label="quality score",
        )
        axes[0].set_xlabel("Iteration")
        axes[0].set_title("Data loss vs quality score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            iters,
            [row["probe_anchor_nrmse"] for row in history],
            label="probe intensity drift",
        )
        axes[1].plot(
            iters,
            [row["probe_support_leakage"] for row in history],
            label="k-support leakage",
        )
        axes[1].set_xlabel("Iteration")
        axes[1].set_title("Probe stability")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(
            iters,
            [row["position_rms_drift_px"] for row in history],
            label="position RMS drift",
        )
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("pixels")
        axes[2].set_title("Position drift")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(self.loss_dir / f"diagnostics_iter{iteration:04d}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(continuous[:, 1], continuous[:, 0], s=4)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Scan positions - iter {iteration:04d}")
        fig.tight_layout()
        fig.savefig(self.position_dir / f"positions_iter{iteration:04d}.png", dpi=180)
        plt.close(fig)

    def save_forward_summary(
        self,
        iteration: int,
        measured: torch.Tensor,
        predicted: torch.Tensor,
    ) -> None:
        measured_np = measured.detach().cpu().numpy()
        predicted_np = predicted.detach().cpu().numpy()
        count = min(
            int(self.config["output"].get("forward_summary_scans", 3)),
            measured_np.shape[0],
        )
        selected = np.linspace(0, measured_np.shape[0] - 1, count, dtype=int)
        fig, axes = plt.subplots(count, 3, figsize=(13, 4 * count), squeeze=False)
        for row, idx in enumerate(selected):
            residual = np.abs(measured_np[idx] - predicted_np[idx])
            for col, (image, title) in enumerate(
                [
                    (measured_np[idx], "Measured"),
                    (predicted_np[idx], "Predicted"),
                    (residual, "|Residual|"),
                ]
            ):
                ax = axes[row, col]
                im = ax.imshow(np.log1p(np.clip(image, 0, None)))
                ax.set_title(title)
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Forward comparison - iter {iteration:04d}")
        fig.tight_layout()
        fig.savefig(self.forward_dir / f"forward_iter{iteration:04d}.png", dpi=180)
        plt.close(fig)

    def save_checkpoint(
        self,
        iteration: int,
        state: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        crop_positions: torch.Tensor,
        history: list[dict[str, Any]],
    ) -> None:
        torch.save(
            {
                "iteration": iteration,
                "state_dict": state.state_dict(),
                "crop_positions": crop_positions.detach().cpu(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            self.checkpoint_dir / f"checkpoint_iter{iteration:04d}.pt",
        )

    def save_hdf5(
        self,
        iteration: int,
        *,
        obja: torch.Tensor,
        objp: torch.Tensor,
        probe: torch.Tensor,
        crop_positions: torch.Tensor,
        probe_pos_shifts: torch.Tensor,
        metadata: dict[str, Any],
        loss: float,
        intensity_loss: float,
        selection_score_value: float,
    ) -> None:
        path = self.checkpoint_dir / f"model_iter{iteration:04d}.hdf5"
        with h5py.File(path, "w") as h5:
            optim = h5.create_group("optimizable_tensors")
            optim.create_dataset(
                "obja",
                data=obja.detach().cpu().numpy()[None, None].astype(np.float32),
            )
            optim.create_dataset(
                "objp",
                data=objp.detach().cpu().numpy()[None, None].astype(np.float32),
            )
            optim.create_dataset(
                "probe", data=probe.detach().cpu().numpy().astype(np.complex64)
            )
            optim.create_dataset(
                "probe_pos_shifts",
                data=probe_pos_shifts.detach().cpu().numpy().astype(np.float32),
            )

            attrs = h5.create_group("model_attributes")
            attrs.create_dataset(
                "crop_pos", data=crop_positions.detach().cpu().numpy().astype(np.int32)
            )
            attrs.create_dataset(
                "dx", data=np.float32(metadata["dx_angstrom_per_object_pixel"])
            )
            attrs.create_dataset(
                "probe_int_sum",
                data=np.float32(
                    torch.sum(torch.abs(probe).square()).real.detach().cpu()
                ),
            )

            losses = h5.create_group("avg_losses")
            losses.create_dataset("data_loss", data=np.float32(loss))
            # Compatibility metric: normalized intensity RMSE (same form as
            # the earlier dp_pow=1 loss_single, although it is no longer the
            # primary V0.8.2 optimization objective).
            losses.create_dataset("loss_single", data=np.float32(intensity_loss))
            losses.create_dataset(
                "selection_score", data=np.float32(selection_score_value)
            )
            h5.create_dataset("niter", data=np.int64(iteration))
            h5.create_dataset("platform_version", data=np.bytes_("0.8.1"))

    def _snapshot_to_numpy(self, snapshot: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        return {
            "obja": snapshot["obja"].detach().cpu().numpy(),
            "objp": snapshot["objp"].detach().cpu().numpy(),
            "probe": snapshot["probe"].detach().cpu().numpy(),
            "probe_pos_shifts": snapshot["probe_pos_shifts"].detach().cpu().numpy(),
            "crop_positions": snapshot["crop_positions"].detach().cpu().numpy(),
        }

    def save_final_summary(
        self,
        *,
        obja: torch.Tensor,
        objp: torch.Tensor,
        probe: torch.Tensor,
        crop_positions: torch.Tensor,
        probe_pos_shifts: torch.Tensor,
        history: list[dict[str, Any]],
        metadata: dict[str, Any],
        selected_iteration: int,
        selected_score: float,
        last_iteration: int,
        last_snapshot: dict[str, torch.Tensor],
    ) -> None:
        amp = obja.detach().cpu().numpy()
        phase = objp.detach().cpu().numpy()
        probe_np = probe.detach().cpu().numpy()
        continuous = (
            crop_positions.to(torch.float32) + probe_pos_shifts
        ).detach().cpu().numpy()
        fractions = mode_fractions(probe)

        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        for ax, image, title in [
            (axes[0, 0], amp, "Selected object amplitude"),
            (axes[0, 1], phase, "Selected object phase"),
            (axes[0, 2], np.abs(probe_np[0]), "Selected probe mode 0"),
        ]:
            im = ax.imshow(image)
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax)

        iters = [r["iteration"] for r in history]
        axes[1, 0].plot(iters, [r["data_loss"] for r in history], label="data loss")
        axes[1, 0].plot(
            iters,
            [r["selection_score"] for r in history],
            label="quality score",
        )
        axes[1, 0].axvline(selected_iteration, linestyle="--", label="selected")
        axes[1, 0].set_title("Optimization trajectory")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        bars = axes[1, 1].bar(np.arange(len(fractions)), fractions)
        axes[1, 1].set_title("Probe mode fractions")
        axes[1, 1].set_xticks(np.arange(len(fractions)))
        for bar, fraction in zip(bars, fractions):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{fraction*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        axes[1, 2].scatter(continuous[:, 1], continuous[:, 0], s=3)
        axes[1, 2].invert_yaxis()
        axes[1, 2].set_aspect("equal", adjustable="box")
        axes[1, 2].set_title("Selected scan positions")

        fig.suptitle(
            f"Ptychography Platform V0.8.2 — selected iter {selected_iteration} "
            f"(last iter {last_iteration})"
        )
        fig.tight_layout()
        fig.savefig(self.final_dir / "final_reconstruction_summary.png", dpi=190)
        plt.close(fig)

        np.save(self.final_dir / "obja.npy", amp)
        np.save(self.final_dir / "objp.npy", phase)
        np.save(
            self.final_dir / "object_complex.npy",
            (amp * np.exp(1j * phase)).astype(np.complex64),
        )
        np.save(self.final_dir / "probe.npy", probe_np)
        np.save(self.final_dir / "positions.npy", continuous)

        if bool(self.config["output"].get("save_last_and_best", True)):
            last = self._snapshot_to_numpy(last_snapshot)
            np.save(self.final_dir / "last_obja.npy", last["obja"])
            np.save(self.final_dir / "last_objp.npy", last["objp"])
            np.save(self.final_dir / "last_probe.npy", last["probe"])
            last_positions = last["crop_positions"].astype(np.float32) + last["probe_pos_shifts"]
            np.save(self.final_dir / "last_positions.npy", last_positions)

        selected_row = history[selected_iteration - 1]
        last_row = history[-1]
        summary = {
            "platform_version": "0.8.1",
            "iterations_run": int(last_iteration),
            "selected_iteration": int(selected_iteration),
            "selected_data_loss": float(selected_row["data_loss"]),
            "selected_quality_score": float(selected_score),
            "last_data_loss": float(last_row["data_loss"]),
            "last_quality_score": float(last_row["selection_score"]),
            "object_shape": list(amp.shape),
            "probe_shape": list(probe_np.shape),
            "probe_mode_fractions": [float(v) for v in fractions],
            "probe_anchor_nrmse": float(selected_row["probe_anchor_nrmse"]),
            "probe_support_leakage": float(selected_row["probe_support_leakage"]),
            "position_rms_drift_px": float(selected_row["position_rms_drift_px"]),
            "dx_angstrom_per_object_pixel": metadata["dx_angstrom_per_object_pixel"],
            "scan_step_object_pixels": metadata["scan_step_object_pixels"],
        }
        (self.final_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )