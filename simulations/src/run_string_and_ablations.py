"""Generate string-propagation and resilient-controller comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

from simulations.src.model import PlatoonSimulator, SimulationOptions

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "simulations" / "figures"
RESULT_DIR = ROOT / "simulations" / "results"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(figure)


def leader_disturbance(t: float) -> float:
    start = 5.0
    stop = 65.0
    if t <= start or t >= stop:
        return 0.0
    local = t - start
    window = np.sin(np.pi * local / (stop - start)) ** 2
    return float(0.8 * window * np.sin(0.55 * local))


def acceleration_energies(time: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    return np.sqrt(np.trapezoid(acceleration**2, time, axis=0))


def run_string_comparison() -> dict[str, list[float] | float]:
    results = {}
    for name, headway in {"cth": 1.2, "constant_offset": 0.0}.items():
        results[name] = PlatoonSimulator(
            SimulationOptions(
                duration=90.0,
                step=0.005,
                sample_step=0.02,
                attack="none",
                perturb_initial_state=False,
                headway_override=headway,
            ),
            leader_command=leader_disturbance,
        ).run()

    energies = {
        name: acceleration_energies(result.time, result.physical[:, :, 2])
        for name, result in results.items()
    }
    ratios = {name: value / value[0] for name, value in energies.items()}

    figure, axis = plt.subplots(figsize=(6.2, 3.6))
    indices = np.arange(8)
    axis.plot(indices, ratios["cth"], marker="o", color="#1f77b4", label=r"CTH, $h=1.2$ s")
    axis.plot(indices, ratios["constant_offset"], marker="s", color="#d62728", label=r"constant offset, $h=0$")
    axis.axhline(1.0, color="#111111", linestyle=":", linewidth=1.0, label="leader energy")
    axis.set_xlabel("vehicle index")
    axis.set_ylabel(r"$\|a_i\|_{2,T}/\|a_0\|_{2,T}$")
    axis.set_xticks(indices)
    axis.set_title("Leader-side disturbance propagation")
    axis.legend(loc="upper left")
    figure.tight_layout()
    save_figure(figure, "string_energy_comparison")

    return {
        "cth_energy_ratios": ratios["cth"].tolist(),
        "constant_offset_energy_ratios": ratios["constant_offset"].tolist(),
        "cth_max_step_ratio": float(np.max(energies["cth"][1:] / energies["cth"][:-1])),
        "constant_offset_max_step_ratio": float(
            np.max(energies["constant_offset"][1:] / energies["constant_offset"][:-1])
        ),
    }


def run_controller_ablation() -> dict[str, float]:
    options = dict(
        duration=35.0,
        step=0.005,
        sample_step=0.02,
        attack="constant",
        perturb_initial_state=True,
    )
    resilient = PlatoonSimulator(
        SimulationOptions(**options, compensate=True)
    ).run()
    uncompensated = PlatoonSimulator(
        SimulationOptions(**options, compensate=False)
    ).run()

    resilient_norm = np.max(np.linalg.norm(resilient.tracking_error, axis=2), axis=1)
    uncompensated_norm = np.max(
        np.linalg.norm(uncompensated.tracking_error, axis=2), axis=1
    )
    resilient_min_gap = np.min(resilient.gaps, axis=1)
    uncompensated_min_gap = np.min(uncompensated.gaps, axis=1)

    figure, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    axes[0].plot(resilient.time, resilient_norm, color="#1f77b4", label="resilient")
    axes[0].plot(uncompensated.time, uncompensated_norm, color="#d62728", linestyle="--", label="uncompensated")
    axes[0].set_ylabel(r"$\max_i\|e_i\|_2$")
    axes[0].set_title("Controller ablation under identical constant FDI")
    axes[0].legend(loc="upper left")
    axes[1].plot(resilient.time, resilient_min_gap, color="#1f77b4")
    axes[1].plot(uncompensated.time, uncompensated_min_gap, color="#d62728", linestyle="--")
    axes[1].axhline(0.0, color="#111111", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("minimum distance (m)")
    figure.tight_layout()
    save_figure(figure, "controller_ablation")

    return {
        "resilient_final_max_tracking_error": float(resilient_norm[-1]),
        "uncompensated_final_max_tracking_error": float(uncompensated_norm[-1]),
        "resilient_minimum_distance": float(np.min(resilient_min_gap)),
        "uncompensated_minimum_distance": float(np.min(uncompensated_min_gap)),
    }


if __name__ == "__main__":
    configure_plotting()
    metrics = {
        "string": run_string_comparison(),
        "controller_ablation": run_controller_ablation(),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "string_and_ablation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
