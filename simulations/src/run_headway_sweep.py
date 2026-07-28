"""Compare headway frequency, switched-LMI, and trajectory boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import cvxpy as cp
import matplotlib
import numpy as np

from simulations.src.certificates import PSD_EPS, SOLVER
from simulations.src.matrices import string_matrices
from simulations.src.model import PlatoonSimulator, SimulationOptions
from simulations.src.run_string_and_ablations import leader_disturbance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "simulations" / "configs" / "baseline.json"
FIGURE_DIR = ROOT / "simulations" / "figures"
RESULT_DIR = ROOT / "simulations" / "results"


def frozen_gain(tau: float, headway: float, gain: np.ndarray) -> float:
    frequencies = np.concatenate(([0.0], np.logspace(-4.0, 2.0, 20000)))
    s = 1j * frequencies
    k1, k2, k3 = gain
    numerator = k3 * s**2 + k2 * s + k1
    denominator = (
        tau * s**3
        + (1.0 + k3) * s**2
        + (k2 + k1 * headway) * s
        + k1
    )
    return float(np.max(np.abs(numerator / denominator)))


def common_switched_gain(
    taus: np.ndarray, headway: float, gains: np.ndarray
) -> tuple[float, float]:
    c_a = np.array([[0.0, 0.0, 1.0]])
    p_a = cp.Variable((3, 3), symmetric=True)
    gain_squared = cp.Variable(nonneg=True)
    constraints = [
        p_a >> PSD_EPS * np.eye(3),
        gain_squared >= PSD_EPS,
    ]
    modes = [
        string_matrices(float(tau), headway, gain)
        for tau, gain in zip(taus, gains)
    ]
    for a_mode, b_a, _ in modes:
        constraints.append(
            cp.bmat(
                [
                    [
                        a_mode.T @ p_a + p_a @ a_mode + c_a.T @ c_a,
                        p_a @ b_a,
                    ],
                    [b_a.T @ p_a, -gain_squared * np.ones((1, 1))],
                ]
            )
            << 0
        )
    problem = cp.Problem(cp.Minimize(gain_squared), constraints)
    problem.solve(solver=SOLVER, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return float("nan"), float("nan")

    p_value = 0.5 * (p_a.value + p_a.value.T)
    gamma = float(np.sqrt(gain_squared.value))
    residuals = []
    for a_mode, b_a, _ in modes:
        matrix = np.block(
            [
                [
                    a_mode.T @ p_value + p_value @ a_mode + c_a.T @ c_a,
                    p_value @ b_a,
                ],
                [b_a.T @ p_value, np.array([[-gamma**2]])],
            ]
        )
        residuals.append(
            float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).max())
        )
    return gamma, max(residuals)


def observed_tail_ratio(headway: float) -> float:
    result = PlatoonSimulator(
        SimulationOptions(
            duration=90.0,
            step=0.01,
            sample_step=0.02,
            attack="none",
            perturb_initial_state=False,
            headway_override=headway,
        ),
        leader_command=leader_disturbance,
    ).run()
    energies = np.sqrt(
        np.trapezoid(result.physical[:, :, 2] ** 2, result.time, axis=0)
    )
    return float(energies[-1] / energies[0])


def crossing(headways: np.ndarray, values: np.ndarray) -> float:
    for index in range(1, len(headways)):
        if values[index - 1] > 1.0 and values[index] <= 1.0 + 1.0e-5:
            x0, x1 = headways[index - 1 : index + 1]
            y0, y1 = values[index - 1 : index + 1]
            if np.isclose(y0, y1):
                return float(x1)
            return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def run_sweep() -> dict[str, object]:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    taus = np.asarray(config["platoon"]["powertrain_lags"], dtype=float)
    gains = np.asarray(config["platoon"]["controller_gains"], dtype=float)
    exact_boundary = max(
        float((-gain[1] + np.sqrt(gain[1] ** 2 + 2.0 * gain[0])) / gain[0])
        for gain in gains
    )

    headways = np.linspace(0.0, 1.6, 33)
    frozen = np.array(
        [
            max(
                frozen_gain(tau, h, gain)
                for tau, gain in zip(taus, gains)
            )
            for h in headways
        ]
    )
    common_results = [
        common_switched_gain(taus, h, gains) for h in headways
    ]
    common = np.array([item[0] for item in common_results])
    residual = np.array([item[1] for item in common_results])

    observed_headways = np.linspace(0.0, 1.6, 9)
    observed = np.array([observed_tail_ratio(h) for h in observed_headways])

    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    axis.plot(headways, frozen, color="#4d4d4d", label="worst frozen-mode gain")
    axis.plot(headways, common, color="#1f77b4", label="common switched-LMI gain")
    axis.plot(
        observed_headways,
        observed,
        color="#d62728",
        marker="o",
        linestyle="--",
        label="vehicle-7 energy ratio",
    )
    axis.axhline(1.0, color="#111111", linestyle=":", linewidth=1.0)
    axis.axvline(
        exact_boundary,
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.1,
        label=rf"exact frozen boundary $h={exact_boundary:.3f}$ s",
    )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1.6)
    axis.set_xlabel("headway $h$ (s)")
    axis.set_ylabel("gain or finite-horizon energy ratio")
    axis.set_title("Headway feasibility and observed propagation")
    axis.legend(loc="upper right")
    axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / "headway_sweep.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "headway_sweep.png", bbox_inches="tight", dpi=300)
    plt.close(figure)

    metrics = {
        "headways": headways.tolist(),
        "worst_frozen_mode_gain": frozen.tolist(),
        "common_switched_lmi_gain": common.tolist(),
        "common_lmi_worst_numeric_residual": residual.tolist(),
        "observed_headways": observed_headways.tolist(),
        "vehicle_7_energy_ratio": observed.tolist(),
        "exact_frozen_boundary_s": exact_boundary,
        "sampled_common_lmi_boundary_s": crossing(headways, common),
        "sampled_observed_boundary_s": crossing(observed_headways, observed),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "headway_sweep_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
        }
    )
    summary = run_sweep()
    print(
        json.dumps(
            {
                "exact_frozen_boundary_s": summary["exact_frozen_boundary_s"],
                "sampled_common_lmi_boundary_s": summary[
                    "sampled_common_lmi_boundary_s"
                ],
                "sampled_observed_boundary_s": summary[
                    "sampled_observed_boundary_s"
                ],
            },
            indent=2,
        )
    )
