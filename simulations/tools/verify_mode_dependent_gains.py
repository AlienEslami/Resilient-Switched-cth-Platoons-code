"""Verify the reported mode-dependent gains using stored common matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulations.src.matrices import (
    frozen_string_quantities,
    string_matrices,
    vehicle_matrices,
)

DEFAULT_CONFIG = ROOT / "simulations" / "configs" / "baseline.json"
DEFAULT_CERTIFICATE = (
    ROOT / "simulations" / "certificates" / "baseline_certificates.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "simulations"
    / "certificates"
    / "mode_dependent_gain_verification.json"
)


def sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def predecessor_graph_matrix(followers: int) -> np.ndarray:
    matrix = np.eye(followers)
    matrix[1:, :-1] -= np.eye(followers - 1)
    return matrix


def verify(
    config: dict[str, Any], certificates: dict[str, Any]
) -> dict[str, Any]:
    platoon = config["platoon"]
    targets = config["certificate_targets"]
    followers = int(platoon["followers"])
    taus = np.asarray(platoon["powertrain_lags"], dtype=float)
    gains = np.asarray(platoon["controller_gains"], dtype=float)
    headway = float(platoon["headway"])
    lambda_c = float(targets["plant_lambda_c"])
    alpha_0 = float(targets["string_alpha_0"])
    gamma_r = float(targets["string_gamma_r"])

    stored_platoon = certificates["platoon"]
    stored_string = certificates["string"]
    p_c = np.asarray(stored_platoon["P_c"], dtype=float)
    p_0 = np.asarray(stored_string["P_0"], dtype=float)
    p_a = np.asarray(stored_string["P_a"], dtype=float)
    p_r = np.asarray(stored_string["P_r"], dtype=float)

    graph = predecessor_graph_matrix(followers)
    e_v = np.array([[0.0, 1.0, 0.0]])
    c_a = np.array([[0.0, 0.0, 1.0]])
    modes: list[dict[str, Any]] = []

    for index, (tau, gain) in enumerate(zip(taus, gains), start=1):
        a, b = vehicle_matrices(float(tau))
        a_bar = a - gain[0] * headway * b @ e_v
        a_c = (
            np.kron(np.eye(followers), a_bar)
            - np.kron(graph, b @ gain[None, :])
        )
        g_c = np.kron(np.eye(followers), b @ gain[None, :])
        a_s, b_a, b_r = string_matrices(float(tau), headway, gain)

        plant_lmi = a_c.T @ p_c + p_c @ a_c + lambda_c * p_c
        stability_lmi = a_s.T @ p_0 + p_0 @ a_s + alpha_0 * p_0
        nominal_lmi = np.block(
            [
                [
                    a_s.T @ p_a + p_a @ a_s + c_a.T @ c_a,
                    p_a @ b_a,
                ],
                [b_a.T @ p_a, np.array([[-1.0]])],
            ]
        )
        residual_lmi = np.block(
            [
                [
                    a_s.T @ p_r + p_r @ a_s + c_a.T @ c_a,
                    p_r @ b_r,
                ],
                [b_r.T @ p_r, -(gamma_r**2) * np.eye(3)],
            ]
        )
        frozen = frozen_string_quantities(float(tau), headway, gain)

        modes.append(
            {
                "mode": index,
                "tau_s": float(tau),
                "controller_gain": gain.tolist(),
                "plant_spectral_abscissa": float(
                    np.max(np.real(np.linalg.eigvals(a_c)))
                ),
                "plant_lmi_max_eigenvalue": float(
                    np.linalg.eigvalsh(sym(plant_lmi)).max()
                ),
                "string_stability_lmi_max_eigenvalue": float(
                    np.linalg.eigvalsh(sym(stability_lmi)).max()
                ),
                "nominal_string_lmi_max_eigenvalue": float(
                    np.linalg.eigvalsh(sym(nominal_lmi)).max()
                ),
                "residual_string_lmi_max_eigenvalue": float(
                    np.linalg.eigvalsh(sym(residual_lmi)).max()
                ),
                "residual_channel_max_eigenvalue": float(
                    np.linalg.eigvalsh(g_c.T @ p_c @ g_c).max()
                ),
                "frozen_mode_test": frozen,
            }
        )

    tolerances = {
        "strict_lmi": -1.0e-9,
        "semidefinite_lmi": 1.0e-8,
    }
    passed = all(
        mode["plant_lmi_max_eigenvalue"] < tolerances["strict_lmi"]
        and mode["string_stability_lmi_max_eigenvalue"]
        < tolerances["strict_lmi"]
        and mode["nominal_string_lmi_max_eigenvalue"]
        <= tolerances["semidefinite_lmi"]
        and mode["residual_string_lmi_max_eigenvalue"]
        < tolerances["strict_lmi"]
        and mode["frozen_mode_test"]["nonamplifying"]
        for mode in modes
    )

    return {
        "passed": passed,
        "reported_targets": {
            "lambda_c_per_s": lambda_c,
            "mu_c": 1.0,
            "xi_c_per_s": lambda_c / 2.0,
            "alpha_0_per_s": alpha_0,
            "gamma_r": gamma_r,
        },
        "worst_residuals": {
            "plant_lmi_max_eigenvalue": max(
                mode["plant_lmi_max_eigenvalue"] for mode in modes
            ),
            "string_stability_lmi_max_eigenvalue": max(
                mode["string_stability_lmi_max_eigenvalue"] for mode in modes
            ),
            "nominal_string_lmi_max_eigenvalue": max(
                mode["nominal_string_lmi_max_eigenvalue"] for mode in modes
            ),
            "residual_string_lmi_max_eigenvalue": max(
                mode["residual_string_lmi_max_eigenvalue"] for mode in modes
            ),
        },
        "modes": modes,
        "tolerances": tolerances,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    certificates = json.loads(args.certificate.read_text(encoding="utf-8"))
    result = verify(config, certificates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
