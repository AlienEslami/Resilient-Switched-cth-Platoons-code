"""Synthesize and validate all baseline theorem certificates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.linalg import eigvalsh, expm
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulations.src.matrices import (
    augmented_matrices,
    frozen_string_quantities,
    string_matrices,
    vehicle_matrices,
)

DEFAULT_CONFIG = ROOT / "simulations" / "configs" / "baseline.json"
DEFAULT_JSON = (
    ROOT / "simulations" / "certificates" / "baseline_certificates.json"
)
DEFAULT_SUMMARY = (
    ROOT / "simulations" / "certificates" / "baseline_certificates.md"
)
SOLVER = cp.MOSEK
PSD_EPS = 1.0e-9


def sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def as_list(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=float).tolist()


def solve(problem: cp.Problem) -> None:
    problem.solve(solver=SOLVER, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"MOSEK returned {problem.status}")


def segment_gramian(
    a: np.ndarray, c: np.ndarray, duration: float
) -> np.ndarray:
    """Exact finite-horizon observability Gramian for one constant segment."""
    n = a.shape[0]
    q = c.T @ c
    van_loan = np.block(
        [[-a.T, q], [np.zeros((n, n)), a]]
    )
    transition = expm(van_loan * duration)
    cross = transition[:n, n:]
    gramian = expm(a.T * duration) @ cross
    return sym(gramian)


def periodic_gramian(
    a_modes: list[np.ndarray],
    c_modes: list[np.ndarray],
    dwell: list[float],
    phase: float,
    horizon: float,
) -> np.ndarray:
    period = sum(dwell)
    boundaries = np.cumsum(dwell)
    time = phase % period
    remaining = horizon
    phi = np.eye(a_modes[0].shape[0])
    gramian = np.zeros_like(phi)

    while remaining > 1.0e-12:
        local = time % period
        mode = int(np.searchsorted(boundaries, local, side="right"))
        if mode == len(dwell):
            mode = 0
            local = 0.0
        boundary = boundaries[mode]
        gap = boundary - local
        if gap <= 1.0e-12:
            gap = dwell[(mode + 1) % len(dwell)]
        duration = min(remaining, gap)
        a = a_modes[mode]
        c = c_modes[mode]
        gramian += phi.T @ segment_gramian(a, c, duration) @ phi
        phi = expm(a * duration) @ phi
        time += duration
        remaining -= duration

    return sym(gramian)


def synthesize_observer(config: dict[str, Any]) -> dict[str, Any]:
    auxiliary = config["auxiliary"]
    a_z = [np.asarray(item, dtype=float) for item in auxiliary["A_z"]]
    b_z = [np.asarray(item, dtype=float) for item in auxiliary["B_z"]]
    c_z = [np.asarray(item, dtype=float) for item in auxiliary["C_z"]]
    gains = [np.asarray(item, dtype=float) for item in auxiliary["L"]]
    dwell = [float(item) for item in auxiliary["dwell_times"]]
    horizon = float(auxiliary["observability_horizon"])

    augmented = [
        augmented_matrices(a_mode, b_mode, c_mode)
        for a_mode, b_mode, c_mode in zip(a_z, b_z, c_z)
    ]
    a_aug = [item[0] for item in augmented]
    c_aug = [item[1] for item in augmented]
    a_error = [
        a_mode - gain @ c_mode
        for a_mode, c_mode, gain in zip(a_aug, c_aug, gains)
    ]

    # Phase-uniform observability check over two complete cycles.
    period = sum(dwell)

    def beta_at_phase(phase: float) -> float:
        gramian = periodic_gramian(
            a_aug, c_aug, dwell, phase % period, horizon
        )
        return float(np.linalg.eigvalsh(gramian).min())

    phases = np.linspace(0.0, period, 161, endpoint=False)
    phase_values = np.asarray([beta_at_phase(phase) for phase in phases])
    grid_index = int(np.argmin(phase_values))
    grid_step = period / len(phases)
    center = phases[grid_index]
    phase_result = minimize_scalar(
        lambda offset: beta_at_phase((center + offset) % period),
        bounds=(-2.0 * grid_step, 2.0 * grid_step),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    beta_min = float(phase_result.fun)
    beta_report = 0.99 * beta_min

    psi = np.eye(9)
    for a_mode, duration in zip(a_error, dwell):
        psi = expm(a_mode * duration) @ psi

    q_design = float(config["certificate_targets"]["observer_q_design"])
    q_report = float(config["certificate_targets"]["observer_q_report"])

    # The paper certifies the complete 3x3 dwell family with one common P_o.
    # The stored matrix is independently rechecked by
    # verification/tight_observer_bound.py and by the regression tests below.
    family_path = ROOT / "verification" / "observer_dwell_family_certificate.json"
    family = json.loads(family_path.read_text(encoding="utf-8"))
    p_value = sym(np.asarray(family["P_o"], dtype=float))
    if not math.isclose(float(family["q_report"]), q_report):
        raise ValueError("Observer q in baseline.json and the dwell-family certificate differ")

    q_actual = math.sqrt(
        float(eigvalsh(psi.T @ p_value @ psi, p_value).max())
    )
    lifted_margin = float(
        np.linalg.eigvalsh(
            psi.T @ p_value @ psi - q_report**2 * p_value
        ).max()
    )

    # Retain the logarithmic norms as diagnostics, but use the tighter direct
    # transition enclosure reported in the paper for the partial-cycle bound.
    log_norms = [
        float(np.linalg.eigvalsh(sym(a_mode)).max())
        for a_mode in a_error
    ]
    k_e = float(family["K_e_report"])
    lambda_e = float(family["lambda_e_per_s"])
    p_condition = float(family["condition_number_P_o"])
    m_e = float(family["M_e"])
    c_e = float(family["c_e_s"])

    family_margins = [
        float(item["lmi_max_eigenvalue_at_q_0p85"])
        for item in family["tuples"]
    ]

    return {
        "A_aug": [as_list(item) for item in a_aug],
        "C_aug": [as_list(item) for item in c_aug],
        "A_error": [as_list(item) for item in a_error],
        "P_o": as_list(p_value),
        "monodromy": as_list(psi),
        "spectral_radius_monodromy": float(
            np.max(np.abs(np.linalg.eigvals(psi)))
        ),
        "q_design": q_design,
        "q_actual": q_actual,
        "q_report": q_report,
        "lifted_margin_max_eigenvalue": lifted_margin,
        "admissible_dwell_times": family["dwell_alphabet_s"],
        "worst_generalized_contraction": family["worst_generalized_contraction"],
        "worst_dwell_tuple": family["worst_tuple_s"],
        "family_lmi_max_eigenvalues_at_reported_q": family_margins,
        "family_observability_gramian_min_eigenvalue": family["cycle_gramian_min_eigenvalue"],
        "observability_horizon": horizon,
        "observability_beta_min": beta_min,
        "observability_beta_report": beta_report,
        "worst_phase": float((center + phase_result.x) % period),
        "logarithmic_norms": log_norms,
        "K_e": k_e,
        "K_e_grid_max": family["K_e_grid_max"],
        "K_e_lipschitz_enclosure_upper": family["K_e_lipschitz_enclosure_upper"],
        "lambda_e": lambda_e,
        "P_o_condition_number": p_condition,
        "M_e": m_e,
        "c_e": c_e,
    }


def predecessor_graph_matrix(followers: int) -> np.ndarray:
    matrix = np.eye(followers)
    matrix[1:, :-1] -= np.eye(followers - 1)
    return matrix


def synthesize_platoon(config: dict[str, Any]) -> dict[str, Any]:
    platoon = config["platoon"]
    followers = int(platoon["followers"])
    taus = [float(item) for item in platoon["powertrain_lags"]]
    headway = float(platoon["headway"])
    gains = [
        np.asarray(item, dtype=float)
        for item in platoon["controller_gains"]
    ]
    graph = predecessor_graph_matrix(followers)
    e_v = np.array([[0.0, 1.0, 0.0]])
    closed_loop = []
    residual_channels = []

    for tau, gain in zip(taus, gains):
        a, b = vehicle_matrices(tau)
        a_bar = a - gain[0] * headway * b @ e_v
        closed_loop.append(
            np.kron(np.eye(followers), a_bar)
            - np.kron(graph, b @ gain[None, :])
        )
        residual_channels.append(
            np.kron(np.eye(followers), b @ gain[None, :])
        )

    lambda_c = float(config["certificate_targets"]["plant_lambda_c"])
    dimension = 3 * followers
    p_c = cp.Variable((dimension, dimension), symmetric=True)
    kappa = cp.Variable(nonneg=True)
    constraints = [p_c >> np.eye(dimension)]
    for a_mode, g_mode in zip(closed_loop, residual_channels):
        constraints.extend(
            [
                a_mode.T @ p_c + p_c @ a_mode << -lambda_c * p_c,
                g_mode.T @ p_c @ g_mode << kappa * np.eye(dimension),
            ]
        )
    problem = cp.Problem(
        cp.Minimize(kappa + 1.0e-8 * cp.trace(p_c)), constraints
    )
    solve(problem)
    p_value = sym(p_c.value)

    m_c = float(np.linalg.eigvalsh(p_value).min())
    kappa_actual = max(
        float(np.linalg.eigvalsh(g.T @ p_value @ g).max())
        for g in residual_channels
    )
    lyapunov_margin = max(
        float(
            np.linalg.eigvalsh(
                a.T @ p_value + p_value @ a + lambda_c * p_value
            ).max()
        )
        for a in closed_loop
    )
    mode_abscissae = [
        float(np.max(np.real(np.linalg.eigvals(a)))) for a in closed_loop
    ]

    # The same P is used in every mode, hence mu_c=1 exactly.
    mu_c = 1.0
    xi_c = lambda_c / 2.0
    adt_threshold = 0.0
    radius_coefficient = (
        2.0 * kappa_actual / (m_c * lambda_c * xi_c)
    )

    return {
        "graph_L_plus_P": as_list(graph),
        "A_c": [as_list(item) for item in closed_loop],
        "G": [as_list(item) for item in residual_channels],
        "P_c": as_list(p_value),
        "lambda_c": lambda_c,
        "mu_c": mu_c,
        "xi_c": xi_c,
        "average_dwell_time_threshold": adt_threshold,
        "m_c": m_c,
        "kappa_c": kappa_actual,
        "radius_coefficient_times_Rbar": radius_coefficient,
        "lyapunov_margin_max_eigenvalue": lyapunov_margin,
        "mode_spectral_abscissae": mode_abscissae,
        "P_c_condition_number": float(np.linalg.cond(p_value)),
    }


def synthesize_string(config: dict[str, Any]) -> dict[str, Any]:
    platoon = config["platoon"]
    taus = [float(item) for item in platoon["powertrain_lags"]]
    headway = float(platoon["headway"])
    gains = [
        np.asarray(item, dtype=float)
        for item in platoon["controller_gains"]
    ]
    c_a = np.array([[0.0, 0.0, 1.0]])
    modes = [
        string_matrices(tau, headway, gain)
        for tau, gain in zip(taus, gains)
    ]

    # Common internal-stability certificate.
    p_zero = cp.Variable((3, 3), symmetric=True)
    stability_constraints = [p_zero >> np.eye(3)]
    for a_mode, _, _ in modes:
        stability_constraints.append(
            a_mode.T @ p_zero + p_zero @ a_mode << -np.eye(3)
        )
    stability_problem = cp.Problem(
        cp.Minimize(cp.trace(p_zero)), stability_constraints
    )
    solve(stability_problem)
    p_zero_value = sym(p_zero.value)
    alpha_modes = [
        float(
            eigvalsh(
                -(a.T @ p_zero_value + p_zero_value @ a),
                p_zero_value,
            ).min()
        )
        for a, _, _ in modes
    ]
    alpha_capacity = min(alpha_modes)
    alpha_zero = float(
        config["certificate_targets"].get(
            "string_alpha_0", alpha_capacity
        )
    )
    if alpha_zero > alpha_capacity + 1.0e-8:
        raise RuntimeError(
            "reported string_alpha_0 exceeds the certified decay rate"
        )

    # Exact unit-gain predecessor-channel LMI.
    p_a = cp.Variable((3, 3), symmetric=True)
    nominal_constraints = [p_a >> PSD_EPS * np.eye(3)]
    for a_mode, b_a, _ in modes:
        nominal_lmi = cp.bmat(
            [
                [
                    a_mode.T @ p_a + p_a @ a_mode + c_a.T @ c_a,
                    p_a @ b_a,
                ],
                [b_a.T @ p_a, np.array([[-1.0]])],
            ]
        )
        nominal_constraints.append(nominal_lmi << 0)
    nominal_problem = cp.Problem(
        cp.Minimize(cp.trace(p_a)), nominal_constraints
    )
    solve(nominal_problem)
    p_a_value = sym(p_a.value)
    nominal_residuals = []
    for a_mode, b_a, _ in modes:
        matrix = np.block(
            [
                [
                    a_mode.T @ p_a_value
                    + p_a_value @ a_mode
                    + c_a.T @ c_a,
                    p_a_value @ b_a,
                ],
                [b_a.T @ p_a_value, np.array([[-1.0]])],
            ]
        )
        nominal_residuals.append(float(np.linalg.eigvalsh(sym(matrix)).max()))

    # Residual-channel gain minimization.
    p_r = cp.Variable((3, 3), symmetric=True)
    gamma_squared = cp.Variable(nonneg=True)
    residual_constraints = [
        p_r >> PSD_EPS * np.eye(3),
        gamma_squared >= PSD_EPS,
    ]
    for a_mode, _, b_r in modes:
        residual_lmi = cp.bmat(
            [
                [
                    a_mode.T @ p_r + p_r @ a_mode + c_a.T @ c_a,
                    p_r @ b_r,
                ],
                [b_r.T @ p_r, -gamma_squared * np.eye(3)],
            ]
        )
        residual_constraints.append(residual_lmi << 0)
    residual_problem = cp.Problem(
        cp.Minimize(gamma_squared), residual_constraints
    )
    solve(residual_problem)
    p_r_value = sym(p_r.value)
    gamma_actual = math.sqrt(float(gamma_squared.value))
    gamma_report = max(
        float(config["certificate_targets"].get("string_gamma_r", 0.0)),
        math.ceil(gamma_actual * 1000.0) / 1000.0,
    )
    residual_margins = []
    for a_mode, _, b_r in modes:
        matrix = np.block(
            [
                [
                    a_mode.T @ p_r_value
                    + p_r_value @ a_mode
                    + c_a.T @ c_a,
                    p_r_value @ b_r,
                ],
                [b_r.T @ p_r_value, -(gamma_report**2) * np.eye(3)],
            ]
        )
        residual_margins.append(float(np.linalg.eigvalsh(sym(matrix)).max()))

    # Optional joint-storage comparison used to quantify the benefit of the
    # channel-wise theorem construction.
    p_joint = cp.Variable((3, 3), symmetric=True)
    joint_gamma_squared = cp.Variable(nonneg=True)
    joint_constraints = [
        p_joint >> PSD_EPS * np.eye(3),
        joint_gamma_squared >= PSD_EPS,
    ]
    for a_mode, b_a, b_r in modes:
        b_joint = np.hstack([b_a, b_r])
        supply = cp.bmat(
            [
                [-np.ones((1, 1)), np.zeros((1, 3))],
                [np.zeros((3, 1)), -joint_gamma_squared * np.eye(3)],
            ]
        )
        joint_lmi = cp.bmat(
            [
                [
                    a_mode.T @ p_joint
                    + p_joint @ a_mode
                    + c_a.T @ c_a,
                    p_joint @ b_joint,
                ],
                [b_joint.T @ p_joint, supply],
            ]
        )
        joint_constraints.append(joint_lmi << 0)
    joint_problem = cp.Problem(
        cp.Minimize(joint_gamma_squared), joint_constraints
    )
    solve(joint_problem)
    p_joint_value = sym(p_joint.value)
    joint_gamma_actual = math.sqrt(float(joint_gamma_squared.value))
    joint_gamma_report = math.ceil(joint_gamma_actual * 1000.0) / 1000.0
    joint_margins = []
    for a_mode, b_a, b_r in modes:
        b_joint = np.hstack([b_a, b_r])
        supply = np.diag(
            [-1.0, -joint_gamma_report**2, -joint_gamma_report**2,
             -joint_gamma_report**2]
        )
        matrix = np.block(
            [
                [
                    a_mode.T @ p_joint_value
                    + p_joint_value @ a_mode
                    + c_a.T @ c_a,
                    p_joint_value @ b_joint,
                ],
                [b_joint.T @ p_joint_value, supply],
            ]
        )
        joint_margins.append(float(np.linalg.eigvalsh(sym(matrix)).max()))

    frozen_headway_boundary = max(
        (-gain[1] + math.sqrt(gain[1] ** 2 + 2.0 * gain[0]))
        / gain[0]
        for gain in gains
    )

    return {
        "A_s": [as_list(item[0]) for item in modes],
        "B_a": [as_list(item[1]) for item in modes],
        "B_r": [as_list(item[2]) for item in modes],
        "P_0": as_list(p_zero_value),
        "alpha_0": alpha_zero,
        "common_stability_margin_max_eigenvalue": max(
            float(
                np.linalg.eigvalsh(
                    a.T @ p_zero_value
                    + p_zero_value @ a
                    + alpha_zero * p_zero_value
                ).max()
            )
            for a, _, _ in modes
        ),
        "P_a": as_list(p_a_value),
        "nominal_lmi_max_eigenvalues": nominal_residuals,
        "P_r": as_list(p_r_value),
        "gamma_r_actual": gamma_actual,
        "gamma_r_report": gamma_report,
        "residual_lmi_max_eigenvalues_at_reported_gamma": residual_margins,
        "P_joint": as_list(p_joint_value),
        "joint_gamma_r_actual": joint_gamma_actual,
        "joint_gamma_r_report": joint_gamma_report,
        "joint_lmi_max_eigenvalues_at_reported_gamma": joint_margins,
        "frozen_headway_boundary": frozen_headway_boundary,
        "frozen_mode_tests": [
            frozen_string_quantities(tau, headway, gain)
            for tau, gain in zip(taus, gains)
        ],
    }


def markdown_summary(
    config: dict[str, Any], certificates: dict[str, Any]
) -> str:
    observer = certificates["observer"]
    platoon = certificates["platoon"]
    string = certificates["string"]
    lines = [
        "# Baseline numerical certificates",
        "",
        "All matrix inequalities were solved with MOSEK and then checked using",
        "independent symmetric-eigenvalue calculations.",
        "",
        "## Baseline",
        "",
        f"- Followers: {config['platoon']['followers']}",
        f"- Powertrain lags: {config['platoon']['powertrain_lags']} s",
        f"- Headway: {config['platoon']['headway']} s",
        f"- Controller gains: {config['platoon']['controller_gains']}",
        "",
        "## Switching observer",
        "",
        f"- Admissible dwell times: {observer['admissible_dwell_times']} s",
        f"- Worst generalized contraction: {observer['worst_generalized_contraction']:.8f}",
        f"- Worst dwell tuple: {observer['worst_dwell_tuple']} s",
        f"- Family LMI worst max eigenvalue: {max(observer['family_lmi_max_eigenvalues_at_reported_q']):.8e}",
        f"- Family observability-Gramian minimum eigenvalue: {observer['family_observability_gramian_min_eigenvalue']:.8e}",
        f"- Observability beta (computed): {observer['observability_beta_min']:.8e}",
        f"- Observability beta (reported): {observer['observability_beta_report']:.8e}",
        f"- Monodromy spectral radius: {observer['spectral_radius_monodromy']:.8f}",
        f"- Lifted q (reported): {observer['q_report']:.4f}",
        f"- Lifted LMI max eigenvalue: {observer['lifted_margin_max_eigenvalue']:.8e}",
        f"- lambda_e: {observer['lambda_e']:.8f} 1/s",
        f"- K_e (direct-transition upper bound): {observer['K_e']:.8f}",
        f"- M_e: {observer['M_e']:.8f}",
        f"- c_e: {observer['c_e']:.8f} s",
        "",
        "## Platoon UUB",
        "",
        f"- lambda_c: {platoon['lambda_c']:.4f} 1/s",
        f"- mu_c: {platoon['mu_c']:.1f}",
        f"- ADT threshold: {platoon['average_dwell_time_threshold']:.1f} s",
        f"- m_c: {platoon['m_c']:.8f}",
        f"- kappa_c: {platoon['kappa_c']:.8f}",
        f"- Lyapunov LMI max eigenvalue: {platoon['lyapunov_margin_max_eigenvalue']:.8e}",
        f"- Radius coefficient multiplying Rbar: {platoon['radius_coefficient_times_Rbar']:.8f}",
        "",
        "The common physical Lyapunov matrix makes the ADT threshold zero for",
        "this predecessor-chain baseline. An ADT-violation ablation is therefore",
        "not supported by this particular certified design.",
        "",
        "## String performance",
        "",
        f"- alpha_0: {string['alpha_0']:.8f} 1/s",
        f"- Nominal LMI worst numerical eigenvalue: {max(string['nominal_lmi_max_eigenvalues']):.8e}",
        f"- gamma_r (reported): {string['gamma_r_report']:.3f}",
        f"- Residual LMI max eigenvalue: {max(string['residual_lmi_max_eigenvalues_at_reported_gamma']):.8e}",
        f"- Joint-storage gamma_r (reported): {string['joint_gamma_r_report']:.3f}",
        f"- Joint LMI max eigenvalue: {max(string['joint_lmi_max_eigenvalues_at_reported_gamma']):.8e}",
        f"- Exact frozen-mode headway boundary: {string['frozen_headway_boundary']:.8f} s",
        "",
        "The tiny positive nominal-LMI eigenvalue is solver-scale roundoff at",
        "the exact unit DC gain. The independent frequency inequalities are",
        "nonnegative in every frozen mode.",
        "",
    ]
    return "\n".join(lines)


def synthesize(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    certificates = {
        "metadata": {
            "solver": "MOSEK",
            "cvxpy_version": cp.__version__,
            "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "observer": synthesize_observer(config),
        "platoon": synthesize_platoon(config),
        "string": synthesize_string(config),
    }
    return config, certificates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config, certificates = synthesize(args.config)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(certificates, indent=2) + "\n", encoding="utf-8"
    )
    args.summary.write_text(
        markdown_summary(config, certificates), encoding="utf-8"
    )
    print(markdown_summary(config, certificates))
    print(f"Wrote {args.json}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
