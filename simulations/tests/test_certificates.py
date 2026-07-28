"""Independent checks for the exported baseline certificates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from simulations.src.certificates import periodic_gramian

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE_FILE = (
    ROOT / "simulations" / "certificates" / "baseline_certificates.json"
)
CONFIG_FILE = ROOT / "simulations" / "configs" / "baseline.json"


def load_certificates() -> dict:
    return json.loads(CERTIFICATE_FILE.read_text(encoding="utf-8"))


def test_observer_lifted_contraction() -> None:
    observer = load_certificates()["observer"]
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    p_o = np.asarray(observer["P_o"])
    psi = np.asarray(observer["monodromy"])
    q = float(observer["q_report"])
    residual = psi.T @ p_o @ psi - q**2 * p_o
    assert np.linalg.eigvalsh(p_o).min() > 0.0
    assert np.linalg.eigvalsh(residual).max() < 0.0
    assert observer["observability_beta_report"] > 0.0

    gramian = periodic_gramian(
        [np.asarray(item) for item in observer["A_aug"]],
        [np.asarray(item) for item in observer["C_aug"]],
        config["auxiliary"]["dwell_times"],
        float(observer["worst_phase"]),
        float(observer["observability_horizon"]),
    )
    assert (
        np.linalg.eigvalsh(gramian).min()
        >= observer["observability_beta_report"]
    )


def test_common_platoon_certificate() -> None:
    platoon = load_certificates()["platoon"]
    p_c = np.asarray(platoon["P_c"])
    lambda_c = float(platoon["lambda_c"])
    assert np.linalg.eigvalsh(p_c).min() > 0.0
    for a_c in platoon["A_c"]:
        a_c = np.asarray(a_c)
        residual = a_c.T @ p_c + p_c @ a_c + lambda_c * p_c
        assert np.linalg.eigvalsh(residual).max() < 1.0e-6
    for g in platoon["G"]:
        g = np.asarray(g)
        residual = g.T @ p_c @ g - platoon["kappa_c"] * np.eye(p_c.shape[0])
        assert np.linalg.eigvalsh(residual).max() < 1.0e-6
    assert platoon["mu_c"] == 1.0


def test_string_certificates() -> None:
    string = load_certificates()["string"]
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    expected_mode_gains = np.array(
        [
            [0.47, 0.99, 0.44],
            [0.50, 1.00, 0.50],
            [0.53, 1.01, 0.70],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(config["platoon"]["controller_gains"]),
        expected_mode_gains,
    )
    c_a = np.array([[0.0, 0.0, 1.0]])
    p_a = np.asarray(string["P_a"])
    p_r = np.asarray(string["P_r"])
    p_zero = np.asarray(string["P_0"])
    p_joint = np.asarray(string["P_joint"])
    alpha_zero = float(string["alpha_0"])
    gamma = float(string["gamma_r_report"])
    joint_gamma = float(string["joint_gamma_r_report"])

    assert np.linalg.eigvalsh(p_a).min() > 0.0
    assert np.linalg.eigvalsh(p_r).min() > 0.0
    for a_s, b_a, b_r in zip(
        string["A_s"], string["B_a"], string["B_r"]
    ):
        a_s = np.asarray(a_s)
        b_a = np.asarray(b_a)
        b_r = np.asarray(b_r)
        nominal = np.block(
            [
                [
                    a_s.T @ p_a + p_a @ a_s + c_a.T @ c_a,
                    p_a @ b_a,
                ],
                [b_a.T @ p_a, np.array([[-1.0]])],
            ]
        )
        residual = np.block(
            [
                [
                    a_s.T @ p_r + p_r @ a_s + c_a.T @ c_a,
                    p_r @ b_r,
                ],
                [b_r.T @ p_r, -(gamma**2) * np.eye(3)],
            ]
        )
        assert eigvalsh(0.5 * (nominal + nominal.T)).max() < 1.0e-6
        assert eigvalsh(0.5 * (residual + residual.T)).max() < 0.0
        stability = (
            a_s.T @ p_zero + p_zero @ a_s + alpha_zero * p_zero
        )
        b_joint = np.hstack([b_a, b_r])
        joint = np.block(
            [
                [
                    a_s.T @ p_joint + p_joint @ a_s + c_a.T @ c_a,
                    p_joint @ b_joint,
                ],
                [
                    b_joint.T @ p_joint,
                    np.diag([-1.0, -joint_gamma**2, -joint_gamma**2, -joint_gamma**2]),
                ],
            ]
        )
        assert eigvalsh(0.5 * (stability + stability.T)).max() < 1.0e-6
        assert eigvalsh(0.5 * (joint + joint.T)).max() < 1.0e-6

    frozen_tests = string["frozen_mode_tests"]
    np.testing.assert_allclose(
        [item["q0"] for item in frozen_tests],
        [0.494816, 0.560000, 0.629216],
    )
    np.testing.assert_allclose(
        [item["q1"] for item in frozen_tests],
        [0.947600, 0.400000, 0.095600],
    )
    assert all(item["nonamplifying"] for item in frozen_tests)
    assert string["joint_gamma_r_report"] > string["gamma_r_report"]
