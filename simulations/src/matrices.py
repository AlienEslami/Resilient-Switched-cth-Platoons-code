"""NumPy-only matrix constructors shared by simulations and certificates."""

from __future__ import annotations

import numpy as np


def vehicle_matrices(tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the third-order vehicle matrices for powertrain lag ``tau``."""
    a = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]]
    )
    b = np.array([[0.0], [0.0], [1.0 / tau]])
    return a, b


def augmented_matrices(
    a_z: np.ndarray, b_z: np.ndarray, c_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the augmented attack-observer state and output matrices."""
    zeros_33 = np.zeros((3, 3))
    a_aug = np.block(
        [
            [a_z, zeros_33, -b_z],
            [zeros_33, zeros_33, zeros_33],
            [zeros_33, zeros_33, zeros_33],
        ]
    )
    c_aug = np.block([c_z, np.eye(3), zeros_33])
    return a_aug, c_aug


def string_matrices(
    tau: float, headway: float, gain: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the local string-system matrices for one physical mode."""
    k1, k2, k3 = gain
    a = np.array(
        [
            [0.0, 1.0, -headway],
            [0.0, 0.0, -1.0],
            [k1 / tau, k2 / tau, -(1.0 + k3) / tau],
        ]
    )
    b_a = np.array([[0.0], [1.0], [k3 / tau]])
    b_r = np.zeros((3, 3))
    b_r[2, :] = gain / tau
    return a, b_a, b_r


def frozen_string_quantities(
    tau: float, headway: float, gain: np.ndarray
) -> dict[str, float | bool]:
    """Evaluate the exact frozen-mode string non-amplification test."""
    k1, k2, k3 = gain
    q0 = k1 * (k1 * headway**2 + 2.0 * headway * k2 - 2.0)
    q1 = 1.0 + 2.0 * k3 - 2.0 * tau * (k2 + k1 * headway)
    second_margin = 4.0 * tau**2 * q0 - q1**2
    nonamplifying = q0 >= 0.0 and (q1 >= 0.0 or second_margin >= 0.0)
    return {
        "q0": float(q0),
        "q1": float(q1),
        "quadratic_margin": float(second_margin),
        "nonamplifying": bool(nonamplifying),
    }
