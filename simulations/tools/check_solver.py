"""Verify that CVXPY can solve a semidefinite LMI with MOSEK."""

from __future__ import annotations

import sys

import cvxpy as cp
import mosek
import numpy as np


def solve_common_lyapunov_lmi() -> tuple[str, float, float]:
    """Solve a normalized common-Lyapunov feasibility problem."""
    modes = (
        np.array([[-1.0, 0.3], [0.0, -1.8]]),
        np.array([[-1.2, -0.2], [0.4, -1.4]]),
    )
    epsilon = 1.0e-7
    identity = np.eye(2)
    p = cp.Variable((2, 2), symmetric=True)
    constraints = [p >> epsilon * identity, cp.trace(p) == 1.0]
    constraints.extend(
        a.T @ p + p @ a << -epsilon * identity for a in modes
    )

    problem = cp.Problem(cp.Minimize(0.0), constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    if p.value is None:
        return problem.status, float("nan"), float("nan")

    min_p_eigenvalue = float(np.linalg.eigvalsh(p.value).min())
    max_lmi_eigenvalue = max(
        float(np.linalg.eigvalsh(a.T @ p.value + p.value @ a).max())
        for a in modes
    )
    return problem.status, min_p_eigenvalue, max_lmi_eigenvalue


def main() -> int:
    print(f"MOSEK API: {mosek.Env.getversion()}")
    print(f"CVXPY: {cp.__version__}")
    print(f"Installed solvers: {cp.installed_solvers()}")

    status, min_p_eigenvalue, max_lmi_eigenvalue = (
        solve_common_lyapunov_lmi()
    )
    print(f"status: {status}")
    print(f"min eigenvalue(P): {min_p_eigenvalue:.6e}")
    print(f"max LMI eigenvalue: {max_lmi_eigenvalue:.6e}")

    valid_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    passed = (
        status in valid_statuses
        and min_p_eigenvalue > 0.0
        and max_lmi_eigenvalue < 0.0
    )
    print("solver diagnostic: PASS" if passed else "solver diagnostic: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

