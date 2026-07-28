"""Verify Assumption 5 over the admissible auxiliary-mode dwell family.

The script searches for a common ``P_o`` and the minimum contraction factor
``q`` by bisection using CLARABEL. The reported result is the worst case over
all admissible dwell tuples.

Run from the repository root with::

    python simulations/tools/verify_assumption5_dwell_family.py
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import expm


def blk(Az, Bz):
    Z = np.zeros((3, 3))
    return np.block([[Az, Z, -Bz], [Z, Z, Z], [Z, Z, Z]])


def Cmat(Cz):
    return np.block([[Cz, np.eye(3), np.zeros((3, 3))]])


d = np.diag
Az1, Az2 = d([-1, -1.3, -1.6]), d([-2, -1, -0.7])
Bz1, Bz2 = d([1, 0.8, 1.2]), d([0.5, 1.4, 0.9])
Cz1, Cz2 = np.eye(3), d([1.4, 0.7, 1.2])
L1 = np.vstack(
    [
        d([1.386, -0.606, -0.873]),
        d([2.560, 2.187, 1.330]),
        d([-1.805, -2.373, 2.706]),
    ]
)
L2 = np.vstack(
    [
        d([0.620, -0.349, -0.742]),
        d([-2.444, 1.873, 3.193]),
        d([-2.300, 2.796, -1.189]),
    ]
)
Ae1 = blk(Az1, Bz1) - L1 @ Cmat(Cz1)
Ae2 = blk(Az2, Bz2) - L2 @ Cmat(Cz2)
D1 = [0.30, 0.35, 0.40]
D2 = [0.30, 0.35, 0.40]
Trho = max(D1) + max(D2)
Psis = [expm(Ae2 * b) @ expm(Ae1 * a) for a in D1 for b in D2]
rhos = [max(abs(np.linalg.eigvals(Psi))) for Psi in Psis]
print(
    f"tuples={len(Psis)} Trho={Trho:.2f} per-tuple rho: "
    f"min={min(rhos):.4f} max={max(rhos):.4f}",
    flush=True,
)
n = 9


def feas(q2):
    Po = cp.Variable((n, n), symmetric=True)
    cons = [Po >> np.eye(n)] + [
        Psi.T @ Po @ Psi << q2 * Po for Psi in Psis
    ]
    problem = cp.Problem(cp.Minimize(0), cons)
    problem.solve(solver=cp.CLARABEL, verbose=False)
    return Po.value if problem.status.startswith("optimal") else None


lo, hi, Pb = max(rhos) ** 2, 0.999, None
for _ in range(16):
    midpoint = 0.5 * (lo + hi)
    P = feas(midpoint)
    if P is not None:
        hi, Pb = midpoint, P
    else:
        lo = midpoint

worst = max(
    max(np.linalg.eigvals(np.linalg.solve(Pb, Psi.T @ Pb @ Psi)).real)
    for Psi in Psis
)
q = np.sqrt(worst)
lam = -np.log(q) / Trho
print(
    f"COMMON-Po: q_worst={q:.4f}  lambda_e={lam:.4f}  "
    f"cond(Po)={np.linalg.cond(Pb):.2f}",
    flush=True,
)
