"""Verify the publication's Assumption 5 dwell-family certificate.

The script loads the reported, deliberately well-conditioned common ``P_o``
and independently rechecks all nine admissible dwell tuples.  The paper does
not claim that its reported ``q=0.85`` minimizes contraction; the certificate
balances contraction and conditioning to tighten the continuous-time bound.

Run from the repository root with::

    python simulations/tools/verify_assumption5_dwell_family.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh, expm


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
root = Path(__file__).resolve().parents[2]
certificate = json.loads(
    (root / "verification" / "observer_dwell_family_certificate.json").read_text(
        encoding="utf-8"
    )
)
Po = np.asarray(certificate["P_o"], dtype=float)
q_report = float(certificate["q_report"])
contractions = [
    np.sqrt(float(eigvalsh(Psi.T @ Po @ Psi, Po).max())) for Psi in Psis
]
margins = [
    float(np.linalg.eigvalsh(Psi.T @ Po @ Psi - q_report**2 * Po).max())
    for Psi in Psis
]
q = max(contractions)
lam = -np.log(q_report) / Trho
assert q < q_report
assert max(margins) < 0.0
print(
    f"PUBLICATION-Po: q_worst={q:.4f}  q_report={q_report:.2f}  "
    f"lambda_e={lam:.4f}  cond(Po)={np.linalg.cond(Po):.2f}  "
    f"worst_lmi_margin={max(margins):.3e}",
    flush=True,
)
