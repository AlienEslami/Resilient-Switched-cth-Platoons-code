"""Reproduce the tightened partial-cycle observer bound used in the paper.

The auxiliary dwell alphabet is {0.30, 0.35, 0.40} s for each of two modes.
For any partial interval contained in one cycle, the transition has the form
    exp(Ae_2 v) exp(Ae_1 u),  0 <= u,v <= 0.40,
including single-mode intervals by setting u=0 or v=0.

A uniform grid is enclosed using
 ||F(u,v)||_2 <= ||F(u0,v0)||_2 exp((||Ae_1||_2+||Ae_2||_2) h/2)
for the nearest grid point (u0,v0), where h is the grid spacing.  Thus the
reported K_e is an upper bound, not merely the largest sampled value.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh, expm

HERE = Path(__file__).resolve().parent
CERT_PATH = HERE / "observer_dwell_family_certificate.json"

# Baseline auxiliary system and observer gains.
Az = [
    np.diag([-1.0, -1.3, -1.6]),
    np.diag([-2.0, -1.0, -0.7]),
]
Bz = [
    np.diag([1.0, 0.8, 1.2]),
    np.diag([0.5, 1.4, 0.9]),
]
Cz = [
    np.eye(3),
    np.diag([1.4, 0.7, 1.2]),
]
L = [
    np.array([
        [1.386, 0.0, 0.0], [0.0, -0.606, 0.0], [0.0, 0.0, -0.873],
        [2.56, 0.0, 0.0], [0.0, 2.187, 0.0], [0.0, 0.0, 1.33],
        [-1.805, 0.0, 0.0], [0.0, -2.373, 0.0], [0.0, 0.0, 2.706],
    ]),
    np.array([
        [0.62, 0.0, 0.0], [0.0, -0.349, 0.0], [0.0, 0.0, -0.742],
        [-2.444, 0.0, 0.0], [0.0, 1.873, 0.0], [0.0, 0.0, 3.193],
        [-2.3, 0.0, 0.0], [0.0, 2.796, 0.0], [0.0, 0.0, -1.189],
    ]),
]


def augmented(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.zeros((3, 3))
    A = np.block([[a, z, -b], [z, z, z], [z, z, z]])
    C = np.block([c, np.eye(3), z])
    return A, C


Ae = []
for a, b, c, gain in zip(Az, Bz, Cz, L):
    A, C = augmented(a, b, c)
    Ae.append(A - gain @ C)

with CERT_PATH.open("r", encoding="utf-8") as handle:
    cert = json.load(handle)
P = np.asarray(cert["P_o"], dtype=float)
q_report = float(cert["q_report"])
dwell = [0.30, 0.35, 0.40]

# Recheck all nine lifted inequalities.
worst_q = 0.0
worst_tuple = None
for d1 in dwell:
    for d2 in dwell:
        psi = expm(Ae[1] * d2) @ expm(Ae[0] * d1)
        q = math.sqrt(float(eigvalsh(psi.T @ P @ psi, P).max()))
        if q > worst_q:
            worst_q = q
            worst_tuple = (d1, d2)
assert worst_q < q_report

# Certified enclosure of all partial-cycle transitions.  Because the largest
# admissible dwell in either mode is 0.40 s, the union of all nine dwell-family
# partial-transition domains is the square [0,0.40]^2.
N = 501
umax = 0.40
u = np.linspace(0.0, umax, N)
v = np.linspace(0.0, umax, N)
h = float(u[1] - u[0])
E1 = [expm(Ae[0] * x) for x in u]
E2 = np.stack([expm(Ae[1] * x) for x in v], axis=0)

grid_max = 0.0
argmax = (0.0, 0.0)
for ui, e1 in zip(u, E1):
    batch = E2 @ e1
    smax = np.linalg.svd(batch, compute_uv=False)[:, 0]
    j = int(np.argmax(smax))
    if float(smax[j]) > grid_max:
        grid_max = float(smax[j])
        argmax = (float(ui), float(v[j]))

lip_factor = math.exp((np.linalg.norm(Ae[0], 2) + np.linalg.norm(Ae[1], 2)) * h / 2.0)
K_upper = grid_max * lip_factor
K_report = 3.31
assert K_upper < K_report

cond_P = float(np.linalg.cond(P, 2))
T_rho = 0.80
lambda_e = -math.log(q_report) / T_rho
M_e = K_report**2 * math.sqrt(cond_P) * q_report**(-2.0)
c_e = M_e / lambda_e  # ||D||_2 = 1.

lines = [
    f"worst generalized contraction: {worst_q:.12f} at {worst_tuple}",
    f"grid maximum ||Phi||_2:       {grid_max:.12f} at u,v={argmax}",
    f"grid spacing:                 {h:.7f} s",
    f"Lipschitz enclosure upper:    {K_upper:.12f}",
    f"reported K_e upper bound:     {K_report:.2f}",
    f"cond_2(P_o):                  {cond_P:.8f}",
    f"lambda_e:                     {lambda_e:.10f} 1/s",
    f"M_e:                          {M_e:.8f}",
    f"c_e:                          {c_e:.8f} s",
]
output = "\n".join(lines) + "\n"
print(output, end="")
(HERE / "tight_observer_bound_output.txt").write_text(output, encoding="utf-8")
