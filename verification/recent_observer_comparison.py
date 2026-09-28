#!/usr/bin/env python3
"""Comparison with the 1-D specialization of Lin et al. (IEEE T-ITS 2025).

This script evaluates two communication-FDI estimation scenarios on a single
predecessor link:
  (i) a bounded multiplicative attack satisfying Lin et al.'s bounded-attack
      setting on the simulated horizon, and
  (ii) a bounded-rate but unbounded position attack obtained from a constant
       multiplicative gain during constant-speed motion.

The Lin baseline is deliberately favorable: its 1-D observer is given the true
predecessor acceleration and all six reported gain pairs are swept, with the
best bounded-case result retained. The proposed observer uses the matrices and
gains reported in the paper/repository.

Outputs:
  recent_observer_comparison_metrics.json
  recent_observer_comparison.pdf
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = Path(__file__).with_name("recent_observer_comparison_metrics.json")
OUT_PDF = Path(__file__).with_name("recent_observer_comparison.pdf")

# Defender-designed auxiliary system and observer gains from baseline.json.
AZ = [
    np.diag([-1.0, -1.3, -1.6]),
    np.diag([-2.0, -1.0, -0.7]),
]
BZ = [
    np.diag([1.0, 0.8, 1.2]),
    np.diag([0.5, 1.4, 0.9]),
]
CZ = [
    np.eye(3),
    np.diag([1.4, 0.7, 1.2]),
]
L = [
    np.array([
        [1.386, 0, 0], [0, -0.606, 0], [0, 0, -0.873],
        [2.56, 0, 0], [0, 2.187, 0], [0, 0, 1.33],
        [-1.805, 0, 0], [0, -2.373, 0], [0, 0, 2.706],
    ], dtype=float),
    np.array([
        [0.62, 0, 0], [0, -0.349, 0], [0, 0, -0.742],
        [-2.444, 0, 0], [0, 1.873, 0], [0, 0, 3.193],
        [-2.3, 0, 0], [0, 2.796, 0], [0, 0, -1.189],
    ], dtype=float),
]

AUG_A = []
AUG_B = []
AUG_C = []
for az, bz, cz in zip(AZ, BZ, CZ):
    z = np.zeros((3, 3))
    AUG_A.append(np.block([[az, z, -bz], [z, z, z], [z, z, z]]))
    AUG_B.append(np.vstack([bz, np.zeros((6, 3))]))
    AUG_C.append(np.hstack([cz, np.eye(3), np.zeros((3, 3))]))

AUX_DWELL = 0.4
# Lin et al., T-ITS 2025, reported observer gains for vehicle 1:
# k_1=[k_px,k_py,k_v,k_theta,k_omega]^T=[2,5.2,15,2,0.5]^T.
KPX_LIN = 2.0
KV_LIN = 15.0
LIN_REPORTED_GAINS = [
    (1, 2.0, 15.0), (2, 5.0, 40.0), (3, 3.0, 20.0),
    (4, 3.0, 10.0), (5, 2.0, 10.0), (6, 2.0, 10.0),
]


def smoothstep5(t: float, start: float, rise: float) -> tuple[float, float]:
    if t <= start:
        return 0.0, 0.0
    if t >= start + rise:
        return 1.0, 0.0
    u = (t - start) / rise
    s = 6*u**5 - 15*u**4 + 10*u**3
    ds = (30*u**2*(u-1)**2) / rise
    return float(s), float(ds)


def scenario_bounded(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bounded kinematically consistent trajectory and multiplicative FDI."""
    w = 0.20
    # p,v,a are dynamically consistent and bounded.
    x = np.array([
        10.0 + 2.0*np.sin(w*t),
        2.0*w*np.cos(w*t),
        -2.0*w*w*np.sin(w*t),
    ])
    jerk = -2.0*w**3*np.cos(w*t)
    s, _ = smoothstep5(t, 5.0, 2.0)
    phi = s * (0.020 + 0.010*np.sin(0.15*(t-5.0)))
    xa = phi * x
    xbar = x.copy()  # v_c=0 for this bounded benchmark.
    return x, xbar, xa, jerk


def scenario_unbounded(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Constant-speed trajectory; constant multiplicative FDI gives ramp position corruption."""
    vc = 20.0
    x = np.array([vc*t, vc, 0.0])
    phi = 0.05
    xa = phi * x
    # Use the paper's moving reference r_c=[v_c t,v_c,0]^T, so xbar=0.
    xbar = np.zeros(3)
    return x, xbar, xa, 0.0


def aux_mode(t: float) -> int:
    return int(math.floor((t + 1e-12) / AUX_DWELL)) % 2


def rhs(t: float, state: np.ndarray, scenario) -> np.ndarray:
    # state: true z(3), proposed hat chi(9), Lin [phat,vhat](2)
    ztrue = state[:3]
    hchi = state[3:12]
    hlin = state[12:14]
    x, xbar, xa, _ = scenario(t)
    mode = aux_mode(t)
    az, bz, cz = AZ[mode], BZ[mode], CZ[mode]
    aa, bb, cc, ll = AUG_A[mode], AUG_B[mode], AUG_C[mode], L[mode]

    # Proposed method uses the moving-frame packet xbar*=xbar+xa.
    # No attack on auxiliary-output packet, isolating the state-packet reconstruction channel.
    xstar = x + xa
    xbarstar = xbar + xa
    yzstar = cz @ ztrue

    dz = az @ ztrue + bz @ xbar
    innovation = yzstar - cc @ hchi
    dhchi = aa @ hchi + bb @ xbarstar + ll @ innovation

    pstar, vstar = xstar[0], xstar[1]
    phat, vhat = hlin
    # 1-D specialization of Lin et al. (their Remark 3): Luenberger structure.
    # Give it exact acceleration a(t) as known input, eliminating plant-model uncertainty.
    dphat = vhat + KPX_LIN * (pstar - phat)
    dvhat = x[2] + KV_LIN * (vstar - vhat)
    return np.concatenate([dz, dhchi, [dphat, dvhat]])


def rk4_step(t, y, dt, scenario):
    k1 = rhs(t, y, scenario)
    k2 = rhs(t + 0.5*dt, y + 0.5*dt*k1, scenario)
    k3 = rhs(t + 0.5*dt, y + 0.5*dt*k2, scenario)
    k4 = rhs(t + dt, y + dt*k3, scenario)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def run_case(name: str, scenario, duration: float, dt: float = 0.002):
    n = int(round(duration/dt)) + 1
    t = np.linspace(0.0, duration, n)
    state = np.zeros(14)
    # Favor both estimators by exact initial state before attacks.
    x0, _, _, _ = scenario(0.0)
    state[12:14] = x0[:2]

    xhist = np.zeros((n, 3))
    xahist = np.zeros((n, 3))
    prop_corr = np.zeros((n, 3))
    lin_corr = np.zeros((n, 2))
    prop_xahat = np.zeros((n, 3))
    lin_xahat = np.zeros((n, 2))

    for k, tk in enumerate(t):
        x, xbar, xa, _ = scenario(float(tk))
        xhist[k] = x
        xahist[k] = xa
        hchi = state[3:12]
        hxa = hchi[6:9]
        xstar = x + xa
        prop_xahat[k] = hxa
        prop_corr[k] = xstar - hxa
        lin_corr[k] = state[12:14]
        lin_xahat[k] = xstar[:2] - state[12:14]
        if k < n - 1:
            state = rk4_step(float(tk), state, dt, scenario)

    eprop = prop_corr[:, :2] - xhist[:, :2]
    elin = lin_corr - xhist[:, :2]
    ea_prop = prop_xahat[:, :2] - xahist[:, :2]
    ea_lin = lin_xahat - xahist[:, :2]

    eval_mask = t >= 10.0
    tail_mask = t >= max(10.0, duration - 20.0)

    def metrics(e, ea):
        norm = np.linalg.norm(e, axis=1)
        anorm = np.linalg.norm(ea, axis=1)
        return {
            "state_rmse_eval": float(np.sqrt(np.mean(norm[eval_mask]**2))),
            "state_max_eval": float(np.max(norm[eval_mask])),
            "state_final_norm": float(norm[-1]),
            "position_rmse_eval_m": float(np.sqrt(np.mean(e[eval_mask,0]**2))),
            "velocity_rmse_eval_m_s": float(np.sqrt(np.mean(e[eval_mask,1]**2))),
            "position_final_abs_m": float(abs(e[-1,0])),
            "position_tail_rmse_m": float(np.sqrt(np.mean(e[tail_mask,0]**2))),
            "attack_rmse_eval": float(np.sqrt(np.mean(anorm[eval_mask]**2))),
            "attack_final_norm": float(anorm[-1]),
        }

    # Estimate final linear growth of position reconstruction error.
    def slope(e):
        tt = t[tail_mask]
        ee = e[tail_mask,0]
        A = np.column_stack([tt, np.ones_like(tt)])
        return float(np.linalg.lstsq(A, ee, rcond=None)[0][0])

    mprop = metrics(eprop, ea_prop)
    mlin = metrics(elin, ea_lin)
    mprop["position_error_tail_slope"] = slope(eprop)
    mlin["position_error_tail_slope"] = slope(elin)

    # Attack rate numerical estimate for proposed threat model.
    dxa = np.gradient(xahist, t, axis=0)
    max_rate = float(np.max(np.linalg.norm(dxa, axis=1)))
    max_amp = float(np.max(np.linalg.norm(xahist, axis=1)))

    return {
        "name": name,
        "time": t,
        "x": xhist,
        "xa": xahist,
        "eprop": eprop,
        "elin": elin,
        "metrics": {
            "proposed": mprop,
            "lin_1d": mlin,
            "attack_max_amplitude_over_horizon": max_amp,
            "attack_max_rate_over_horizon": max_rate,
        }
    }


def main():
    global KPX_LIN, KV_LIN
    # Sweep all six reported (k_px,k_v) pairs and use the best bounded-case
    # position RMSE. This avoids selecting a weak baseline gain.
    sweep = []
    best = None
    for vehicle, kp, kv in LIN_REPORTED_GAINS:
        KPX_LIN, KV_LIN = kp, kv
        case = run_case("bounded", scenario_bounded, 60.0, dt=0.01)
        rmse = case["metrics"]["lin_1d"]["position_rmse_eval_m"]
        rec = {"vehicle": vehicle, "k_px": kp, "k_v": kv, "bounded_position_rmse_m": rmse}
        sweep.append(rec)
        if best is None or rmse < best[0]:
            best = (rmse, vehicle, kp, kv)

    _, best_vehicle, KPX_LIN, KV_LIN = best
    bounded = run_case("bounded", scenario_bounded, 60.0)
    unbounded = run_case("unbounded", scenario_unbounded, 100.0)

    # One-column publication figure with embedded, non-Type-3 fonts.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.9), sharex=False)
    axes[0].plot(
        bounded["time"], bounded["eprop"][:,0],
        color="#1f77b4", linestyle="-", label="Proposed"
    )
    axes[0].plot(
        bounded["time"], bounded["elin"][:,0],
        color="#ff7f0e", linestyle="--", label="Lin et al."
    )
    axes[0].axvline(5.0, linewidth=0.6, linestyle="--")
    axes[0].set_ylabel("position error (m)")
    axes[0].text(0.02, 0.08, "(a) bounded FDI", transform=axes[0].transAxes, va="bottom")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    ep = np.maximum(np.abs(unbounded["eprop"][:,0]), 1e-3)
    el = np.maximum(np.abs(unbounded["elin"][:,0]), 1e-3)
    axes[1].semilogy(
        unbounded["time"], ep, color="#1f77b4", linestyle="-"
    )
    axes[1].semilogy(
        unbounded["time"], el, color="#ff7f0e", linestyle="--"
    )
    axes[1].set_ylabel("|position error| (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].text(0.02, 0.08, "(b) bounded-rate, unbounded FDI", transform=axes[1].transAxes, va="bottom")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout(pad=0.6)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "baseline": {
            "citation": "M.-F. Lin et al., IEEE T-ITS, vol. 26, no. 12, 2025, DOI 10.1109/TITS.2025.3611976",
            "specialization": "1-D specialization of Eq. (10); exact predecessor acceleration supplied and all six reported (k_px,k_v) gain pairs swept",
            "selected_reported_gain": {"vehicle": best_vehicle, "k_px": KPX_LIN, "k_v": KV_LIN},
            "gain_sweep": sweep,
            "important_scope_note": "The unbounded-ramp case violates Lin et al.'s Assumption 1 that the injected false data and its derivative are bounded; it is an out-of-model capability stress test, not an in-assumption superiority claim."
        },
        "bounded": bounded["metrics"],
        "unbounded": unbounded["metrics"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_JSON}")

if __name__ == "__main__":
    main()
