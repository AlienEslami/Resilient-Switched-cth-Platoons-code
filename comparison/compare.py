"""Head-to-head: MSR (redundancy-based rejection) vs. this paper's per-link
reconstruction, under a growing number of attacked communication links.

Both controllers run on the SAME third-order CTH plant and the SAME redundant
look-ahead topology, so the comparison is apples-to-apples. The only difference
is how each processes the attacked neighbour packets:

  * MSR (Safeguard-style): from the per-neighbour deviation vectors it discards
    the ``F`` largest-norm ones (the "farthest from the origin") and averages the
    rest.  This is provably resilient only while the number of attacked links per
    vehicle stays below ``F`` so that a clean majority survives removal.
  * OURS: each link runs the switching augmented observer of the paper, so the
    attack on every link is reconstructed and compensated, independently of how
    many links are attacked.

The experiment sweeps ``a`` = number of attacked links per vehicle and reports
the steady tracking error and the minimum inter-vehicle distance for each method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulations.src.matrices import augmented_matrices, vehicle_matrices

# --------------------------------------------------------------------------- #
# Fixed scenario parameters (single physical mode keeps MSR in its LTI comfort
# zone; the attack-cardinality axis is what we isolate here).
# --------------------------------------------------------------------------- #
N = 7                      # followers, indexed 1..N; leader is 0
TAU = 0.5                  # powertrain lag (single mode)
H = 1.2                    # time headway
D_GAP = 5.0                # standstill gap
V0 = 20.0                  # cruise speed
GAIN = np.array([0.5, 1.0, 0.5])
WINDOW = 3                 # look-ahead redundancy: up to 3 nearest predecessors
F_REMOVE = 1               # MSR removes this many largest-norm deviations
DURATION = 40.0
STEP = 0.005
SAMPLE = 0.02
ATTACK_START = 8.0
ATTACK_RISE = 4.0
# A negative drift reports each predecessor as moving more slowly and falling
# behind.  This encourages platoon dispersion rather than a spacing collapse.
ATTACK_DRIFT_RATE = -0.5  # m/s after the smooth onset
# followers with the full redundancy WINDOW (used for the headline metric)
FULL_REDUNDANCY_FOLLOWERS = list(range(WINDOW, N + 1))

POS_PERTURB = np.array([0.0, 1.5, -1.0, 2.0, -1.5, 1.0, -0.5, 1.2])
VEL_PERTURB = np.array([0.0, 0.4, -0.3, 0.2, -0.4, 0.3, -0.2, 0.1])


def neighbours(i: int) -> list[int]:
    """Up to WINDOW nearest predecessors of follower i (look-ahead graph)."""
    return [j for j in range(i - 1, max(-1, i - 1 - WINDOW), -1) if j >= 0]


def coherent_state_attack(t: float) -> np.ndarray:
    """Kinematically consistent [position, velocity, acceleration] attack.

    The false velocity is introduced with a quintic smoothstep.  Integrating
    that velocity gives the false position, and differentiating it gives the
    false acceleration.  Hence all three packet components start continuously
    from zero and remain mutually consistent.
    """
    elapsed = t - ATTACK_START
    if elapsed <= 0.0:
        return np.zeros(3)
    if elapsed >= ATTACK_RISE:
        return np.array([
            ATTACK_DRIFT_RATE * (elapsed - 0.5 * ATTACK_RISE),
            ATTACK_DRIFT_RATE,
            0.0,
        ])

    u = elapsed / ATTACK_RISE
    smoothstep = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
    smoothstep_dot = 30.0 * u**2 * (u - 1.0) ** 2 / ATTACK_RISE
    smoothstep_integral = ATTACK_RISE * (u**6 - 3.0 * u**5 + 2.5 * u**4)
    return np.array([
        ATTACK_DRIFT_RATE * smoothstep_integral,
        ATTACK_DRIFT_RATE * smoothstep,
        ATTACK_DRIFT_RATE * smoothstep_dot,
    ])


def all_edges() -> list[tuple[int, int]]:
    return [(i, j) for i in range(1, N + 1) for j in neighbours(i)]


# --------------------------------------------------------------------------- #
# Auxiliary/observer configuration reused from the paper's certified design.
# --------------------------------------------------------------------------- #
_AUX = json.loads((ROOT / "simulations" / "configs" / "baseline.json").read_text())["auxiliary"]
A_Z = [np.asarray(m, float) for m in _AUX["A_z"]]
B_Z = [np.asarray(m, float) for m in _AUX["B_z"]]
C_Z = [np.asarray(m, float) for m in _AUX["C_z"]]
L_OBS = [np.asarray(m, float) for m in _AUX["L"]]
DWELL = np.asarray(_AUX["dwell_times"], float)
_AUG = [augmented_matrices(a, b, c) for a, b, c in zip(A_Z, B_Z, C_Z)]
A_AUG = [m[0] for m in _AUG]
C_AUG = [m[1] for m in _AUG]
B_AUG = [np.vstack([b, np.zeros((6, 3))]) for b in B_Z]


def auxiliary_mode(t: float) -> int:
    period = float(np.sum(DWELL))
    local = (t + 1e-12) % period
    return int(np.searchsorted(np.cumsum(DWELL), local, side="right")) % len(DWELL)


A_P, B_P = vehicle_matrices(TAU)
B_COL = B_P[:, 0]
EDGES = [(i, j) for i in range(1, N + 1) for j in neighbours(i)]
EDGE_INDEX = {e: k for k, e in enumerate(EDGES)}


def initial_physical() -> np.ndarray:
    phys = np.zeros((N + 1, 3))
    phys[:, 1] = V0
    desired = D_GAP + H * V0
    phys[:, 0] = -desired * np.arange(N + 1)
    phys[:, 0] += POS_PERTURB[: N + 1]
    phys[:, 1] += VEL_PERTURB[: N + 1]
    return phys


def deviation(i: int, j: int, xi: np.ndarray, xj_used: np.ndarray) -> np.ndarray:
    """Per-neighbour CTH deviation vector [spacing err, rel vel, rel accel]."""
    p = i - j
    spacing = (xj_used[0] - xi[0]) - p * (D_GAP + H * xi[1])
    return np.array([spacing, xj_used[1] - xi[1], xj_used[2] - xi[2]])


def leader_command(t: float) -> float:
    return 0.0


def attack_signals(t: float, i: int, j: int, attacked: set) -> tuple[np.ndarray, np.ndarray]:
    if (i, j) in attacked:
        # The MSR baseline has no auxiliary-output channel, so this direct
        # comparison attacks only the physical-state packet shared by both
        # methods.
        return coherent_state_attack(t), np.zeros(3)
    return np.zeros(3), np.zeros(3)


def run(method: str, attacked: set) -> dict:
    """Integrate the closed loop for one method ('msr' or 'ours') and attack count a."""
    phys = initial_physical()
    use_obs = method == "ours"
    Z = np.zeros((len(EDGES), 3))
    O = np.zeros((len(EDGES), 9))

    steps = int(round(DURATION / STEP))
    save_every = int(round(SAMPLE / STEP))
    samples = steps // save_every + 1
    times = np.empty(samples)
    phys_hist = np.empty((samples, N + 1, 3))

    def controls(t, phys, O):
        u = np.zeros(N + 1)
        for i in range(1, N + 1):
            nb = neighbours(i)
            devs = []
            for j in nb:
                xa, _ = attack_signals(t, i, j, attacked)
                x_recv = phys[j] + xa
                if use_obs:
                    x_recv = x_recv - O[EDGE_INDEX[(i, j)], 6:9]  # compensate
                devs.append((j, deviation(i, j, phys[i], x_recv)))
            if method == "msr" and len(devs) > 1:
                order = sorted(range(len(devs)), key=lambda idx: np.linalg.norm(devs[idx][1]))
                keep = order[: max(1, len(devs) - F_REMOVE)]  # drop F largest-norm
                used = [devs[idx][1] for idx in keep]
            else:
                used = [d for _, d in devs]
            dbar = np.mean(used, axis=0)
            u[i] = float(GAIN @ dbar)
        return u

    def derivative(t, phys, Z, O):
        u = controls(t, phys, O)
        phys_dot = np.zeros_like(phys)
        phys_dot[0] = A_P @ phys[0] + B_COL * leader_command(t)
        for i in range(1, N + 1):
            phys_dot[i] = A_P @ phys[i] + B_COL * u[i]
        Z_dot = np.zeros_like(Z)
        O_dot = np.zeros_like(O)
        if use_obs:
            zmode = auxiliary_mode(t)
            for (i, j), k in EDGE_INDEX.items():
                xa, ya = attack_signals(t, i, j, attacked)
                x_star = phys[j] + xa
                y_star = C_Z[zmode] @ Z[k] + ya
                Z_dot[k] = A_Z[zmode] @ Z[k] + B_Z[zmode] @ phys[j]
                innov = y_star - C_AUG[zmode] @ O[k]
                O_dot[k] = A_AUG[zmode] @ O[k] + B_AUG[zmode] @ x_star + L_OBS[zmode] @ innov
        return phys_dot, Z_dot, O_dot

    sample = 0
    for step in range(steps + 1):
        t = step * STEP
        if step % save_every == 0:
            times[sample] = t
            phys_hist[sample] = phys
            sample += 1
        if step == steps:
            break
        k1 = derivative(t, phys, Z, O)
        k2 = derivative(t + 0.5 * STEP, phys + 0.5 * STEP * k1[0], Z + 0.5 * STEP * k1[1], O + 0.5 * STEP * k1[2])
        k3 = derivative(t + 0.5 * STEP, phys + 0.5 * STEP * k2[0], Z + 0.5 * STEP * k2[1], O + 0.5 * STEP * k2[2])
        k4 = derivative(t + STEP, phys + STEP * k3[0], Z + STEP * k3[1], O + STEP * k3[2])
        phys = phys + (STEP / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Z = Z + (STEP / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        O = O + (STEP / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

    # metrics
    ref = np.empty((samples, N, 3))
    for idx in range(N):
        ref[:, idx, 0] = phys_hist[:, 0, 0] - (idx + 1) * (D_GAP + H * V0)
        ref[:, idx, 1] = V0
        ref[:, idx, 2] = 0.0
    err = phys_hist[:, 1:, :] - ref
    err_norm = np.linalg.norm(err, axis=2)  # (samples, N)
    gaps = phys_hist[:, :-1, 0] - phys_hist[:, 1:, 0]
    tail = times >= (DURATION - 5.0)
    tail_err_by_follower = np.max(err_norm[tail], axis=0)  # (N,)
    return {
        "times": times,
        "err_norm": err_norm,
        "gaps": gaps,
        "tail_max_err": float(np.max(err_norm[tail])),
        "tail_err_by_follower": tail_err_by_follower.tolist(),
        "final_max_err": float(np.max(err_norm[-1])),
        "min_gap": float(np.min(gaps)),
        "final_max_gap": float(np.max(gaps[-1])),
    }


SCENARIOS = {
    # one single attacked channel at the platoon head (in-degree-1 vehicle)
    "head_single": {(1, 0)},
    # one single attacked channel at a fully-redundant interior vehicle
    "interior_single": {(5, 4)},
    # every V2V link attacked at once (cardinality-independence of ours)
    "all_links": set(all_edges()),
}


def main() -> None:
    summary = {"config": {
        "followers": N, "tau": TAU, "headway": H, "window": WINDOW,
        "F_remove": F_REMOVE, "attack_start": ATTACK_START,
        "attack_rise": ATTACK_RISE, "attack_drift_rate": ATTACK_DRIFT_RATE,
        "auxiliary_output_attack": [0.0, 0.0, 0.0],
        "note": "MSR runs on the redundant look-ahead graph it is designed for; "
                "in-degrees are 1,2,3,3,3,3,3 for followers 1..7.",
    }, "results": {}}
    series = {}
    for name, attacked in SCENARIOS.items():
        summary["results"][name] = {"attacked_links": sorted(str(e) for e in attacked)}
        print(f"\n=== scenario '{name}'  ({len(attacked)} attacked link(s)) ===")
        for method in ("msr", "ours"):
            r = run(method, attacked)
            series[(method, name)] = r
            summary["results"][name][method] = {
                "tail_max_err": r["tail_max_err"],
                "tail_err_by_follower": r["tail_err_by_follower"],
                "final_max_err": r["final_max_err"],
                "min_intervehicle_distance": r["min_gap"],
                "final_max_intervehicle_distance": r["final_max_gap"],
            }
            if r["min_gap"] <= 0.0:
                verdict = "collision"
            elif r["tail_max_err"] > 5.0:
                verdict = "large deviation"
            else:
                verdict = "bounded"
            print(f"  {method:>4}: steady max err={r['tail_max_err']:9.4f}  "
                  f"final max gap={r['final_max_gap']:8.3f} m  [{verdict}]")

    out = ROOT / "comparison" / "results" / "single_channel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    _plots(series)


def _plots(series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ROOT / "comparison" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    # time series of max tracking error for the two headline scenarios
    for name in ("head_single", "all_links"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, style in (("msr", "-"), ("ours", "--")):
            r = series[(method, name)]
            ax.plot(r["times"], np.max(r["err_norm"], axis=1), style,
                    label="MSR (reject)" if method == "msr" else "Ours (reconstruct)")
        ax.axvline(ATTACK_START, ls=":", color="gray")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("max tracking-error norm")
        title = "One attacked channel at platoon head" if name == "head_single" \
            else "All V2V links attacked"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / f"{name}.png", dpi=150)
        fig.savefig(figs / f"{name}.pdf")

    # per-follower steady error for the head-single case
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, N + 1)
    for method, mk in (("msr", "o-"), ("ours", "s-")):
        r = series[(method, "head_single")]
        ax.semilogy(idx, np.maximum(r["tail_err_by_follower"], 1e-6), mk,
                    label="MSR (reject)" if method == "msr" else "Ours (reconstruct)")
    ax.set_xlabel("follower index")
    ax.set_ylabel("steady tracking-error norm")
    ax.set_title("Propagation of a single head attack")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "head_single_by_follower.png", dpi=150)
    fig.savefig(figs / "head_single_by_follower.pdf")
    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
