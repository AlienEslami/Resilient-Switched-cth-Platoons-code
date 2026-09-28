"""Generate the single publication figure used in the paper's comparison
subsection: all V2V links attacked, MSR (redundancy-based rejection) vs. the
proposed per-link reconstruction. Writes into simulations/figures/ so the paper's
\\graphicspath picks it up, and prints the exact scalars quoted in the text.
"""

from __future__ import annotations

import numpy as np

from compare import ROOT, N, run, all_edges, ATTACK_START, D_GAP, H, V0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams.update(
        {
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
        }
    )
    attacked = set(all_edges())
    msr = run("msr", attacked)
    ours = run("ours", attacked)

    equilibrium_gap = D_GAP + H * V0  # 29 m

    def max_err(r):
        return np.max(r["err_norm"], axis=1)

    def max_gap(r):
        return np.max(r["gaps"], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 4.0), sharex=True)

    ax1.semilogy(msr["times"], np.maximum(max_err(msr), 1e-4), "-", color="C3",
                 label="MSR rejection")
    ax1.semilogy(ours["times"], np.maximum(max_err(ours), 1e-4), "--", color="C0",
                 label="Proposed")
    ax1.axvline(ATTACK_START, ls=":", color="gray", lw=1)
    ax1.set_ylabel(r"$\max_i\,\|e_i(t)\|$")
    ax1.legend(loc="center right")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(msr["times"], max_gap(msr), "-", color="C3", label="MSR rejection")
    ax2.plot(ours["times"], max_gap(ours), "--", color="C0", label="Proposed")
    ax2.axhline(equilibrium_gap, ls="-.", color="gray", lw=1, label="equilibrium gap")
    ax2.axvline(ATTACK_START, ls=":", color="gray", lw=1)
    ax2.set_ylabel("max. spacing (m)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.25, h_pad=0.2)
    out = ROOT / "simulations" / "figures" / "comparison_all_links"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    separate = ROOT / "simulations" / "figures" / "separate"
    separate.mkdir(parents=True, exist_ok=True)
    for axis, suffix in [
        (ax1, "tracking_error"),
        (ax2, "maximum_spacing"),
    ]:
        original_xlabel = axis.get_xlabel()
        axis.set_xlabel("time (s)")
        temporary_legend = None
        if axis.get_legend() is None:
            temporary_legend = axis.legend(loc="best")
        fig.canvas.draw()
        bbox = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted()
        )
        bbox = bbox.expanded(1.02, 1.02)
        panel = separate / f"comparison_all_links_{suffix}"
        fig.savefig(f"{panel}.pdf", bbox_inches=bbox, pad_inches=0.01)
        fig.savefig(
            f"{panel}.png",
            bbox_inches=bbox,
            pad_inches=0.01,
            dpi=150,
        )
        if temporary_legend is not None:
            temporary_legend.remove()
        axis.set_xlabel(original_xlabel)
    print(f"Wrote {out}.pdf/.png")

    # exact scalars for the paper text
    print("\n--- numbers for the text ---")
    print(f"followers N = {N}, equilibrium gap = {equilibrium_gap:.0f} m")
    print(f"MSR : steady max err = {msr['tail_max_err']:.1f}, final max spacing = {msr['final_max_gap']:.1f} m")
    print(f"Ours: steady max err = {ours['tail_max_err']:.3f}, final max spacing = {ours['final_max_gap']:.2f} m")
    print("MSR per-follower steady err :", [round(x, 1) for x in msr["tail_err_by_follower"]])
    print("Ours per-follower steady err:", [round(x, 3) for x in ours["tail_err_by_follower"]])


if __name__ == "__main__":
    main()
