# MSR vs. reconstruction: attack-cardinality comparison

Head-to-head between a **redundancy-based** resilient controller (mean-subsequence
-reduced / "Safeguard"-style outlier rejection) and **this paper's per-link
reconstruction** controller, run on the **same** third-order CTH plant and the
**same** look-ahead topology.

## What is compared

Both controllers use the identical nominal CTH law
`u_i = k^T * mean_j (deviation_ij)`. They differ only in how the attacked
neighbour packets are processed:

- **MSR**: from the per-neighbour deviation vectors, discard the `F` largest-norm
  ones ("farthest from the origin") and average the rest. Provably resilient only
  while the number of attacked links incident to a vehicle stays `<= F`, so that a
  clean majority survives removal.
- **Ours**: each link runs the switching augmented observer of the paper
  (`simulations/src` matrices reused), reconstructs the attack, and compensates it.
  No honest-neighbour redundancy is used.

MSR is given the redundant look-ahead graph it is designed for (in-degrees
1,2,3,3,3,3,3 for followers 1..7 with `WINDOW=3`, `F=1`).

Every attacked physical-state packet receives the kinematically consistent
signal `[q_a, q_a_dot, q_a_ddot]`.  Its false velocity changes smoothly from
zero to `-0.5 m/s` over 4 s using a quintic smoothstep; false position is its
integral and false acceleration is its derivative.  The onset therefore has no
step in any state component.  The negative drift makes a predecessor appear to
move more slowly and fall behind, inducing the uncompensated controller to
increase the actual gap and disperse the platoon.  The auxiliary-output attack
is zero in this comparison because the MSR baseline has no corresponding
channel.

## Scenarios (`compare.py`)

| Scenario | Attacked links | Point |
| --- | --- | --- |
| `head_single` | one link `(1,0)` at the platoon head | head vehicle has no redundant honest neighbour |
| `interior_single` | one link `(5,4)` at a redundant vehicle | fairness check: MSR *does* reject where redundancy exists |
| `all_links` | every V2V link | our threat model; defeats any redundancy-based scheme |

## Run

    .venv/Scripts/python.exe comparison/compare.py

Outputs `comparison/results/single_channel.json` and figures in
`comparison/figures/`.

## Honesty notes

- MSR is not strawmanned: it runs on its intended redundant graph and, per the
  `interior_single` scenario, correctly rejects a single attack on a vehicle that
  has a clean majority.
- The failures arise where MSR's structural requirement is violated: a
  low-redundancy head vehicle (`head_single`), or all links attacked at once
  (`all_links`). The latter is topology-independent and is the fundamental point.
