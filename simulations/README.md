# Simulations

This directory contains the reproducible numerical study for the paper.

Stages:

1. Synthesize and certify the switching-observable auxiliary observer.
2. Solve the switched platoon Lyapunov inequalities and verify the ADT margin.
3. Solve the nominal and residual-channel string-performance LMIs.
4. Run nominal, attack, leader-link, string-propagation, and ablation scenarios.
5. Generate publication figures and numerical certificate tables.

Generated figures are ignored by Git. The scripts, tests, and the exact
configuration files used to produce them are committed.

## Current milestone

The MOSEK/CVXPY semidefinite environment is operational. Run
**tools/check_solver.py** from the repository root to verify the license and
solve a normalized common-Lyapunov LMI.

The baseline theorem certificates are synthesized with:

    .\.venv\Scripts\python.exe simulations\src\certificates.py

This produces a machine-readable JSON certificate and a concise Markdown
summary in **simulations/certificates/**.

The three physical powertrain modes use the mode-dependent gains
\([0.47,0.99,0.44]^\top\), \([0.50,1.00,0.50]^\top\), and
\([0.53,1.01,0.70]^\top\), respectively. Each vector is one mode's complete
gain \(k_p=[k_{1,p},k_{2,p},k_{3,p}]^\top\); the vectors are not columns that
collect one fixed gain component across modes. Independently recheck the
stored common plant and string certificates for these gains with:

    .\.venv\Scripts\python.exe simulations\tools\verify_mode_dependent_gains.py

The resulting per-mode margins are written to
**simulations/certificates/mode_dependent_gain_verification.json**.

Verify the common Assumption 5 contraction certificate over the nine
admissible auxiliary-mode dwell tuples with:

    .\.venv\Scripts\python.exe simulations\tools\verify_assumption5_dwell_family.py

The deterministic trajectory regression suite is run with:

    .\.venv\Scripts\python.exe -m pytest simulations\tests -q

Generate the paper's nominal baseline and the low-frequency and ramp-attack
studies with:

    .\.venv\Scripts\python.exe -m simulations.src.run_scenarios

Every paper scenario uses the reproducible low-power communication-noise model
reported in **configs/baseline.json**. Each attack is run with both the
uncompensated controller and the proposed resilient controller. The script
generates attack-reconstruction, estimation-error, and platoon-state
comparisons. It also attacks only the middle link $(4,3)$ in a dedicated
low-frequency experiment and reports the downstream attack-induced
acceleration after subtracting a matched no-attack trajectory with the
identical noise realization.

The trajectory schedule is right-continuous. All vehicles synchronously cycle
through powertrain lags \(0.3\), \(0.5\), and \(0.7\) s, dwelling for \(2\) s
in each mode (a \(6\) s physical cycle). All auxiliary systems and their
observers synchronously alternate between modes 1 and 2 with \(0.4\) s in
each mode (a \(0.8\) s auxiliary cycle). The physical and auxiliary signals
are separately specified. Because \(2/0.4=5\), every physical transition
coincides with one auxiliary transition in this deterministic baseline, while
the auxiliary system also switches four additional times during each
physical-mode dwell.

The generated figures and metrics are written to **simulations/figures/** and
**simulations/results/**. Figures are not tracked in this repository; run the
scripts above to regenerate them. The baseline metrics and the numerical
certificates are included, so the reported numbers can be checked without
rerunning the solver. The exact model, noise realization, configurations, and
plotting scripts are version controlled.

Generate the leader-disturbance string comparison and resilient-controller
ablation with:

    .\.venv\Scripts\python.exe -m simulations.src.run_string_and_ablations

Generate the headway comparison between the frozen-mode frequency test, common
switched bounded-real LMI, and downstream acceleration energy with:

    .\.venv\Scripts\python.exe -m simulations.src.run_headway_sweep
