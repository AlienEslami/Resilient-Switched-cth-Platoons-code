# Resilient Switched CTH Platoons — Simulation Code

Reproducible simulation and certificate-synthesis code for a resilient
constant-time-headway (CTH) vehicle platoon with:

- switched, mode-dependent powertrain dynamics;
- false-data-injection attacks on V2V state and auxiliary-output channels;
- a defender-designed switching-observable auxiliary system;
- amplitude-independent attack-estimation bounds based on attack rates;
- nominal acceleration string stability and bounded-power residual analysis.

This repository contains the code, configurations, and numerical certificates
only. The manuscript is maintained separately.

## Repository layout

- **simulations/src/** — plant model, observer, certificate synthesis, and the
  scenario/sweep drivers.
- **simulations/configs/** — the exact parameter files used for every reported
  result (`baseline.json`).
- **simulations/certificates/** — synthesized numerical certificates in JSON and
  Markdown.
- **simulations/results/** — baseline scenario metrics.
- **simulations/tests/** — deterministic trajectory regression suite.
- **simulations/tools/** — solver check and independent certificate verifiers.
- **comparison/** — head-to-head comparison against a redundancy-based (MSR)
  resilient controller.
- **verification/** — the nine-tuple observer certificate, tightened
  partial-cycle bound, nontrivial ADT stress certificate, exact
  low-frequency rate calculation, and recent-observer comparison.

Generated figures are not tracked; the scripts regenerate them into
`simulations/figures/` and `comparison/figures/`.

## Setup

Create a virtual environment and install the pinned numerical stack:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

The semidefinite programs are solved through CVXPY with MOSEK. MOSEK requires a
valid license at **%USERPROFILE%\mosek\mosek.lic** (free academic licenses are
available from MOSEK). Verify the complete SDP path with:

    .\.venv\Scripts\python.exe simulations\tools\check_solver.py

The stored certificates under `simulations/certificates/` and the metrics under
`simulations/results/` are committed, so the reported numbers can be inspected
without a solver license.

## Reproducing the results

Run all commands from the repository root. See
[simulations/README.md](simulations/README.md) for the full description of each
stage, the switching schedule, and the noise model.

Synthesize the baseline theorem certificates:

    .\.venv\Scripts\python.exe simulations\src\certificates.py

Run the regression suite:

    .\.venv\Scripts\python.exe -m pytest simulations\tests -q

Run the nominal baseline and the low-frequency and ramp-attack studies:

    .\.venv\Scripts\python.exe -m simulations.src.run_scenarios

Run the leader-disturbance string comparison and controller ablation:

    .\.venv\Scripts\python.exe -m simulations.src.run_string_and_ablations

Run the headway sweep:

    .\.venv\Scripts\python.exe -m simulations.src.run_headway_sweep

Run the MSR comparison:

    .\.venv\Scripts\python.exe comparison\compare.py
    .\.venv\Scripts\python.exe comparison\make_paper_figure.py

Verify the auxiliary dwell family and tightened observer constants:

    .\.venv\Scripts\python.exe verification\tight_observer_bound.py
    .\.venv\Scripts\python.exe verification\low_frequency_tail_bound.py

Run the comparison with the 1-D specialization of the recent observer:

    .\.venv\Scripts\python.exe verification\recent_observer_comparison.py

## Reproducibility policy

Every reported numerical result and figure is generated from the committed
configuration files and scripts. Generated figure files are not tracked, but
the model, noise realization, parameters, stored metrics, and certificate
matrices are fixed in version control so the results are auditable.
