# Baseline numerical certificates

All matrix inequalities were solved with MOSEK and then checked using
independent symmetric-eigenvalue calculations.

## Baseline

- Followers: 7
- Powertrain lags: [0.3, 0.5, 0.7] s
- Headway: 1.2 s
- Controller gains: [[0.47, 0.99, 0.44], [0.5, 1.0, 0.5], [0.53, 1.01, 0.7]]

## Switching observer

- Admissible dwell times: [0.3, 0.35, 0.4] s
- Worst generalized contraction: 0.84002534
- Worst dwell tuple: [0.4, 0.3] s
- Family LMI worst max eigenvalue: -2.11373007e-02
- Family observability-Gramian minimum eigenvalue: 6.61077335e-04
- Observability beta (computed): 4.49447643e-03
- Observability beta (reported): 4.44953167e-03
- Monodromy spectral radius: 0.53283306
- Lifted q (reported): 0.8500
- Lifted LMI max eigenvalue: -1.17645962e-01
- lambda_e: 0.20314866 1/s
- K_e (direct-transition upper bound): 3.31000000
- M_e: 92.29828708
- c_e: 454.33864162 s

## Platoon UUB

- lambda_c: 0.2500 1/s
- mu_c: 1.0
- ADT threshold: 0.0 s
- m_c: 1.00000000
- kappa_c: 89.85504900
- Lyapunov LMI max eigenvalue: 2.64778199e-11
- Radius coefficient multiplying Rbar: 5750.72313646

The common physical Lyapunov matrix makes the ADT threshold zero for
this predecessor-chain baseline. An ADT-violation ablation is therefore
not supported by this particular certified design.

## String performance

- alpha_0: 0.47000000 1/s
- Nominal LMI worst numerical eigenvalue: 1.69996385e-09
- gamma_r (reported): 1.244
- Residual LMI max eigenvalue: -3.07101531e-02
- Joint-storage gamma_r (reported): 1.436
- Joint LMI max eigenvalue: 6.51728624e-11
- Exact frozen-mode headway boundary: 0.84186538 s

The tiny positive nominal-LMI eigenvalue is solver-scale roundoff at
the exact unit DC gain. The independent frequency inequalities are
nonnegative in every frozen mode.
