# Baseline numerical certificates

All matrix inequalities were solved with MOSEK and then checked using
independent symmetric-eigenvalue calculations.

## Baseline

- Followers: 7
- Powertrain lags: [0.3, 0.5, 0.7] s
- Headway: 1.2 s
- Controller gains: [[0.47, 0.99, 0.44], [0.50, 1.00, 0.50], [0.53, 1.01, 0.70]]

## Switching observer

- Observability beta (computed): 4.49447643e-03
- Observability beta (reported): 4.44953167e-03
- Monodromy spectral radius: 0.53283306
- Lifted q (reported): 0.8100
- Lifted LMI max eigenvalue: -2.07052247e-02
- cond_2(P_o): 35.63819610
- lambda_e: 0.26340129 1/s
- K_e (log-norm bound): 8.40260075
- M_e: 642.41441215
- c_e: 2438.91901303 s

## Platoon UUB

- lambda_c: 0.2500 1/s
- mu_c: 1.0
- ADT threshold: 0.0 s
- xi_c: 0.1250 1/s
- m_c: 1.00000000
- kappa_c: 117.93990104
- cond_2(P_c): 69.64881147
- Lyapunov LMI max eigenvalue: -2.67328654e-02
- Radius coefficient multiplying Rbar: 7548.15366684

The common physical Lyapunov matrix makes the ADT threshold zero for
this predecessor-chain baseline. An ADT-violation ablation is therefore
not supported by this particular certified design.

## String performance

- alpha_0: 0.47000000 1/s
- Common-stability LMI worst numerical eigenvalue: -1.21653697e-02
- Nominal LMI worst numerical eigenvalue: 5.44844491e-10
- gamma_r (reported): 1.244
- Residual LMI max eigenvalue: -8.35408424e-03
- Joint-storage gamma_r (reported): 1.713
- Joint LMI max eigenvalue: 2.36754834e-10
- Exact frozen-mode headway boundary: 0.84186538 s

The tiny positive nominal-LMI eigenvalue is solver-scale roundoff at
the exact unit DC gain. The independent frequency inequalities are
nonnegative in every frozen mode.

## Theorem bounds

- Attack reconstruction: limsup ||x_tilde^a_ij|| <= 2438.91901303 dbar_ij.
- Platoon UUB: limsup ||X||^2 <= 7548.15366684 Rbar_a.
- For the seven-link predecessor chain with dbar_i,i-1 <= dbar:
  limsup ||X|| <= 5.60617649053e5 dbar.
- String-performance increment per link:
  gamma_r c_e = 3034.01525221.

These are deterministic worst-case bounds. Packet noise used in selected
simulations is not part of the theorem disturbance model.
