"""Exact eventual attack-rate and theorem-bound check for the low-frequency case."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
observer = json.loads((ROOT / "observer_dwell_family_certificate.json").read_text())
baseline = json.loads(
    (ROOT.parent / "simulations" / "certificates" / "baseline_certificates.json").read_text()
)

omega = 0.05
c = (-4.0, 1.4, 0.8)
ell = [1.0 + 0.05 * i for i in range(7)]

# After the 12-s activation, s_12(t)=1 and theta=omega(t-t_a).
# nu = col(dot y^a, dot x^a), with x^a=ell*[q,qdot,qddot]^T,
# q=20 sin(theta), y^a=ell*c*sin(theta).
cos_coeff = (20.0 * omega) ** 2 + (20.0 * omega**3) ** 2 + omega**2 * sum(v*v for v in c)
sin_coeff = (20.0 * omega**2) ** 2
if cos_coeff < sin_coeff:
    raise RuntimeError("Expected cosine coefficient to dominate for this case")
base_tail_rate = math.sqrt(cos_coeff)
tail_rates = [scale * base_tail_rate for scale in ell]
max_tail_rate = max(tail_rates)

ce = float(observer["c_e_s"])
reconstruction_bound = ce * max_tail_rate
observed_final_state_attack_error = 0.19506071785063786
reconstruction_gap_ratio = reconstruction_bound / observed_final_state_attack_error

# Baseline Theorem-2 coefficient from the regenerated public certificate.
physical_radius_coefficient = float(
    baseline["platoon"]["radius_coefficient_times_Rbar"]
)
rbar_tail = ce**2 * sum(rate**2 for rate in tail_rates)
theorem2_raw_X_norm_bound = math.sqrt(physical_radius_coefficient * rbar_tail)
observed_final_max_tracking_error_norm = 3.762635475115395
raw_uub_gap_ratio = theorem2_raw_X_norm_bound / observed_final_max_tracking_error_norm

result = {
    "omega_rad_s": omega,
    "auxiliary_attack_vector_c": list(c),
    "link_scales": ell,
    "cosine_coefficient": cos_coeff,
    "sine_coefficient": sin_coeff,
    "base_eventual_rate": base_tail_rate,
    "eventual_rate_by_link": tail_rates,
    "maximum_eventual_rate": max_tail_rate,
    "c_e_s": ce,
    "theorem1_eventual_reconstruction_bound": reconstruction_bound,
    "observed_final_state_attack_estimation_error": observed_final_state_attack_error,
    "theorem1_bound_to_observed_ratio": reconstruction_gap_ratio,
    "theorem2_physical_radius_coefficient": physical_radius_coefficient,
    "theorem2_Rbar_a_using_eventual_rates": rbar_tail,
    "theorem2_raw_stacked_state_norm_bound": theorem2_raw_X_norm_bound,
    "observed_final_max_tracking_error_norm": observed_final_max_tracking_error_norm,
    "theorem2_raw_bound_to_observed_ratio": raw_uub_gap_ratio,
    "note": (
        "The Theorem-2 quantity is the raw norm of the stacked deviation state X, "
        "which combines position, velocity, and acceleration coordinates; it is not labeled in metres."
    ),
}

output = json.dumps(result, indent=2) + "\n"
print(output, end="")
(ROOT / "low_frequency_tail_bound.json").write_text(output, encoding="utf-8")
(ROOT / "low_frequency_tail_bound_output.txt").write_text(output, encoding="utf-8")
