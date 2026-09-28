"""Run the paper's nominal, attacked, and resilient trajectory studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from simulations.src.model import (
    LOW_FREQUENCY_ATTACK_OMEGA,
    PlatoonSimulator,
    SimulationOptions,
    SimulationResult,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "simulations" / "figures"
SEPARATE_FIGURE_DIR = FIGURE_DIR / "separate"
RESULT_DIR = ROOT / "simulations" / "results"

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#4d4d4d"]
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-."]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
STATE_LABELS = [r"position channel (m)", r"velocity channel (m/s)", r"acceleration channel (m/s$^2$)"]
AUX_LABELS = [r"auxiliary channel 1", r"auxiliary channel 2", r"auxiliary channel 3"]

# Publication figures use the raw observer trajectories.  Keeping this helper
# at zero preserves a single plotting path while ensuring no display-only
# filtering is applied.
DISPLAY_SMOOTH_S = 0.0


def _display_smooth(time: np.ndarray, y: np.ndarray, seconds: float = DISPLAY_SMOOTH_S) -> np.ndarray:
    dt = float(time[1] - time[0])
    window = max(1, int(round(seconds / dt)))
    if window <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    left = (window - 1) // 2
    right = (window - 1) - left
    padded = np.pad(np.asarray(y, dtype=float), (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "legend.handlelength": 2.4,
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.25,
        }
    )


def vehicle_trajectory_style(index: int, sample_count: int) -> dict[str, object]:
    """Return a color- and grayscale-distinguishable vehicle-trajectory style."""
    marker_step = max(1, sample_count // 14)
    return {
        "color": COLORS[index % len(COLORS)],
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)],
        "markevery": (index, marker_step),
        "markersize": 3.0,
        "markerfacecolor": "none",
        "markeredgewidth": 0.7,
    }


def save_figure(
    figure: plt.Figure,
    name: str,
    panel_names: list[str] | None = None,
    panel_xlabels: list[str] | None = None,
) -> None:
    """Save a composite figure and optionally crop each axis as a standalone panel."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight")
    if panel_names is not None:
        if len(panel_names) != len(figure.axes):
            raise ValueError("Each exported panel requires one panel name.")
        if panel_xlabels is None:
            panel_xlabels = [""] * len(panel_names)
        if len(panel_xlabels) != len(panel_names):
            raise ValueError("Each exported panel requires one x-axis label.")
        SEPARATE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        for axis, panel_name, standalone_xlabel in zip(
            figure.axes, panel_names, panel_xlabels
        ):
            original_xlabel = axis.get_xlabel()
            if standalone_xlabel:
                axis.set_xlabel(standalone_xlabel)
            temporary_legend = None
            if axis.get_legend() is None:
                handles, labels = axis.get_legend_handles_labels()
                visible = [
                    (handle, label)
                    for handle, label in zip(handles, labels)
                    if label and not label.startswith("_")
                ]
                if len(visible) >= 2:
                    temporary_legend = axis.legend(
                        *zip(*visible),
                        ncol=2 if len(visible) > 4 else 1,
                        loc="best",
                    )
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            bbox = axis.get_tightbbox(renderer).transformed(
                figure.dpi_scale_trans.inverted()
            )
            bbox = bbox.expanded(1.02, 1.02)
            output_stem = SEPARATE_FIGURE_DIR / f"{name}_{panel_name}"
            figure.savefig(
                f"{output_stem}.pdf", bbox_inches=bbox, pad_inches=0.01
            )
            figure.savefig(
                f"{output_stem}.png",
                bbox_inches=bbox,
                pad_inches=0.01,
            )
            if temporary_legend is not None:
                temporary_legend.remove()
            axis.set_xlabel(original_xlabel)
    plt.close(figure)


def nominal_figure(result: SimulationResult) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(3.35, 3.8), sharex=True)
    tracking_norm = np.linalg.norm(result.tracking_error, axis=2)
    for follower in range(tracking_norm.shape[1]):
        color = COLORS[follower % len(COLORS)]
        axes[0].plot(result.time, tracking_norm[:, follower], color=color, label=rf"$i={follower + 1}$")
        axes[1].plot(result.time, result.spacing_error[:, follower], color=color, label=rf"$i={follower + 1}$")
        axes[2].plot(result.time, result.gaps[:, follower], color=color, label=rf"$i={follower + 1}$")
    axes[0].set_ylabel(r"$\|e_i\|_2$")
    axes[1].set_ylabel(r"$\zeta_i$ (m)")
    axes[2].set_ylabel("distance (m)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(ncol=4, loc="upper right")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    figure.tight_layout(pad=0.25, h_pad=0.2)
    save_figure(
        figure,
        "nominal_validation",
        ["tracking_error", "spacing_error", "intervehicle_distance"],
        ["time (s)"] * 3,
    )


def attack_figure(result: SimulationResult, scenario: str, link: int) -> None:
    if scenario == "ramp":
        figure, axes = plt.subplots(3, 1, figsize=(3.35, 4.6), sharex=True)
        compact_labels = [
            r"$x_s^a$ (m)",
            r"$x_v^a$ (m/s)",
            r"$x_a^a$ (m/s$^2$)",
        ]
        for component in range(3):
            axis = axes[component]
            axis.plot(
                result.time,
                result.x_attack[:, link, component],
                color="#111111",
                label="true",
            )
            axis.plot(
                result.time,
                result.estimated_x_attack[:, link, component],
                color="#d62728",
                linestyle="--",
                label="estimated",
            )
            axis.set_ylabel(compact_labels[component])
        axes[-1].set_xlabel("time (s)")
        axes[0].legend(loc="upper left")
        figure.tight_layout(pad=0.25, h_pad=0.2)
        save_figure(
            figure,
            f"attack_{scenario}",
            ["state_position", "state_velocity", "state_acceleration"],
            ["time (s)"] * 3,
        )
        return

    figure, axes = plt.subplots(6, 1, figsize=(3.35, 7.0), sharex=True)
    for component in range(3):
        state_axis = axes[component]
        auxiliary_axis = axes[component + 3]
        state_axis.plot(result.time, result.x_attack[:, link, component], color="#111111", label="true")
        state_axis.plot(result.time, _display_smooth(result.time, result.estimated_x_attack[:, link, component]), color="#d62728", linestyle="--", label="estimated")
        auxiliary_axis.plot(result.time, result.y_attack[:, link, component], color="#111111", label="true")
        auxiliary_axis.plot(
            result.time,
            _display_smooth(result.time, result.estimated_y_attack[:, link, component]),
            color="#1f77b4",
            linestyle="--",
            linewidth=0.8 if component == 0 else 1.2,
            label="estimated",
        )
        state_axis.set_ylabel(
            [
                r"$x_s^a$ (m)",
                r"$x_v^a$ (m/s)",
                r"$x_a^a$ (m/s$^2$)",
            ][component]
        )
        auxiliary_axis.set_ylabel(
            [r"$y_1^a$", r"$y_2^a$", r"$y_3^a$"][component]
        )
    axes[0].set_title("State-packet FDI")
    axes[3].set_title("Auxiliary-output FDI")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right")
    axes[3].legend(loc="upper right")
    figure.tight_layout(pad=0.25, h_pad=0.2)
    save_figure(
        figure,
        f"attack_{scenario}",
        [
            "state_position",
            "state_velocity",
            "state_acceleration",
            "auxiliary_1",
            "auxiliary_2",
            "auxiliary_3",
        ],
        ["time (s)"] * 6,
    )


def estimation_error_figure(
    result: SimulationResult, scenario: str
) -> None:
    x_error = np.linalg.norm(
        result.x_attack - result.estimated_x_attack, axis=2
    )
    y_error = np.linalg.norm(
        result.y_attack - result.estimated_y_attack, axis=2
    )
    figure, axes = plt.subplots(2, 1, figsize=(3.35, 2.8), sharex=True)
    for link in range(x_error.shape[1]):
        color = COLORS[link % len(COLORS)]
        axes[0].plot(
            result.time,
            x_error[:, link],
            color=color,
            label=rf"$({link + 1},{link})$",
        )
        axes[1].plot(
            result.time,
            y_error[:, link],
            color=color,
            label=rf"$({link + 1},{link})$",
        )
    axes[0].set_ylabel(r"$\|x^a_{ij}-\hat x^a_{ij}\|_2$")
    axes[1].set_ylabel(r"$\|y^a_{ij}-\hat y^a_{ij}\|_2$")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(ncol=4, loc="upper right")
    figure.tight_layout(pad=0.25, h_pad=0.2)
    save_figure(
        figure,
        f"estimation_errors_{scenario}",
        ["state_packet_error", "auxiliary_output_error"],
        ["time (s)"] * 2,
    )


def platoon_comparison_figure(
    uncompensated: SimulationResult,
    resilient: SimulationResult,
    scenario: str,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(3.35, 6.2), sharex=True)
    cases = [
        ("Uncompensated controller", uncompensated),
        ("Proposed resilient controller", resilient),
    ]
    for case_index, (title, result) in enumerate(cases):
        spacing_axis = axes[2 * case_index]
        velocity_axis = axes[2 * case_index + 1]
        velocity_error = (
            result.physical[:, 1:, 1] - result.physical[:, [0], 1]
        )
        for follower in range(result.spacing_error.shape[1]):
            label = rf"$i={follower + 1}$"
            style = vehicle_trajectory_style(follower, result.time.size)
            spacing_axis.plot(
                result.time,
                result.spacing_error[:, follower],
                label=label,
                **style,
            )
            velocity_axis.plot(
                result.time,
                velocity_error[:, follower],
                label=label,
                **style,
            )
        spacing_axis.set_title(title)
        spacing_axis.set_ylabel(r"$\zeta_i$ (m)")
        velocity_axis.set_ylabel(r"$v_i-v_0$ (m/s)")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(ncol=2, loc="upper left")
    figure.tight_layout(pad=0.25, h_pad=0.3)
    save_figure(
        figure,
        f"platoon_response_{scenario}",
        [
            "uncompensated_spacing",
            "uncompensated_velocity",
            "resilient_spacing",
            "resilient_velocity",
        ],
        ["time (s)"] * 4,
    )


def downstream_acceleration_metrics(
    result: SimulationResult,
    attacked_link: int,
    reference: SimulationResult | None = None,
    window_start: float = 30.0,
) -> dict[str, float | list[float]]:
    """Measure propagation from the directly affected follower downstream."""
    window = result.time >= window_start
    first_vehicle = attacked_link + 1
    acceleration = result.physical[
        window, first_vehicle:, 2
    ]
    if reference is not None:
        acceleration = (
            acceleration
            - reference.physical[window, first_vehicle:, 2]
        )
    rms = np.sqrt(np.mean(acceleration**2, axis=0))
    ratios = rms / rms[0]
    adjacent = rms[1:] / rms[:-1]
    return {
        "window_start_s": window_start,
        "vehicle_indices": list(
            range(first_vehicle, result.physical.shape[1])
        ),
        "incremental_rms_acceleration_m_s2": rms.tolist(),
        "normalized_incremental_rms_acceleration": ratios.tolist(),
        "maximum_adjacent_rms_ratio": float(np.max(adjacent)),
        "maximum_absolute_incremental_acceleration_m_s2": float(
            np.max(np.abs(acceleration))
        ),
    }


def single_link_propagation_figure(
    uncompensated: SimulationResult,
    resilient: SimulationResult,
    uncompensated_reference: SimulationResult,
    resilient_reference: SimulationResult,
    attacked_link: int,
) -> None:
    first_vehicle = attacked_link + 1
    vehicle_indices = np.arange(
        first_vehicle, resilient.physical.shape[1]
    )
    cases = [
        (
            "Uncompensated controller",
            uncompensated,
            uncompensated_reference,
        ),
        (
            "Proposed resilient controller",
            resilient,
            resilient_reference,
        ),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(3.35, 3.8), sharex=True)
    for case_index, (title, result, reference) in enumerate(cases):
        acceleration_axis = axes[case_index]
        for vehicle in vehicle_indices:
            style_index = vehicle - first_vehicle
            style = vehicle_trajectory_style(style_index, result.time.size)
            acceleration_axis.plot(
                result.time,
                (
                    result.physical[:, vehicle, 2]
                    - reference.physical[:, vehicle, 2]
                ),
                label=rf"$i={vehicle}$",
                **style,
            )
        acceleration_axis.set_title(title)
        acceleration_axis.set_ylabel(r"$\Delta a_i$ (m/s$^2$)")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(ncol=2, loc="upper right")
    figure.tight_layout(pad=0.25, h_pad=0.3)
    save_figure(
        figure,
        "single_link_low_frequency_propagation",
        [
            "uncompensated_acceleration",
            "resilient_acceleration",
        ],
        ["time (s)", "time (s)"],
    )


def sampled_rate_bound(result: SimulationResult) -> float:
    combined = np.concatenate([result.y_attack, result.x_attack], axis=2)
    derivative = np.gradient(combined, result.time, axis=0, edge_order=2)
    return float(np.max(np.linalg.norm(derivative, axis=2)))


def scenario_metrics(result: SimulationResult, attack_start: float = 8.0) -> dict[str, float | list[float]]:
    tracking_norm = np.linalg.norm(result.tracking_error, axis=2)
    x_error = np.linalg.norm(result.x_estimation_error, axis=2)
    y_error = np.linalg.norm(result.y_attack - result.estimated_y_attack, axis=2)
    attacked = result.time >= attack_start
    tail = result.time >= max(0.0, result.time[-1] - 5.0)
    return {
        "duration_s": float(result.time[-1]),
        "minimum_intervehicle_distance_m": float(np.min(result.gaps)),
        "maximum_tracking_error_norm": float(np.max(tracking_norm)),
        "maximum_tracking_error_norm_after_attack": float(np.max(tracking_norm[attacked])),
        "final_maximum_tracking_error_norm": float(np.max(tracking_norm[-1])),
        "final_maximum_state_attack_error_norm": float(np.max(x_error[-1])),
        "final_maximum_auxiliary_attack_error_norm": float(np.max(y_error[-1])),
        "tail_rms_state_attack_error_by_link": np.sqrt(np.mean(x_error[tail] ** 2, axis=0)).tolist(),
        "tail_rms_auxiliary_attack_error_by_link": np.sqrt(np.mean(y_error[tail] ** 2, axis=0)).tolist(),
        "sampled_maximum_attack_rate": sampled_rate_bound(result),
        "maximum_absolute_control": float(np.max(np.abs(result.control))),
        "rms_state_packet_noise_by_component": np.sqrt(
            np.mean(result.x_noise**2, axis=(0, 1))
        ).tolist(),
        "rms_auxiliary_output_noise_by_component": np.sqrt(
            np.mean(result.y_noise**2, axis=(0, 1))
        ).tolist(),
    }


def run_all(step: float) -> dict[str, object]:
    definitions = {
        "low_frequency": ("low_frequency", 300.0, 3),
        "ramp": ("ramp", 50.0, 3),
    }
    config = json.loads(
        (ROOT / "simulations" / "configs" / "baseline.json").read_text(
            encoding="utf-8"
        )
    )
    metrics: dict[str, object] = {
        "noise": config["noise"],
        "attacks": {},
    }

    nominal = PlatoonSimulator(
        SimulationOptions(
            duration=35.0,
            step=step,
            sample_step=0.02,
            attack="none",
            compensate=True,
            perturb_initial_state=True,
            communication_noise=True,
        )
    ).run()
    metrics["nominal"] = scenario_metrics(nominal)
    nominal_figure(nominal)

    attack_metrics: dict[str, object] = {}
    for name, (attack, duration, representative_link) in definitions.items():
        common_options = dict(
            duration=duration,
            step=step,
            sample_step=0.02,
            attack=attack,
            perturb_initial_state=True,
            communication_noise=True,
        )
        resilient = PlatoonSimulator(
            SimulationOptions(**common_options, compensate=True)
        ).run()
        uncompensated = PlatoonSimulator(
            SimulationOptions(**common_options, compensate=False)
        ).run()
        attack_metrics[name] = {
            "resilient": scenario_metrics(resilient),
            "uncompensated": scenario_metrics(uncompensated),
        }
        attack_figure(resilient, name, representative_link)
        estimation_error_figure(resilient, name)
        platoon_comparison_figure(uncompensated, resilient, name)
    metrics["attacks"] = attack_metrics

    attack_metrics["low_frequency"].update(
        {
            "attack_angular_frequency_rad_s": LOW_FREQUENCY_ATTACK_OMEGA,
            "attack_frequency_hz": LOW_FREQUENCY_ATTACK_OMEGA / (2.0 * np.pi),
            "attack_period_s": 2.0 * np.pi / LOW_FREQUENCY_ATTACK_OMEGA,
        }
    )

    attacked_link = 3
    single_link_options = dict(
        duration=300.0,
        step=step,
        sample_step=0.02,
        attack="low_frequency",
        attacked_links=(attacked_link,),
        perturb_initial_state=True,
        communication_noise=True,
    )
    single_resilient = PlatoonSimulator(
        SimulationOptions(**single_link_options, compensate=True)
    ).run()
    single_uncompensated = PlatoonSimulator(
        SimulationOptions(**single_link_options, compensate=False)
    ).run()
    reference_options = dict(single_link_options)
    reference_options.update(attack="none", attacked_links=None)
    single_resilient_reference = PlatoonSimulator(
        SimulationOptions(**reference_options, compensate=True)
    ).run()
    single_uncompensated_reference = PlatoonSimulator(
        SimulationOptions(**reference_options, compensate=False)
    ).run()
    metrics["single_link_low_frequency"] = {
        "attacked_link": [4, 3],
        "propagation_quantity": (
            "attack-induced acceleration obtained by subtracting the "
            "matched no-attack trajectory with identical noise"
        ),
        "resilient": {
            "scenario": scenario_metrics(single_resilient),
            "downstream_propagation": downstream_acceleration_metrics(
                single_resilient,
                attacked_link,
                single_resilient_reference,
            ),
        },
        "uncompensated": {
            "scenario": scenario_metrics(single_uncompensated),
            "downstream_propagation": downstream_acceleration_metrics(
                single_uncompensated,
                attacked_link,
                single_uncompensated_reference,
            ),
        },
    }
    single_link_propagation_figure(
        single_uncompensated,
        single_resilient,
        single_uncompensated_reference,
        single_resilient_reference,
        attacked_link,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "baseline_scenario_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.005, help="RK4 integration step in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    configure_plotting()
    results = run_all(parse_args().step)
    print(json.dumps(results, indent=2))
