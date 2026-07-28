"""Regression tests for the switched trajectory model."""

from __future__ import annotations

import numpy as np

from simulations.src.model import (
    LOW_FREQUENCY_ATTACK_OMEGA,
    PlatoonSimulator,
    SimulationOptions,
    low_frequency_attack,
    ramp_attack,
)


def test_nominal_equilibrium_is_preserved_in_relative_coordinates() -> None:
    result = PlatoonSimulator(
        SimulationOptions(
            duration=2.0,
            step=0.01,
            sample_step=0.02,
            attack="none",
            perturb_initial_state=False,
        )
    ).run()
    assert np.max(np.abs(result.tracking_error)) < 1.0e-9
    assert np.max(np.abs(result.spacing_error)) < 1.0e-9
    assert np.min(result.gaps) > 0.0


def test_constant_attack_estimate_converges_after_smooth_onset() -> None:
    result = PlatoonSimulator(
        SimulationOptions(
            duration=28.0,
            step=0.01,
            sample_step=0.05,
            attack="constant",
            perturb_initial_state=False,
        )
    ).run()
    final_error = np.linalg.norm(result.x_estimation_error[-1], axis=1)
    assert np.max(final_error) < 2.0e-2
    assert np.min(result.gaps) > 0.0


def test_low_frequency_attack_is_continuous_and_has_expected_period() -> None:
    start_x, start_y = low_frequency_attack(8.0, links=2)
    period = 2.0 * np.pi / LOW_FREQUENCY_ATTACK_OMEGA
    repeated_x, repeated_y = low_frequency_attack(20.0 + period, links=2)
    reference_x, reference_y = low_frequency_attack(20.0, links=2)

    np.testing.assert_allclose(start_x, 0.0)
    np.testing.assert_allclose(start_y, 0.0)
    np.testing.assert_allclose(repeated_x, reference_x, atol=1.0e-12)
    np.testing.assert_allclose(repeated_y, reference_y, atol=1.0e-12)


def test_ramp_attack_has_consistent_steady_components() -> None:
    x_attack, y_attack = ramp_attack(20.0, links=2)
    later_x, _ = ramp_attack(21.0, links=2)

    np.testing.assert_allclose(later_x[:, 0] - x_attack[:, 0], x_attack[:, 1])
    np.testing.assert_allclose(x_attack[:, 2], 0.0)
    np.testing.assert_allclose(y_attack, 0.0)


def test_communication_noise_is_reproducible_and_low_power() -> None:
    options = SimulationOptions(
        duration=1.0,
        step=0.01,
        sample_step=0.02,
        communication_noise=True,
    )
    first = PlatoonSimulator(options)
    second = PlatoonSimulator(options)
    first_x, first_y = first.communication_noise(0.37)
    second_x, second_y = second.communication_noise(0.37)

    np.testing.assert_allclose(first_x, second_x)
    np.testing.assert_allclose(first_y, second_y)
    assert np.max(np.abs(first_x)) < 0.1
    assert np.max(np.abs(first_y)) < 0.1


def test_attack_can_be_restricted_to_one_link() -> None:
    simulator = PlatoonSimulator(
        SimulationOptions(
            duration=1.0,
            attack="low_frequency",
            attacked_links=(3,),
        )
    )
    x_attack, y_attack = simulator.attacks(40.0)

    inactive = [0, 1, 2, 4, 5, 6]
    np.testing.assert_allclose(x_attack[inactive], 0.0)
    np.testing.assert_allclose(y_attack[inactive], 0.0)
    assert np.linalg.norm(x_attack[3]) > 0.0
    assert np.linalg.norm(y_attack[3]) > 0.0


def test_switching_modes_are_both_exercised() -> None:
    result = PlatoonSimulator(
        SimulationOptions(
            duration=6.1,
            step=0.01,
            sample_step=0.05,
            attack="none",
        )
    ).run()
    assert set(result.physical_mode) == {0, 1, 2}
    assert set(result.auxiliary_mode) == {0, 1}


def test_switching_schedule_and_boundary_convention() -> None:
    simulator = PlatoonSimulator(SimulationOptions(duration=8.0))

    physical_times = [0.0, 1.999999, 2.0, 3.999999, 4.0, 5.999999, 6.0]
    physical_modes = [simulator.physical_mode(t) for t in physical_times]
    assert physical_modes == [0, 0, 1, 1, 2, 2, 0]

    auxiliary_times = [0.0, 0.399999, 0.4, 0.799999, 0.8, 1.199999, 1.2]
    auxiliary_modes = [simulator.auxiliary_mode(t) for t in auxiliary_times]
    assert auxiliary_modes == [0, 0, 1, 1, 0, 0, 1]

    assert np.isclose(2.0 / simulator.options.step, 400.0)
    assert np.isclose(0.4 / simulator.options.step, 80.0)
