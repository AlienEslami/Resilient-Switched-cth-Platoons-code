"""Deterministic switched-platoon trajectory model used by all scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json

import numpy as np

from simulations.src.matrices import augmented_matrices, vehicle_matrices

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "simulations" / "configs" / "baseline.json"

AttackFunction = Callable[[float, int], tuple[np.ndarray, np.ndarray]]
LeaderCommand = Callable[[float], float]


def smooth_activation(t: float, start: float, rise: float) -> float:
    """C2 transition from zero to one over ``[start, start + rise]``."""
    return smooth_activation_derivatives(t, start, rise)[0]


def smooth_activation_derivatives(
    t: float, start: float, rise: float
) -> tuple[float, float, float]:
    """Return a quintic activation and its first two time derivatives."""
    if t <= start:
        return 0.0, 0.0, 0.0
    if t >= start + rise:
        return 1.0, 0.0, 0.0
    u = (t - start) / rise
    value = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
    first = (30.0 * u**2 * (u - 1.0) ** 2) / rise
    second = (60.0 * u * (2.0 * u**2 - 3.0 * u + 1.0)) / rise**2
    return float(value), float(first), float(second)


def activated_trajectory(
    t: float,
    start: float,
    rise: float,
    value: float,
    first: float,
    second: float,
) -> np.ndarray:
    """Form ``[q, q_dot, q_ddot]`` for an activated scalar trajectory."""
    envelope, envelope_dot, envelope_ddot = (
        smooth_activation_derivatives(t, start, rise)
    )
    return np.array(
        [
            envelope * value,
            envelope_dot * value + envelope * first,
            envelope_ddot * value
            + 2.0 * envelope_dot * first
            + envelope * second,
        ]
    )


def no_attack(t: float, links: int) -> tuple[np.ndarray, np.ndarray]:
    del t
    return np.zeros((links, 3)), np.zeros((links, 3))


def constant_attack(t: float, links: int) -> tuple[np.ndarray, np.ndarray]:
    start = 8.0
    rise = 8.0
    envelope = smooth_activation(t, start=start, rise=rise)
    link_scale = 1.0 + 0.08 * np.arange(links, dtype=float)[:, None]
    x_base = activated_trajectory(
        t, start=start, rise=rise, value=15.0, first=0.0, second=0.0
    )
    x_attack = link_scale * x_base
    y_attack = link_scale * np.array([-4.0, 1.2, 0.8]) * envelope
    return x_attack, y_attack


def sinusoidal_attack(t: float, links: int) -> tuple[np.ndarray, np.ndarray]:
    start = 8.0
    envelope = smooth_activation(t, start=start, rise=2.0)
    phase = max(0.0, t - start)
    link_scale = 1.0 + 0.05 * np.arange(links, dtype=float)[:, None]
    x_base = np.array(
        [
            7.0 * np.sin(0.32 * phase),
            1.6 * np.sin(0.57 * phase + 0.4),
            0.45 * np.cos(0.43 * phase),
        ]
    )
    y_base = np.array(
        [
            3.0 * np.cos(0.27 * phase),
            -1.3 * np.sin(0.48 * phase),
            0.7 * np.sin(0.36 * phase + 0.2),
        ]
    )
    return link_scale * x_base * envelope, link_scale * y_base * envelope


LOW_FREQUENCY_ATTACK_OMEGA = 0.05


def low_frequency_attack(t: float, links: int) -> tuple[np.ndarray, np.ndarray]:
    """Kinematically consistent false trajectory at 0.05 rad/s."""
    start = 8.0
    envelope = smooth_activation(t, start=start, rise=12.0)
    phase = max(0.0, t - start)
    omega = LOW_FREQUENCY_ATTACK_OMEGA
    amplitude = 20.0
    x_base = activated_trajectory(
        t,
        start=start,
        rise=12.0,
        value=amplitude * np.sin(omega * phase),
        first=amplitude * omega * np.cos(omega * phase),
        second=-amplitude * omega**2 * np.sin(omega * phase),
    )
    signal = np.sin(omega * phase)
    link_scale = 1.0 + 0.05 * np.arange(links, dtype=float)[:, None]
    x_attack = link_scale * x_base
    y_attack = link_scale * np.array([-4.0, 1.4, 0.8]) * signal * envelope
    return x_attack, y_attack


def ramp_attack(t: float, links: int) -> tuple[np.ndarray, np.ndarray]:
    """Kinematically consistent ramp false trajectory."""
    start = 8.0
    phase = max(0.0, t - start)
    link_scale = 1.0 + 0.05 * np.arange(links, dtype=float)[:, None]
    slope = 0.35
    x_base = activated_trajectory(
        t,
        start=start,
        rise=2.0,
        value=slope * phase,
        first=slope,
        second=0.0,
    )
    x_attack = link_scale * x_base
    y_attack = np.zeros((links, 3))
    return x_attack, y_attack


def leader_link_attack(t: float, links: int) -> tuple[np.ndarray, np.ndarray]:
    x_attack = np.zeros((links, 3))
    y_attack = np.zeros((links, 3))
    envelope = smooth_activation(t, start=8.0, rise=2.0)
    phase = max(0.0, t - 8.0)
    x_attack[0] = envelope * np.array(
        [12.0 + 2.0 * np.sin(0.25 * phase), 2.2, 0.6]
    )
    y_attack[0] = envelope * np.array(
        [-5.0, 1.5 * np.sin(0.4 * phase), 1.0]
    )
    return x_attack, y_attack


ATTACKS: dict[str, AttackFunction] = {
    "none": no_attack,
    "constant": constant_attack,
    "sinusoidal": sinusoidal_attack,
    "low_frequency": low_frequency_attack,
    "ramp": ramp_attack,
    "leader_link": leader_link_attack,
}


@dataclass(frozen=True)
class SimulationOptions:
    duration: float = 40.0
    step: float = 0.005
    sample_step: float = 0.02
    standstill_gap: float = 5.0
    cruise_speed: float = 20.0
    physical_dwell: float = 2.0
    attack: str = "none"
    compensate: bool = True
    perturb_initial_state: bool = True
    headway_override: float | None = None
    communication_noise: bool = False
    attacked_links: tuple[int, ...] | None = None


@dataclass
class SimulationResult:
    time: np.ndarray
    physical: np.ndarray
    auxiliary: np.ndarray
    observer: np.ndarray
    x_attack: np.ndarray
    y_attack: np.ndarray
    x_noise: np.ndarray
    y_noise: np.ndarray
    control: np.ndarray
    physical_mode: np.ndarray
    auxiliary_mode: np.ndarray
    headway: float
    standstill_gap: float
    cruise_speed: float

    @property
    def tracking_error(self) -> np.ndarray:
        followers = self.physical.shape[1] - 1
        reference = np.empty_like(self.physical[:, 1:, :])
        for index in range(followers):
            reference[:, index, 0] = (
                self.physical[:, 0, 0]
                - (index + 1)
                * (self.standstill_gap + self.headway * self.cruise_speed)
            )
            reference[:, index, 1] = self.cruise_speed
            reference[:, index, 2] = 0.0
        return self.physical[:, 1:, :] - reference

    @property
    def spacing_error(self) -> np.ndarray:
        predecessor = self.physical[:, :-1, :]
        follower = self.physical[:, 1:, :]
        return (
            predecessor[:, :, 0]
            - follower[:, :, 0]
            - self.standstill_gap
            - self.headway * follower[:, :, 1]
        )

    @property
    def gaps(self) -> np.ndarray:
        return self.physical[:, :-1, 0] - self.physical[:, 1:, 0]

    @property
    def estimated_x_attack(self) -> np.ndarray:
        return self.observer[:, :, 6:9]

    @property
    def estimated_y_attack(self) -> np.ndarray:
        return self.observer[:, :, 3:6]

    @property
    def x_estimation_error(self) -> np.ndarray:
        return self.x_attack - self.estimated_x_attack


class PlatoonSimulator:
    """Switched physical, auxiliary, and observer dynamics for a chain."""

    def __init__(
        self,
        options: SimulationOptions,
        config_path: Path = DEFAULT_CONFIG,
        leader_command: LeaderCommand | None = None,
    ) -> None:
        self.options = options
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        platoon = self.config["platoon"]
        auxiliary = self.config["auxiliary"]

        self.followers = int(platoon["followers"])
        self.taus = np.asarray(platoon["powertrain_lags"], dtype=float)
        self.headway = (
            float(platoon["headway"])
            if options.headway_override is None
            else float(options.headway_override)
        )
        self.gains = np.asarray(platoon["controller_gains"], dtype=float)
        if self.gains.shape != (len(self.taus), 3):
            raise ValueError(
                "controller_gains must contain one three-component gain "
                "for each powertrain mode"
            )
        self.auxiliary_dwell = np.asarray(
            auxiliary["dwell_times"], dtype=float
        )
        self.a_z = [np.asarray(item, dtype=float) for item in auxiliary["A_z"]]
        self.b_z = [np.asarray(item, dtype=float) for item in auxiliary["B_z"]]
        self.c_z = [np.asarray(item, dtype=float) for item in auxiliary["C_z"]]
        self.observer_gain = [
            np.asarray(item, dtype=float) for item in auxiliary["L"]
        ]
        noise = self.config["noise"]
        self.x_noise_std = np.asarray(noise["state_packet_std"], dtype=float)
        self.y_noise_std = np.asarray(
            noise["auxiliary_output_std"], dtype=float
        )
        self.noise_sample_period = float(noise["sample_period"])
        self.noise_correlation_time = float(noise["correlation_time"])
        self.noise_seed = int(noise["seed"])
        augmented = [
            augmented_matrices(a, b, c)
            for a, b, c in zip(self.a_z, self.b_z, self.c_z)
        ]
        self.a_aug = [item[0] for item in augmented]
        self.c_aug = [item[1] for item in augmented]
        self.b_aug = [
            np.vstack([b, np.zeros((6, 3))]) for b in self.b_z
        ]
        self.attack = ATTACKS[options.attack]
        if options.attacked_links is not None:
            invalid = [
                link
                for link in options.attacked_links
                if link < 0 or link >= self.followers
            ]
            if invalid:
                raise ValueError(
                    f"attacked link indices out of range: {invalid}"
                )
        self.leader_command = leader_command or (lambda t: 0.0)

        self.physical_size = 3 * (self.followers + 1)
        self.auxiliary_size = 3 * self.followers
        self.observer_size = 9 * self.followers
        self.state_size = (
            self.physical_size + self.auxiliary_size + self.observer_size
        )
        self._prepare_communication_noise()

    def _prepare_communication_noise(self) -> None:
        """Generate reproducible, low-power, correlated Gaussian packet noise."""
        if not self.options.communication_noise:
            self.noise_time = np.array([0.0, self.options.duration + 1.0])
            self.x_noise_samples = np.zeros((2, self.followers, 3))
            self.y_noise_samples = np.zeros((2, self.followers, 3))
            return

        count = int(
            np.ceil(self.options.duration / self.noise_sample_period)
        ) + 2
        self.noise_time = (
            np.arange(count, dtype=float) * self.noise_sample_period
        )
        rho = np.exp(
            -self.noise_sample_period / self.noise_correlation_time
        )
        innovation_scale = np.sqrt(1.0 - rho**2)
        rng = np.random.default_rng(self.noise_seed)

        def correlated_samples(std: np.ndarray) -> np.ndarray:
            samples = np.empty((count, self.followers, 3))
            samples[0] = rng.normal(size=(self.followers, 3)) * std
            for index in range(1, count):
                innovation = rng.normal(size=(self.followers, 3)) * std
                samples[index] = (
                    rho * samples[index - 1]
                    + innovation_scale * innovation
                )
            return samples

        self.x_noise_samples = correlated_samples(self.x_noise_std)
        self.y_noise_samples = correlated_samples(self.y_noise_std)

    def communication_noise(
        self, t: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate the fixed noise realization at time ``t``."""
        local = np.clip(
            t / self.noise_sample_period,
            0.0,
            len(self.noise_time) - 1.0,
        )
        left = min(int(np.floor(local)), len(self.noise_time) - 2)
        weight = local - left
        x_noise = (
            (1.0 - weight) * self.x_noise_samples[left]
            + weight * self.x_noise_samples[left + 1]
        )
        y_noise = (
            (1.0 - weight) * self.y_noise_samples[left]
            + weight * self.y_noise_samples[left + 1]
        )
        return x_noise, y_noise

    def attacks(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the configured attacks after applying an optional link mask."""
        x_attack, y_attack = self.attack(t, self.followers)
        if self.options.attacked_links is None:
            return x_attack, y_attack

        mask = np.zeros((self.followers, 1))
        mask[list(self.options.attacked_links)] = 1.0
        return x_attack * mask, y_attack * mask

    def physical_mode(self, t: float) -> int:
        return int(np.floor((t + 1.0e-12) / self.options.physical_dwell)) % len(
            self.taus
        )

    def auxiliary_mode(self, t: float) -> int:
        period = float(np.sum(self.auxiliary_dwell))
        local = (t + 1.0e-12) % period
        boundaries = np.cumsum(self.auxiliary_dwell)
        return int(np.searchsorted(boundaries, local, side="right")) % len(
            self.auxiliary_dwell
        )

    def initial_state(self) -> np.ndarray:
        physical = np.zeros((self.followers + 1, 3))
        physical[:, 1] = self.options.cruise_speed
        desired_gap = (
            self.options.standstill_gap
            + self.headway * self.options.cruise_speed
        )
        physical[:, 0] = -desired_gap * np.arange(self.followers + 1)

        if self.options.perturb_initial_state:
            position_offsets = np.array(
                [0.0, 1.5, -1.0, 2.0, -1.5, 1.0, -0.5, 1.2]
            )
            velocity_offsets = np.array(
                [0.0, 0.4, -0.3, 0.2, -0.4, 0.3, -0.2, 0.1]
            )
            physical[:, 0] += position_offsets[: self.followers + 1]
            physical[:, 1] += velocity_offsets[: self.followers + 1]

        auxiliary = np.zeros((self.followers, 3))
        observer = np.zeros((self.followers, 9))
        return np.concatenate(
            [physical.ravel(), auxiliary.ravel(), observer.ravel()]
        )

    def unpack(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        physical_end = self.physical_size
        auxiliary_end = physical_end + self.auxiliary_size
        physical = state[:physical_end].reshape(self.followers + 1, 3)
        auxiliary = state[physical_end:auxiliary_end].reshape(self.followers, 3)
        observer = state[auxiliary_end:].reshape(self.followers, 9)
        return physical, auxiliary, observer

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        physical, auxiliary, observer = self.unpack(state)
        p_mode = self.physical_mode(t)
        z_mode = self.auxiliary_mode(t)
        a_p, b_p = vehicle_matrices(float(self.taus[p_mode]))
        gain = self.gains[p_mode]
        x_attack, y_attack = self.attacks(t)
        x_noise, y_noise = self.communication_noise(t)

        physical_dot = np.zeros_like(physical)
        auxiliary_dot = np.zeros_like(auxiliary)
        observer_dot = np.zeros_like(observer)

        physical_dot[0] = a_p @ physical[0] + b_p[:, 0] * self.leader_command(t)
        for link in range(self.followers):
            predecessor = physical[link]
            follower = physical[link + 1]
            x_star = predecessor + x_attack[link] + x_noise[link]
            y_star = (
                self.c_z[z_mode] @ auxiliary[link]
                + y_attack[link]
                + y_noise[link]
            )

            auxiliary_dot[link] = (
                self.a_z[z_mode] @ auxiliary[link]
                + self.b_z[z_mode] @ predecessor
            )
            innovation = y_star - self.c_aug[z_mode] @ observer[link]
            observer_dot[link] = (
                self.a_aug[z_mode] @ observer[link]
                + self.b_aug[z_mode] @ x_star
                + self.observer_gain[z_mode] @ innovation
            )

            estimate = observer[link, 6:9] if self.options.compensate else 0.0
            compensated = x_star - estimate
            spacing = (
                compensated[0]
                - follower[0]
                - self.options.standstill_gap
                - self.headway * follower[1]
            )
            relative_velocity = compensated[1] - follower[1]
            relative_acceleration = compensated[2] - follower[2]
            control = float(
                gain
                @ np.array([spacing, relative_velocity, relative_acceleration])
            )
            physical_dot[link + 1] = a_p @ follower + b_p[:, 0] * control

        return np.concatenate(
            [physical_dot.ravel(), auxiliary_dot.ravel(), observer_dot.ravel()]
        )

    def controls(self, t: float, state: np.ndarray) -> np.ndarray:
        physical, _, observer = self.unpack(state)
        gain = self.gains[self.physical_mode(t)]
        x_attack, _ = self.attacks(t)
        x_noise, _ = self.communication_noise(t)
        controls = np.zeros(self.followers)
        for link in range(self.followers):
            estimate = observer[link, 6:9] if self.options.compensate else 0.0
            compensated = (
                physical[link] + x_attack[link] + x_noise[link] - estimate
            )
            follower = physical[link + 1]
            local_state = np.array(
                [
                    compensated[0]
                    - follower[0]
                    - self.options.standstill_gap
                    - self.headway * follower[1],
                    compensated[1] - follower[1],
                    compensated[2] - follower[2],
                ]
            )
            controls[link] = gain @ local_state
        return controls

    def run(self) -> SimulationResult:
        ratio = self.options.sample_step / self.options.step
        save_every = int(round(ratio))
        if not np.isclose(ratio, save_every) or save_every < 1:
            raise ValueError("sample_step must be an integer multiple of step")

        steps = int(round(self.options.duration / self.options.step))
        if not np.isclose(steps * self.options.step, self.options.duration):
            raise ValueError("duration must be an integer multiple of step")

        state = self.initial_state()
        samples = steps // save_every + 1
        time = np.empty(samples)
        state_history = np.empty((samples, self.state_size))
        x_attack_history = np.empty((samples, self.followers, 3))
        y_attack_history = np.empty((samples, self.followers, 3))
        x_noise_history = np.empty((samples, self.followers, 3))
        y_noise_history = np.empty((samples, self.followers, 3))
        control_history = np.empty((samples, self.followers))
        physical_modes = np.empty(samples, dtype=int)
        auxiliary_modes = np.empty(samples, dtype=int)

        sample = 0
        dt = self.options.step
        for step in range(steps + 1):
            t = step * dt
            if step % save_every == 0:
                time[sample] = t
                state_history[sample] = state
                x_attack_history[sample], y_attack_history[sample] = (
                    self.attacks(t)
                )
                x_noise_history[sample], y_noise_history[sample] = (
                    self.communication_noise(t)
                )
                control_history[sample] = self.controls(t, state)
                physical_modes[sample] = self.physical_mode(t)
                auxiliary_modes[sample] = self.auxiliary_mode(t)
                sample += 1
            if step == steps:
                break

            k1 = self.derivative(t, state)
            k2 = self.derivative(t + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = self.derivative(t + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = self.derivative(t + dt, state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        physical_end = self.physical_size
        auxiliary_end = physical_end + self.auxiliary_size
        physical_history = state_history[:, :physical_end].reshape(
            samples, self.followers + 1, 3
        )
        auxiliary_history = state_history[:, physical_end:auxiliary_end].reshape(
            samples, self.followers, 3
        )
        observer_history = state_history[:, auxiliary_end:].reshape(
            samples, self.followers, 9
        )
        return SimulationResult(
            time=time,
            physical=physical_history,
            auxiliary=auxiliary_history,
            observer=observer_history,
            x_attack=x_attack_history,
            y_attack=y_attack_history,
            x_noise=x_noise_history,
            y_noise=y_noise_history,
            control=control_history,
            physical_mode=physical_modes,
            auxiliary_mode=auxiliary_modes,
            headway=self.headway,
            standstill_gap=self.options.standstill_gap,
            cruise_speed=self.options.cruise_speed,
        )
