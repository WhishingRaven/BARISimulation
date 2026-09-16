"""Experimental traveling-wave gait; motion remains entirely physics-driven."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sin

from ..types import LocalObservation, RobotAction


@dataclass
class TravelingWaveGait:
    # Keep a margin below the +/-135-degree hard stops while using nearly the
    # entire available stroke. A slow cycle lets the torque-limited joints
    # physically reach these targets instead of merely requesting them.
    amplitude_rad: float = 0.7
    frequency_hz: float = 1.5
    phase_offset_rad: float = 5.0 * pi / 12.0
    direction: float = 1.0
    startup_duration_s: float = 0.15
    _start_time_s: float | None = field(default=None, init=False, repr=False)
    _start_positions_rad: tuple[float, ...] = field(
        default_factory=tuple, init=False, repr=False
    )

    def targets(self, simulation_time_s: float, joint_count: int) -> tuple[float, ...]:
        # Positive direction means robot-forward (+local X). The shape wave must
        # travel toward the rear to produce forward motion with this pitch chain.
        phase = -self.direction * 2.0 * pi * self.frequency_hz * simulation_time_s
        return tuple(
            self.amplitude_rad * sin(phase + joint_id * self.phase_offset_rad)
            for joint_id in range(joint_count)
        )

    def stance_links(
        self, simulation_time_s: float, link_count: int
    ) -> tuple[int, ...]:
        """Keep the outer links planted while the middle link swings."""

        if link_count < 2:
            return tuple(range(link_count))
        del simulation_time_s
        swing_link = link_count // 2
        return tuple(link_id for link_id in range(link_count) if link_id != swing_link)

    def act(self, observation: LocalObservation) -> RobotAction:
        if self._start_time_s is None:
            self._start_time_s = observation.simulation_time_s
            self._start_positions_rad = observation.joint_positions_rad
        elapsed_s = max(0.0, observation.simulation_time_s - self._start_time_s)
        targets = self.targets(elapsed_s, len(observation.joint_positions_rad))
        blend = min(1.0, elapsed_s / max(self.startup_duration_s, 1e-9))
        blend = blend * blend * (3.0 - 2.0 * blend)
        return RobotAction(
            joint_targets_rad=tuple(
                start + blend * (target - start)
                for start, target in zip(self._start_positions_rad, targets)
            )
        )

    def reset(self) -> None:
        self._start_time_s = None
        self._start_positions_rad = ()
