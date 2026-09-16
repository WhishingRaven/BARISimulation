"""Strict decentralized policy contracts for planar swarm experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class NeighborReading:
    """One locally sensed neighbor in the observing robot's body frame."""

    relative_position_m: tuple[float, float]
    distance_m: float
    relative_heading_rad: float
    broadcast: tuple[float, ...] = ()


@dataclass(frozen=True)
class LocalSwarmObservation:
    """Everything available to one controller.

    Absolute position, absolute heading, robot identifiers, other robots'
    actions, and task-global metrics are deliberately absent. ``task_cue`` is
    defined by each task and contains only egocentric sensor readings.
    """

    simulation_time_s: float
    neighbors: tuple[NeighborReading, ...]
    boundary_ranges_m: tuple[float, float, float, float]
    task_cue: tuple[float, ...] = ()
    task_contact: bool = False


@dataclass(frozen=True)
class SwarmAction:
    """Normalized unicycle command and an optional local broadcast."""

    forward_speed: float = 0.0
    turn_rate: float = 0.0
    broadcast: tuple[float, ...] = ()

    def clipped(self) -> SwarmAction:
        values = (self.forward_speed, self.turn_rate, *self.broadcast)
        if not all(isfinite(value) for value in values):
            raise ValueError("swarm actions must contain only finite values")
        return SwarmAction(
            forward_speed=max(-1.0, min(1.0, self.forward_speed)),
            turn_rate=max(-1.0, min(1.0, self.turn_rate)),
            broadcast=tuple(max(-1.0, min(1.0, value)) for value in self.broadcast),
        )
