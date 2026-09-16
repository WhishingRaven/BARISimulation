"""Policy-facing and diagnostic data contracts.

LocalObservation intentionally contains no world position, world orientation, or
map. GlobalRobotState is a separate simulator/evaluation contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForceReading:
    force_n: float = 0.0


@dataclass(frozen=True)
class RangeReading:
    distance_m: float
    detected: bool
    hit_robot_id: int | None = None


@dataclass(frozen=True)
class AttachmentObservation:
    active: bool = False
    force_n: float = 0.0
    moment_nm: float = 0.0
    utilization: float = 0.0
    target_is_robot: bool = False


@dataclass(frozen=True)
class LocalObservation:
    """Information available to one decentralized policy."""

    robot_id: int
    simulation_time_s: float
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]
    force_sensors: Mapping[str, ForceReading]
    ranges: Mapping[str, RangeReading]
    attachment: AttachmentObservation
    gripper_contact: bool
    locally_detected_robot_ids: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotAction:
    """Low-level action suitable for a future multi-agent wrapper."""

    joint_targets_rad: tuple[float, ...] | None = None
    joint_torques_nm: tuple[float, ...] | None = None
    attach: bool = False
    detach: bool = False

    def __post_init__(self) -> None:
        if self.joint_targets_rad is not None and self.joint_torques_nm is not None:
            raise ValueError("action cannot command position and torque simultaneously")
        if self.attach and self.detach:
            raise ValueError("action cannot attach and detach simultaneously")


@dataclass(frozen=True)
class GlobalRobotState:
    """Privileged state for logging, evaluation, and visualization only."""

    robot_id: int
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity_m_s: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]


@dataclass(frozen=True)
class AttachmentEvent:
    simulation_time_s: float
    event: str
    source_robot_id: int
    target_robot_id: int | None
    target_body: str
    force_n: float = 0.0
    moment_nm: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class StepResult:
    observations: Mapping[int, LocalObservation]
    attachment_events: tuple[AttachmentEvent, ...] = field(default_factory=tuple)
