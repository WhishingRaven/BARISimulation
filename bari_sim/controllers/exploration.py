"""Intentionally simple local exploration/obstacle/gap baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import LocalObservation, RobotAction
from .gait import TravelingWaveGait


@dataclass
class ExplorationController:
    gait: TravelingWaveGait = field(default_factory=TravelingWaveGait)
    gap_distance_threshold_m: float = 0.24
    obstacle_distance_threshold_m: float = 0.11
    obstacle_contact_threshold_n: float = 0.12
    blocked_hold_time_s: float = 1.5
    allow_local_attachment: bool = True

    def __post_init__(self) -> None:
        self._blocked_since_s: float | None = None

    def reset(self) -> None:
        self._blocked_since_s = None
        self.gait.reset()

    def act(self, observation: LocalObservation) -> RobotAction:
        joint_count = len(observation.joint_positions_rad)
        neutral = (0.0,) * joint_count
        if observation.attachment.active and observation.attachment.utilization >= 0.65:
            return RobotAction(joint_targets_rad=neutral)

        forward_down = observation.ranges["forward_down"]
        gap_ahead = (
            not forward_down.detected
            or forward_down.distance_m > self.gap_distance_threshold_m
        )
        if gap_ahead:
            if observation.gripper_contact and self.allow_local_attachment:
                return RobotAction(joint_targets_rad=neutral, attach=True)
            search = tuple(
                0.42 * (1.0 if index % 2 == 0 else -1.0) for index in range(joint_count)
            )
            return RobotAction(joint_targets_rad=search)

        forward = observation.ranges["forward"]
        surface_force = sum(
            reading.force_n for reading in observation.force_sensors.values()
        )
        obstacle = (
            forward.detected and forward.distance_m < self.obstacle_distance_threshold_m
        ) or surface_force > self.obstacle_contact_threshold_n
        if obstacle:
            if self._blocked_since_s is None:
                self._blocked_since_s = observation.simulation_time_s
            blocked_time = observation.simulation_time_s - self._blocked_since_s
            bend = tuple(
                0.62 * (1.0 if index % 2 == 0 else -1.0) for index in range(joint_count)
            )
            if blocked_time >= self.blocked_hold_time_s:
                return RobotAction(
                    joint_targets_rad=bend,
                    attach=observation.gripper_contact and self.allow_local_attachment,
                )
            return RobotAction(joint_targets_rad=bend)

        self._blocked_since_s = None
        if (
            observation.locally_detected_robot_ids
            and observation.gripper_contact
            and self.allow_local_attachment
        ):
            return RobotAction(
                joint_targets_rad=self.gait.targets(
                    observation.simulation_time_s, joint_count
                ),
                attach=True,
            )
        return self.gait.act(observation)
