"""Local edge/anchor heuristic with no bridge plan or global information."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import LocalObservation, RobotAction
from .gait import TravelingWaveGait


@dataclass
class BridgeBaselineController:
    gait: TravelingWaveGait = field(
        default_factory=lambda: TravelingWaveGait(amplitude_rad=0.38, frequency_hz=0.55)
    )
    edge_distance_threshold_m: float = 0.28
    structural_load_hold_ratio: float = 0.45

    def reset(self) -> None:
        self.gait.reset()

    def act(self, observation: LocalObservation) -> RobotAction:
        joint_count = len(observation.joint_positions_rad)
        neutral = (0.0,) * joint_count
        if observation.attachment.active:
            if observation.attachment.utilization >= self.structural_load_hold_ratio:
                return RobotAction(joint_targets_rad=neutral)
            extension = tuple(
                -0.12 if index % 2 == 0 else 0.12 for index in range(joint_count)
            )
            return RobotAction(joint_targets_rad=extension)

        down = observation.ranges["forward_down"]
        edge = not down.detected or down.distance_m > self.edge_distance_threshold_m
        if edge:
            reach = tuple(
                0.28 if index % 2 == 0 else -0.28 for index in range(joint_count)
            )
            return RobotAction(
                joint_targets_rad=reach, attach=observation.gripper_contact
            )

        if observation.locally_detected_robot_ids and observation.gripper_contact:
            return RobotAction(joint_targets_rad=neutral, attach=True)
        return self.gait.act(observation)
