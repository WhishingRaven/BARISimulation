"""Contact/attachment-driven inchworm hypothesis.

Only the front end has a gripper. Rear anchoring therefore comes from ordinary
contact/friction; failed cycles remain visible experimental results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ..types import LocalObservation, RobotAction


class InchwormPhase(Enum):
    SEEK_FRONT_CONTACT = auto()
    ESTABLISH_FRONT_ANCHOR = auto()
    CONTRACT = auto()
    REAR_SETTLE = auto()
    RELEASE_FRONT = auto()
    EXTEND = auto()


@dataclass
class InchwormController:
    bend_amplitude_rad: float = 0.58
    seek_duration_s: float = 0.60
    attach_wait_s: float = 0.12
    contract_duration_s: float = 0.65
    settle_duration_s: float = 0.18
    extend_duration_s: float = 0.65

    def __post_init__(self) -> None:
        self.phase = InchwormPhase.SEEK_FRONT_CONTACT
        self.phase_start_s: float | None = None

    def reset(self) -> None:
        self.phase = InchwormPhase.SEEK_FRONT_CONTACT
        self.phase_start_s = None

    def act(self, observation: LocalObservation) -> RobotAction:
        if self.phase_start_s is None:
            self.phase_start_s = observation.simulation_time_s
        elapsed = observation.simulation_time_s - self.phase_start_s
        joint_count = len(observation.joint_positions_rad)
        neutral = (0.0,) * joint_count
        bent = tuple(
            self.bend_amplitude_rad * (1.0 if joint_id % 2 == 0 else -1.0)
            for joint_id in range(joint_count)
        )

        if self.phase is InchwormPhase.SEEK_FRONT_CONTACT:
            if observation.gripper_contact:
                self._transition(
                    InchwormPhase.ESTABLISH_FRONT_ANCHOR, observation.simulation_time_s
                )
                return RobotAction(joint_targets_rad=neutral, attach=True)
            if elapsed >= self.seek_duration_s:
                self.phase_start_s = observation.simulation_time_s
            # Small alternating reach motion, not artificial translation.
            sign = (
                1.0
                if int(elapsed / max(self.seek_duration_s / 2.0, 1e-6)) % 2 == 0
                else -1.0
            )
            return RobotAction(
                joint_targets_rad=tuple(sign * 0.25 * value for value in bent)
            )

        if self.phase is InchwormPhase.ESTABLISH_FRONT_ANCHOR:
            if observation.attachment.active:
                self._transition(InchwormPhase.CONTRACT, observation.simulation_time_s)
            elif elapsed >= self.attach_wait_s:
                self._transition(
                    InchwormPhase.SEEK_FRONT_CONTACT, observation.simulation_time_s
                )
            return RobotAction(joint_targets_rad=neutral)

        if self.phase is InchwormPhase.CONTRACT:
            if not observation.attachment.active:
                self._transition(
                    InchwormPhase.SEEK_FRONT_CONTACT, observation.simulation_time_s
                )
            elif elapsed >= self.contract_duration_s:
                self._transition(
                    InchwormPhase.REAR_SETTLE, observation.simulation_time_s
                )
            return RobotAction(joint_targets_rad=bent)

        if self.phase is InchwormPhase.REAR_SETTLE:
            if elapsed >= self.settle_duration_s:
                self._transition(
                    InchwormPhase.RELEASE_FRONT, observation.simulation_time_s
                )
            return RobotAction(joint_targets_rad=bent)

        if self.phase is InchwormPhase.RELEASE_FRONT:
            self._transition(InchwormPhase.EXTEND, observation.simulation_time_s)
            return RobotAction(joint_targets_rad=bent, detach=True)

        if elapsed >= self.extend_duration_s:
            self._transition(
                InchwormPhase.SEEK_FRONT_CONTACT, observation.simulation_time_s
            )
        return RobotAction(joint_targets_rad=neutral)

    def _transition(self, phase: InchwormPhase, simulation_time_s: float) -> None:
        self.phase = phase
        self.phase_start_s = simulation_time_s
