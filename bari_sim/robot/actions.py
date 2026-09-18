"""Discrete action contract shared by teleoperation and learned policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class MotionAction(IntEnum):
    CURL_BODY = 0
    FLATTEN_BODY = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4


class LiftAction(IntEnum):
    LIFT_FRONT = 0
    UNLIFT_FRONT = 1
    STOP = 2


class GripAction(IntEnum):
    ATTACH = 0
    DETACH = 1
    STOP = 2


@dataclass(frozen=True)
class RobotAction:
    motion: MotionAction = MotionAction.STOP
    lift: LiftAction = LiftAction.UNLIFT_FRONT
    grip: GripAction = GripAction.DETACH

    @classmethod
    def from_indices(cls, motion: int, lift: int, grip: int) -> RobotAction:
        return cls(MotionAction(motion), LiftAction(lift), GripAction(grip))

    def as_indices(self) -> tuple[int, int, int]:
        return int(self.motion), int(self.lift), int(self.grip)
