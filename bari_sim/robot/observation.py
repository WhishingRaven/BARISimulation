"""The complete information boundary visible to one robot policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RobotObservation:
    nearby_robot_ids: tuple[int, ...]
    strain_value: float
    distance1: float
    distance2: float
    distance3: float
    distance4: float
    is_curled: bool
    is_front_lifted: bool
    is_possible_to_attach: bool
    is_attaching: bool
    is_detached: bool

    @property
    def distances(self) -> tuple[float, float, float, float]:
        """Front, down, left, and right ultrasonic distances, in metres."""

        return self.distance1, self.distance2, self.distance3, self.distance4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
