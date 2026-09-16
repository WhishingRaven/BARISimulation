"""Environment construction metadata and privileged task metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .config import EnvironmentConfig, RobotConfig
from .types import GlobalRobotState


@dataclass
class EnvironmentEvaluator:
    config: EnvironmentConfig
    robot_config: RobotConfig
    crossed_robot_ids: set[int] = field(default_factory=set)

    def reset(self) -> None:
        self.crossed_robot_ids.clear()

    def update(self, states: Iterable[GlobalRobotState]) -> set[int]:
        newly_crossed: set[int] = set()
        for state in states:
            if state.robot_id in self.crossed_robot_ids:
                continue
            if self._has_crossed(state):
                self.crossed_robot_ids.add(state.robot_id)
                newly_crossed.add(state.robot_id)
        return newly_crossed

    def _has_crossed(self, state: GlobalRobotState) -> bool:
        if self.config.name == "gap":
            return state.position_m[0] > self.config.gap_width_m / 2.0 + 0.05
        if self.config.name == "step":
            obstacle_end = self.config.step_x_m + self.config.step_width_m
            return (
                state.position_m[0] > obstacle_end
                and state.position_m[2] > self.config.step_height_m * 0.75
            )
        return False

    @property
    def dimension_text(self) -> str:
        if self.config.name == "gap":
            return f"gap={self.config.gap_width_m:.3f} m, platform z={self.config.platform_height_m:.3f} m"
        if self.config.name == "step":
            return f"step={self.config.step_height_m:.3f} m high x {self.config.step_width_m:.3f} m wide"
        return "flat surface"
