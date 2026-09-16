"""The three task definitions and their five fixed difficulty levels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import ceil


class TaskName(StrEnum):
    COLLISION_AVOIDANCE = "collision-avoidance"
    GAP = "gap"
    STEP = "step"


DIFFICULTY_VALUES: dict[TaskName, tuple[float, ...]] = {
    TaskName.COLLISION_AVOIDANCE: (2.0, 4.0, 6.0, 8.0, 10.0),
    TaskName.GAP: (0.10, 0.15, 0.20, 0.25, 0.30),
    TaskName.STEP: (0.03, 0.05, 0.08, 0.10, 0.15),
}


@dataclass(frozen=True)
class RobotGrid:
    rows: int
    columns: int

    @property
    def count(self) -> int:
        return self.rows * self.columns

    def __str__(self) -> str:
        return f"{self.rows}*{self.columns}"


def parse_robot_grid(value: str) -> RobotGrid:
    normalized = value.lower().replace("x", "*")
    parts = normalized.split("*")
    if len(parts) != 2:
        raise ValueError("robot layout must use m*n form, for example 2*5")
    try:
        rows, columns = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("robot layout dimensions must be integers") from error
    if rows < 1 or columns < 1:
        raise ValueError("robot layout dimensions must be positive")
    return RobotGrid(rows, columns)


@dataclass(frozen=True)
class TaskDefinition:
    name: TaskName
    difficulty: int
    value: float
    value_name: str
    unit: str

    @property
    def environment(self) -> str:
        return "flat" if self.name is TaskName.COLLISION_AVOIDANCE else self.name.value

    def required_successes(self, robot_count: int) -> int:
        if self.name is TaskName.COLLISION_AVOIDANCE:
            return robot_count
        return ceil(0.8 * robot_count)


def task_definition(task: str | TaskName, difficulty: int) -> TaskDefinition:
    name = TaskName(task)
    if difficulty not in range(1, 6):
        raise ValueError("difficulty must be between 1 and 5")
    names = {
        TaskName.COLLISION_AVOIDANCE: ("target_distance", "m"),
        TaskName.GAP: ("gap_width", "m"),
        TaskName.STEP: ("step_height", "m"),
    }
    value_name, unit = names[name]
    return TaskDefinition(
        name=name,
        difficulty=difficulty,
        value=DIFFICULTY_VALUES[name][difficulty - 1],
        value_name=value_name,
        unit=unit,
    )
