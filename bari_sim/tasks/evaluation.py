"""Privileged task completion checks and metrics (never policy observations)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..robot.specification import DEFAULT_ROBOT, RobotSpecification
from ..simulation.scene import LINK_NAMES, BuiltScene, body_name
from .catalog import TaskDefinition, TaskName


@dataclass(frozen=True)
class GlobalRobotState:
    robot_id: int
    position_m: tuple[float, float, float]
    upright: bool


@dataclass(frozen=True)
class TaskResult:
    task: str
    difficulty: int
    success: bool
    elapsed_time_s: float
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "difficulty": self.difficulty,
            "success": self.success,
            "elapsed_time_s": self.elapsed_time_s,
            "metrics": self.metrics,
        }


class TaskEvaluator:
    def __init__(
        self,
        task: TaskDefinition,
        scene: BuiltScene,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: RobotSpecification = DEFAULT_ROBOT,
    ):
        self.task = task
        self.scene = scene
        self.model = model
        self.data = data
        self.robot = robot
        self.robot_count = scene.request.grid.count
        self.body_ids = {
            (robot_id, link): model.body(body_name(robot_id, link)).id
            for robot_id in range(self.robot_count)
            for link in LINK_NAMES
        }
        initial_mean_position, _ = self._position_metrics(self.global_states())
        self.initial_mean_x_m = float(initial_mean_position[0])
        self.successful_robot_ids: set[int] = set()
        self.flipped_robot_ids: set[int] = set()
        self.active_collision_pairs: set[tuple[int, int]] = set()
        self.collision_count = 0
        self.position_variance_sum_m2 = 0.0
        self.trajectory_sample_count = 0
        self.completion_time_s: float | None = None

    @property
    def complete(self) -> bool:
        return len(self.successful_robot_ids) >= self.task.required_successes(
            self.robot_count
        )

    def update(self, collision_pairs: set[tuple[int, int]]) -> None:
        self.collision_count += len(collision_pairs - self.active_collision_pairs)
        self.active_collision_pairs = set(collision_pairs)
        states = self.global_states()
        _, variance = self._position_metrics(states)
        self.position_variance_sum_m2 += variance
        self.trajectory_sample_count += 1
        for state in states:
            if not state.upright:
                self.flipped_robot_ids.add(state.robot_id)
            if self._robot_succeeded(state):
                self.successful_robot_ids.add(state.robot_id)
        if self.complete and self.completion_time_s is None:
            self.completion_time_s = float(self.data.time)

    def global_states(self) -> tuple[GlobalRobotState, ...]:
        masses = self.robot.segment_masses_kg
        states: list[GlobalRobotState] = []
        for robot_id in range(self.robot_count):
            positions = np.asarray(
                [self.data.xpos[self.body_ids[(robot_id, link)]] for link in LINK_NAMES]
            )
            center = np.average(positions, axis=0, weights=masses)
            rear_rotation = np.asarray(
                self.data.xmat[self.body_ids[(robot_id, "rear")]]
            ).reshape(3, 3)
            states.append(
                GlobalRobotState(
                    robot_id=robot_id,
                    position_m=tuple(float(value) for value in center),
                    upright=bool(rear_rotation[2, 2] > 0.0),
                )
            )
        return tuple(states)

    def result(self) -> TaskResult:
        elapsed = (
            self.completion_time_s
            if self.completion_time_s is not None
            else float(self.data.time)
        )
        states = self.global_states()
        mean_position, variance = self._position_metrics(states)
        mean_trajectory_variance = (
            self.position_variance_sum_m2 / self.trajectory_sample_count
            if self.trajectory_sample_count
            else variance
        )
        success_fraction = len(self.successful_robot_ids) / self.robot_count
        common: dict[str, Any] = {
            "robot_count": self.robot_count,
            "successful_robot_count": len(self.successful_robot_ids),
            "successful_fraction": success_fraction,
            "required_successful_robot_count": self.task.required_successes(
                self.robot_count
            ),
            "position_variance_m2": variance,
            "mean_position_variance_m2": mean_trajectory_variance,
            "collision_count": self.collision_count,
            "flipped_immobile_robot_count": len(self.flipped_robot_ids),
            "mean_position_m": [float(mean_position[0]), float(mean_position[1])],
        }
        common[self.task.value_name + "_m"] = self.task.value
        if self.task.name is TaskName.COLLISION_AVOIDANCE:
            target_x = self.scene.target_x_m
            assert target_x is not None
            initial_distance = max(abs(target_x - self.initial_mean_x_m), 1e-9)
            remaining = abs(target_x - float(mean_position[0]))
            progress = (initial_distance - remaining) / initial_distance
            common.update(
                {
                    "target_x_m": target_x,
                    "initial_target_distance_m": initial_distance,
                    "distance_remaining_m": remaining,
                    "travel_time_s": elapsed,
                    "progress_fraction": progress,
                }
            )
        return TaskResult(
            task=self.task.name.value,
            difficulty=self.task.difficulty,
            success=self.complete,
            elapsed_time_s=float(elapsed),
            metrics=common,
        )

    @staticmethod
    def _position_metrics(
        states: tuple[GlobalRobotState, ...],
    ) -> tuple[np.ndarray, float]:
        positions = np.asarray([state.position_m[:2] for state in states])
        mean_position = np.mean(positions, axis=0)
        variance = float(np.mean(np.sum((positions - mean_position) ** 2, axis=1)))
        return mean_position, variance

    def _robot_succeeded(self, state: GlobalRobotState) -> bool:
        x, y, z = state.position_m
        if not state.upright:
            return False
        if abs(y) > self.scene.platform_width_m / 2.0:
            return False
        if self.task.name is TaskName.COLLISION_AVOIDANCE:
            assert self.scene.target_x_m is not None
            return abs(x - self.scene.target_x_m) <= 0.35 and z >= -self.robot.height_m
        if self.task.name is TaskName.GAP:
            assert self.scene.gap_width_m is not None
            # 받침 바닥 때문에 떨어진 로봇이 건너편 플랫폼 아래에 있을 수 있으므로
            # 플랫폼 높이에 있는지도 확인한다.
            return (
                x >= self.scene.gap_width_m / 2.0 + self.robot.length_m / 2.0
                and z >= -self.robot.height_m
            )
        assert self.scene.step_height_m is not None
        return (
            x >= 0.05
            and z >= self.scene.step_height_m + self.robot.height_m / 2.0 - 0.008
        )
