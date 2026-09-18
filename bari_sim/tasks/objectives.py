"""Task-owned conversion from episode metrics to scalar fitness."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .catalog import TaskDefinition, TaskName
from .evaluation import TaskResult


@dataclass(frozen=True)
class ObjectiveResult:
    """Scalar fitness and its diagnostic, weighted components."""

    fitness: float
    components: dict[str, float]


class TaskObjective(Protocol):
    def evaluate(
        self, result: TaskResult, *, episode_time_limit_s: float
    ) -> ObjectiveResult: ...


@dataclass(frozen=True)
class CollisionAvoidanceObjectiveConfig:
    """Weights and normalization scales for collision-avoidance fitness."""

    progress_weight: float = 3.0
    success_weight: float = 5.0
    partial_success_scale: float = 0.25
    cohesion_weight: float = 1.0
    variance_scale_m2: float = 0.10
    time_weight: float = 0.5
    collision_weight: float = 2.0
    collision_events_per_robot_scale: float = 1.0
    flipped_weight: float = 3.0

    def validate(self) -> None:
        values = (
            self.progress_weight,
            self.success_weight,
            self.partial_success_scale,
            self.cohesion_weight,
            self.variance_scale_m2,
            self.time_weight,
            self.collision_weight,
            self.collision_events_per_robot_scale,
            self.flipped_weight,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("objective weights and scales must be non-negative")
        if self.variance_scale_m2 == 0.0:
            raise ValueError("variance scale must be positive")
        if self.collision_events_per_robot_scale == 0.0:
            raise ValueError("collision event scale must be positive")


class CollisionAvoidanceObjective:
    def __init__(self, config: CollisionAvoidanceObjectiveConfig | None = None) -> None:
        self.config = config or CollisionAvoidanceObjectiveConfig()
        self.config.validate()

    def evaluate(
        self, result: TaskResult, *, episode_time_limit_s: float
    ) -> ObjectiveResult:
        if episode_time_limit_s <= 0.0:
            raise ValueError("episode time limit must be positive")
        metrics = result.metrics
        robot_count = max(1, int(metrics["robot_count"]))
        progress = float(np.clip(metrics["progress_fraction"], -1.0, 1.0))
        successful_fraction = float(np.clip(metrics["successful_fraction"], 0.0, 1.0))
        success_value = (
            1.0
            if result.success
            else self.config.partial_success_scale * successful_fraction
        )
        mean_variance = float(
            metrics.get("mean_position_variance_m2", metrics["position_variance_m2"])
        )
        normalized_variance = float(
            np.clip(mean_variance / self.config.variance_scale_m2, 0.0, 1.0)
        )
        # Failure always pays the full time penalty. This prevents a crash or
        # flip from looking like a fast completion.
        normalized_time = (
            min(result.elapsed_time_s / episode_time_limit_s, 1.0)
            if result.success
            else 1.0
        )
        collisions_per_robot = float(metrics["collision_count"]) / robot_count
        normalized_collisions = float(
            np.clip(
                collisions_per_robot / self.config.collision_events_per_robot_scale,
                0.0,
                1.0,
            )
        )
        normalized_flipped = float(
            np.clip(
                float(metrics["flipped_immobile_robot_count"]) / robot_count,
                0.0,
                1.0,
            )
        )

        components = {
            "progress_reward": self.config.progress_weight * progress,
            "success_bonus": self.config.success_weight * success_value,
            "cohesion_penalty": self.config.cohesion_weight * normalized_variance,
            "time_penalty": self.config.time_weight * normalized_time,
            "collision_penalty": self.config.collision_weight * normalized_collisions,
            "flipped_penalty": self.config.flipped_weight * normalized_flipped,
        }
        fitness = (
            components["progress_reward"]
            + components["success_bonus"]
            - components["cohesion_penalty"]
            - components["time_penalty"]
            - components["collision_penalty"]
            - components["flipped_penalty"]
        )
        components["total_fitness"] = fitness
        return ObjectiveResult(fitness=fitness, components=components)


class StandardTaskObjective:
    """Compatibility objective for gap and step until task-specific ones exist."""

    def evaluate(
        self, result: TaskResult, *, episode_time_limit_s: float
    ) -> ObjectiveResult:
        del episode_time_limit_s
        metrics = result.metrics
        robot_count = max(1, int(metrics["robot_count"]))
        fitness = (
            float(metrics["successful_fraction"])
            - float(metrics["collision_count"]) / robot_count * 0.005
            - float(metrics["flipped_immobile_robot_count"]) / robot_count * 0.20
        )
        return ObjectiveResult(
            fitness=fitness,
            components={"total_fitness": fitness},
        )


def objective_for_task(
    task: TaskDefinition,
    *,
    collision_avoidance_config: CollisionAvoidanceObjectiveConfig | None = None,
) -> TaskObjective:
    if task.name is TaskName.COLLISION_AVOIDANCE:
        return CollisionAvoidanceObjective(collision_avoidance_config)
    return StandardTaskObjective()


def score_task_result(
    result: TaskResult,
    task: TaskDefinition,
    *,
    episode_time_limit_s: float,
    collision_avoidance_config: CollisionAvoidanceObjectiveConfig | None = None,
) -> TaskResult:
    """Attach shared evaluation/training fitness to a completed episode."""

    objective = objective_for_task(
        task, collision_avoidance_config=collision_avoidance_config
    )
    evaluation = objective.evaluate(result, episode_time_limit_s=episode_time_limit_s)
    return replace(
        result,
        metrics={
            **result.metrics,
            **evaluation.components,
            "score": evaluation.fitness,
        },
    )
