"""Task definitions with local observations and privileged evaluators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import cos, pi, sin

import numpy as np

from .core import PlanarWorld, WorldConfig, WorldStep
from .metrics import (
    grid_coverage_fraction,
    largest_component_fraction,
    mean_nearest_neighbor_distance_m,
    polarization,
    swarm_radius_m,
)
from .types import LocalSwarmObservation, SwarmAction


def _sample_positions(
    rng: np.random.Generator,
    count: int,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    minimum_distance_m: float,
) -> np.ndarray:
    positions: list[np.ndarray] = []
    for _ in range(20_000):
        candidate = np.asarray(
            (rng.uniform(*x_range), rng.uniform(*y_range)), dtype=np.float64
        )
        if all(
            np.linalg.norm(candidate - existing) >= minimum_distance_m
            for existing in positions
        ):
            positions.append(candidate)
            if len(positions) == count:
                return np.asarray(positions)
    raise RuntimeError("could not sample a non-overlapping initial swarm")


@dataclass
class _MetricState:
    steps: int = 0
    collision_robot_steps: int = 0
    total_effort: float = 0.0
    total_distance_m: float = 0.0

    def update(self, world_step: WorldStep) -> None:
        self.steps += 1
        collided = {robot for pair in world_step.collision_pairs for robot in pair}
        self.collision_robot_steps += len(collided)
        self.total_effort += sum(world_step.command_effort)
        self.total_distance_m += sum(world_step.distance_traveled_m)

    def common(self, world: PlanarWorld) -> dict[str, float]:
        denominator = max(1, self.steps * world.robot_count)
        return {
            "collision_rate": self.collision_robot_steps / denominator,
            "mean_command_effort": self.total_effort / denominator,
            "mean_distance_traveled_m": self.total_distance_m / world.robot_count,
        }


@dataclass
class AggregationTask:
    """Form one compact connected cluster using neighbor sensing only."""

    name: str = field(init=False, default="aggregation")
    world_config: WorldConfig = field(
        default_factory=lambda: WorldConfig(
            width_m=5.5,
            height_m=4.0,
            sensing_range_m=1.25,
        )
    )
    connectivity_range_m: float = 0.78
    success_radius_m: float = 0.86
    hold_steps: int = 20

    def reset(self, world: PlanarWorld, rng: np.random.Generator) -> None:
        positions = _sample_positions(
            rng,
            world.robot_count,
            x_range=(-2.15, 2.15),
            y_range=(-1.45, 1.45),
            minimum_distance_m=2.2 * world.config.robot_radius_m,
        )
        headings = rng.uniform(-pi, pi, world.robot_count)
        world.reset(positions, headings)
        self._metric_state = _MetricState()
        self._stable_steps = 0
        self._convergence_time_s: float | None = None
        self._complete = False

    def observation(
        self,
        world: PlanarWorld,
        robot_index: int,
        broadcasts: Sequence[tuple[float, ...]],
    ) -> LocalSwarmObservation:
        return world.local_observation(robot_index, broadcasts)

    def after_step(
        self,
        world: PlanarWorld,
        actions: Sequence[SwarmAction],
        world_step: WorldStep,
    ) -> None:
        del actions
        self._metric_state.update(world_step)
        connected = largest_component_fraction(
            world.positions, self.connectivity_range_m
        )
        radius = swarm_radius_m(world.positions)
        if connected >= 1.0 and radius <= self.success_radius_m:
            self._stable_steps += 1
            if self._convergence_time_s is None:
                self._convergence_time_s = world.time_s
        else:
            self._stable_steps = 0
            self._convergence_time_s = None
        self._complete = self._stable_steps >= self.hold_steps

    @property
    def complete(self) -> bool:
        return self._complete

    def results(self, world: PlanarWorld) -> tuple[bool, Mapping[str, float]]:
        component = largest_component_fraction(
            world.positions, self.connectivity_range_m
        )
        radius = swarm_radius_m(world.positions)
        success = self._complete
        score = (
            0.60 * component
            + 0.40 * np.exp(-radius / self.success_radius_m)
            - 0.15 * self._metric_state.common(world)["collision_rate"]
        )
        metrics = {
            **self._metric_state.common(world),
            "largest_component_fraction": component,
            "swarm_radius_m": radius,
            "polarization": polarization(world.headings),
            "convergence_time_s": (
                world.time_s
                if self._convergence_time_s is None
                else self._convergence_time_s
            ),
            "aggregation_score": float(score),
        }
        return success, metrics


@dataclass
class CoverageTask:
    """Spread sensing footprints through a bounded arena without a map."""

    name: str = field(init=False, default="coverage")
    world_config: WorldConfig = field(
        default_factory=lambda: WorldConfig(
            width_m=6.0,
            height_m=4.0,
            sensing_range_m=1.20,
        )
    )
    footprint_radius_m: float = 0.62
    success_coverage: float = 0.60
    minimum_spacing_m: float = 0.42
    hold_steps: int = 15

    def reset(self, world: PlanarWorld, rng: np.random.Generator) -> None:
        positions = _sample_positions(
            rng,
            world.robot_count,
            x_range=(-0.65, 0.65),
            y_range=(-0.65, 0.65),
            minimum_distance_m=2.15 * world.config.robot_radius_m,
        )
        headings = rng.uniform(-pi, pi, world.robot_count)
        world.reset(positions, headings)
        self._metric_state = _MetricState()
        self._stable_steps = 0
        self._complete = False
        self._peak_coverage = self._coverage(world)
        self._coverage_time_s: float | None = None

    def observation(
        self,
        world: PlanarWorld,
        robot_index: int,
        broadcasts: Sequence[tuple[float, ...]],
    ) -> LocalSwarmObservation:
        return world.local_observation(robot_index, broadcasts)

    def after_step(
        self,
        world: PlanarWorld,
        actions: Sequence[SwarmAction],
        world_step: WorldStep,
    ) -> None:
        del actions
        self._metric_state.update(world_step)
        coverage = self._coverage(world)
        spacing = mean_nearest_neighbor_distance_m(world.positions)
        self._peak_coverage = max(self._peak_coverage, coverage)
        if coverage >= self.success_coverage and spacing >= self.minimum_spacing_m:
            self._stable_steps += 1
            if self._coverage_time_s is None:
                self._coverage_time_s = world.time_s
        else:
            self._stable_steps = 0
            self._coverage_time_s = None
        self._complete = self._stable_steps >= self.hold_steps

    @property
    def complete(self) -> bool:
        return self._complete

    def results(self, world: PlanarWorld) -> tuple[bool, Mapping[str, float]]:
        coverage = self._coverage(world)
        spacing = mean_nearest_neighbor_distance_m(world.positions)
        metrics = {
            **self._metric_state.common(world),
            "coverage_fraction": coverage,
            "peak_coverage_fraction": self._peak_coverage,
            "mean_nearest_neighbor_distance_m": spacing,
            "coverage_time_s": (
                world.time_s
                if self._coverage_time_s is None
                else self._coverage_time_s
            ),
            "coverage_score": coverage - 0.20 * self._metric_state.common(world)[
                "collision_rate"
            ],
        }
        return self._complete, metrics

    def _coverage(self, world: PlanarWorld) -> float:
        return grid_coverage_fraction(
            world.positions,
            (world.config.width_m, world.config.height_m),
            self.footprint_radius_m,
        )


@dataclass
class TransportTask:
    """Move a heavy object that requires simultaneous local pushing."""

    name: str = field(init=False, default="transport")
    world_config: WorldConfig = field(
        default_factory=lambda: WorldConfig(
            width_m=7.0,
            height_m=4.0,
            sensing_range_m=1.10,
            maximum_speed_m_s=0.38,
        )
    )
    object_radius_m: float = 0.30
    object_sensor_range_m: float = 1.55
    required_pushers: int = 3
    required_forward_effort: float = 1.65
    object_speed_m_s: float = 0.20
    goal_tolerance_m: float = 0.35

    def reset(self, world: PlanarWorld, rng: np.random.Generator) -> None:
        self.object_position = np.asarray((-0.55, 0.0), dtype=np.float64)
        self.goal_position = np.asarray((2.45, 0.0), dtype=np.float64)
        positions = _sample_positions(
            rng,
            world.robot_count,
            x_range=(-1.75, -0.95),
            y_range=(-1.05, 1.05),
            minimum_distance_m=2.1 * world.config.robot_radius_m,
        )
        # Start approximately goal-facing but retain enough variation to make
        # local alignment and collision avoidance necessary.
        headings = rng.normal(0.0, 0.38, world.robot_count)
        world.reset(positions, headings)
        self._metric_state = _MetricState()
        self._initial_goal_distance_m = float(
            np.linalg.norm(self.goal_position - self.object_position)
        )
        self._complete = False
        self._success_time_s: float | None = None
        self._cooperative_steps = 0
        self._contact_robot_steps = 0
        self._aligned_push_sum = 0.0
        self._push_samples = 0
        self._maximum_pushers = 0

    def observation(
        self,
        world: PlanarWorld,
        robot_index: int,
        broadcasts: Sequence[tuple[float, ...]],
    ) -> LocalSwarmObservation:
        object_delta = self.object_position - world.positions[robot_index]
        object_distance = float(np.linalg.norm(object_delta))
        object_detected = object_distance <= self.object_sensor_range_m
        object_body = world.world_vector_to_body(robot_index, object_delta)
        goal_delta = self.goal_position - world.positions[robot_index]
        goal_norm = max(float(np.linalg.norm(goal_delta)), 1e-12)
        goal_body = world.world_vector_to_body(robot_index, goal_delta / goal_norm)
        scale = self.object_sensor_range_m
        cue = (
            1.0 if object_detected else 0.0,
            float(object_body[0] / scale) if object_detected else 0.0,
            float(object_body[1] / scale) if object_detected else 0.0,
            float(goal_body[0]),
            float(goal_body[1]),
        )
        contact_distance = (
            self.object_radius_m + world.config.robot_radius_m + 0.035
        )
        return world.local_observation(
            robot_index,
            broadcasts,
            task_cue=cue,
            task_contact=object_distance <= contact_distance,
        )

    def after_step(
        self,
        world: PlanarWorld,
        actions: Sequence[SwarmAction],
        world_step: WorldStep,
    ) -> None:
        self._metric_state.update(world_step)
        contact_distance = self.object_radius_m + world.config.robot_radius_m + 0.055
        goal_delta = self.goal_position - self.object_position
        goal_direction = goal_delta / max(float(np.linalg.norm(goal_delta)), 1e-12)
        pushers: list[int] = []
        contributions: list[float] = []
        for index, action in enumerate(actions):
            to_object = self.object_position - world.positions[index]
            distance = float(np.linalg.norm(to_object))
            if distance > contact_distance or action.forward_speed <= 0.05:
                continue
            direction = np.asarray(
                (cos(world.headings[index]), sin(world.headings[index])),
                dtype=np.float64,
            )
            approach = float(
                np.dot(direction, to_object / max(distance, 1e-12))
            )
            if approach <= 0.15:
                continue
            pushers.append(index)
            aligned = max(0.0, float(np.dot(direction, goal_direction)))
            contribution = action.forward_speed * aligned
            contributions.append(contribution)
            self._aligned_push_sum += aligned
            self._push_samples += 1

        self._maximum_pushers = max(self._maximum_pushers, len(pushers))
        self._contact_robot_steps += len(pushers)
        total_forward_effort = sum(contributions)
        if (
            len(pushers) >= self.required_pushers
            and total_forward_effort >= self.required_forward_effort
        ):
            self._cooperative_steps += 1
            normalized_effort = min(
                1.25, total_forward_effort / self.required_pushers
            )
            self.object_position += (
                goal_direction
                * self.object_speed_m_s
                * normalized_effort
                * world.config.timestep_s
            )

        self._exclude_robots_from_object(world)
        if np.linalg.norm(self.goal_position - self.object_position) <= self.goal_tolerance_m:
            self._complete = True
            if self._success_time_s is None:
                self._success_time_s = world.time_s

    @property
    def complete(self) -> bool:
        return self._complete

    def results(self, world: PlanarWorld) -> tuple[bool, Mapping[str, float]]:
        remaining = float(np.linalg.norm(self.goal_position - self.object_position))
        progress = float(
            np.clip(1.0 - remaining / self._initial_goal_distance_m, 0.0, 1.0)
        )
        step_count = max(1, self._metric_state.steps)
        metrics = {
            **self._metric_state.common(world),
            "transport_progress": progress,
            "object_goal_distance_m": remaining,
            "delivery_time_s": (
                world.time_s if self._success_time_s is None else self._success_time_s
            ),
            "maximum_simultaneous_pushers": float(self._maximum_pushers),
            "cooperative_push_fraction": self._cooperative_steps / step_count,
            "pusher_utilization": self._contact_robot_steps
            / (step_count * world.robot_count),
            "mean_push_alignment": (
                0.0
                if self._push_samples == 0
                else self._aligned_push_sum / self._push_samples
            ),
            "transport_score": progress
            - 0.10 * self._metric_state.common(world)["collision_rate"],
        }
        return self._complete, metrics

    def _exclude_robots_from_object(self, world: PlanarWorld) -> None:
        minimum = self.object_radius_m + world.config.robot_radius_m
        for index in range(world.robot_count):
            delta = world.positions[index] - self.object_position
            distance = float(np.linalg.norm(delta))
            if distance >= minimum:
                continue
            if distance <= 1e-12:
                angle = index * 2.399963
                normal = np.asarray((cos(angle), sin(angle)), dtype=np.float64)
            else:
                normal = delta / distance
            world.positions[index] = self.object_position + minimum * normal


TASKS = {
    "aggregation": AggregationTask,
    "coverage": CoverageTask,
    "transport": TransportTask,
}


def make_task(name: str):
    try:
        return TASKS[name.lower()]()
    except KeyError as error:
        choices = ", ".join(sorted(TASKS))
        raise ValueError(f"unknown swarm task {name!r}; choose {choices}") from error
