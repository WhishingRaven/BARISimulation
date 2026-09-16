"""Deterministic planar physics and episode execution.

The backend intentionally models only the dynamics needed by the task suite:
bounded unicycle motion, finite sensing, and robot-robot exclusion.  It runs
quickly enough for multi-seed controller comparisons and lightweight policy
search.  Task metrics may inspect global state; policies never can.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import cos, pi, sin
from typing import Protocol

import numpy as np

from .types import LocalSwarmObservation, NeighborReading, SwarmAction


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class WorldConfig:
    width_m: float = 6.0
    height_m: float = 4.0
    timestep_s: float = 0.1
    robot_radius_m: float = 0.10
    maximum_speed_m_s: float = 0.35
    maximum_turn_rate_rad_s: float = 2.4
    sensing_range_m: float = 1.15

    def validate(self) -> None:
        values = (
            self.width_m,
            self.height_m,
            self.timestep_s,
            self.robot_radius_m,
            self.maximum_speed_m_s,
            self.maximum_turn_rate_rad_s,
            self.sensing_range_m,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all world dimensions and limits must be positive")
        if 2.0 * self.robot_radius_m >= min(self.width_m, self.height_m):
            raise ValueError("robot is too large for the arena")


@dataclass(frozen=True)
class WorldStep:
    collision_pairs: tuple[tuple[int, int], ...]
    distance_traveled_m: tuple[float, ...]
    command_effort: tuple[float, ...]


class PlanarWorld:
    """Small unicycle world with a strict local observation constructor."""

    def __init__(self, config: WorldConfig, robot_count: int):
        config.validate()
        if robot_count < 1:
            raise ValueError("robot_count must be at least one")
        self.config = config
        self.robot_count = robot_count
        self.positions = np.zeros((robot_count, 2), dtype=np.float64)
        self.headings = np.zeros(robot_count, dtype=np.float64)
        self.time_s = 0.0

    def reset(self, positions: np.ndarray, headings_rad: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.float64)
        headings = np.asarray(headings_rad, dtype=np.float64)
        if positions.shape != (self.robot_count, 2):
            raise ValueError("positions must have shape (robot_count, 2)")
        if headings.shape != (self.robot_count,):
            raise ValueError("headings must have shape (robot_count,)")
        if not np.isfinite(positions).all() or not np.isfinite(headings).all():
            raise ValueError("initial state must be finite")
        self.positions[:] = positions
        self.headings[:] = wrap_angle(headings)
        self._clip_to_bounds()
        self.time_s = 0.0

    def local_observation(
        self,
        robot_index: int,
        broadcasts: Sequence[tuple[float, ...]],
        *,
        task_cue: tuple[float, ...] = (),
        task_contact: bool = False,
    ) -> LocalSwarmObservation:
        """Construct an egocentric observation without identity leakage."""

        if len(broadcasts) != self.robot_count:
            raise ValueError("one broadcast is required per robot")
        relative_world = self.positions - self.positions[robot_index]
        distances = np.linalg.norm(relative_world, axis=1)
        heading = self.headings[robot_index]
        c, s = cos(heading), sin(heading)
        world_to_body = np.asarray(((c, s), (-s, c)), dtype=np.float64)
        neighbors: list[NeighborReading] = []
        visible = np.flatnonzero(
            (distances > 1e-12) & (distances <= self.config.sensing_range_m)
        )
        for neighbor_index in visible:
            index = int(neighbor_index)
            local = world_to_body @ relative_world[index]
            neighbors.append(
                NeighborReading(
                    relative_position_m=(float(local[0]), float(local[1])),
                    distance_m=float(distances[index]),
                    relative_heading_rad=float(
                        wrap_angle(self.headings[index] - heading)
                    ),
                    broadcast=tuple(float(value) for value in broadcasts[index]),
                )
            )
        neighbors.sort(key=lambda reading: reading.distance_m)
        return LocalSwarmObservation(
            simulation_time_s=self.time_s,
            neighbors=tuple(neighbors),
            boundary_ranges_m=self._boundary_ranges(robot_index),
            task_cue=task_cue,
            task_contact=task_contact,
        )

    def world_vector_to_body(
        self, robot_index: int, vector_world: np.ndarray
    ) -> np.ndarray:
        heading = self.headings[robot_index]
        c, s = cos(heading), sin(heading)
        return np.asarray(
            (c * vector_world[0] + s * vector_world[1],
             -s * vector_world[0] + c * vector_world[1]),
            dtype=np.float64,
        )

    def step(self, actions: Sequence[SwarmAction]) -> WorldStep:
        if len(actions) != self.robot_count:
            raise ValueError("one action is required per robot")
        commands = tuple(action.clipped() for action in actions)
        old_positions = self.positions.copy()
        dt = self.config.timestep_s
        turn = np.asarray([action.turn_rate for action in commands])
        speed = np.asarray([action.forward_speed for action in commands])
        self.headings[:] = wrap_angle(
            self.headings + turn * self.config.maximum_turn_rate_rad_s * dt
        )
        directions = np.column_stack((np.cos(self.headings), np.sin(self.headings)))
        self.positions += (
            directions * speed[:, None] * self.config.maximum_speed_m_s * dt
        )
        self._clip_to_bounds()
        collisions = self._resolve_robot_collisions()
        self._clip_to_bounds()
        traveled = np.linalg.norm(self.positions - old_positions, axis=1)
        effort = np.abs(speed) + 0.25 * np.abs(turn)
        self.time_s += dt
        return WorldStep(
            collision_pairs=tuple(collisions),
            distance_traveled_m=tuple(float(value) for value in traveled),
            command_effort=tuple(float(value) for value in effort),
        )

    def _resolve_robot_collisions(self) -> list[tuple[int, int]]:
        minimum = 2.0 * self.config.robot_radius_m
        collisions: list[tuple[int, int]] = []
        for first in range(self.robot_count):
            for second in range(first + 1, self.robot_count):
                delta = self.positions[second] - self.positions[first]
                distance = float(np.linalg.norm(delta))
                if distance >= minimum:
                    continue
                collisions.append((first, second))
                if distance <= 1e-12:
                    angle = (first * 2.399963 + second * 0.754877) % (2.0 * pi)
                    normal = np.asarray((cos(angle), sin(angle)))
                else:
                    normal = delta / distance
                correction = 0.5 * (minimum - distance + 1e-9) * normal
                self.positions[first] -= correction
                self.positions[second] += correction
        return collisions

    def _clip_to_bounds(self) -> None:
        margin = self.config.robot_radius_m
        self.positions[:, 0] = np.clip(
            self.positions[:, 0], -self.config.width_m / 2.0 + margin,
            self.config.width_m / 2.0 - margin,
        )
        self.positions[:, 1] = np.clip(
            self.positions[:, 1], -self.config.height_m / 2.0 + margin,
            self.config.height_m / 2.0 - margin,
        )

    def _boundary_ranges(self, robot_index: int) -> tuple[float, float, float, float]:
        # Front, left, right, and rear rays in the robot body frame.
        heading = self.headings[robot_index]
        offsets = (0.0, pi / 2.0, -pi / 2.0, pi)
        return tuple(
            self._ray_to_boundary(self.positions[robot_index], heading + offset)
            for offset in offsets
        )

    def _ray_to_boundary(self, origin: np.ndarray, angle: float) -> float:
        direction = np.asarray((cos(angle), sin(angle)), dtype=np.float64)
        half_width = self.config.width_m / 2.0 - self.config.robot_radius_m
        half_height = self.config.height_m / 2.0 - self.config.robot_radius_m
        candidates: list[float] = []
        if direction[0] > 1e-12:
            candidates.append((half_width - origin[0]) / direction[0])
        elif direction[0] < -1e-12:
            candidates.append((-half_width - origin[0]) / direction[0])
        if direction[1] > 1e-12:
            candidates.append((half_height - origin[1]) / direction[1])
        elif direction[1] < -1e-12:
            candidates.append((-half_height - origin[1]) / direction[1])
        positive = [distance for distance in candidates if distance >= 0.0]
        return float(min(positive)) if positive else 0.0


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    controller: str
    seed: int
    robot_count: int
    steps: int
    duration_s: float
    success: bool
    metrics: Mapping[str, float]


class EpisodeTask(Protocol):
    name: str
    world_config: WorldConfig

    def reset(self, world: PlanarWorld, rng: np.random.Generator) -> None: ...

    def observation(
        self,
        world: PlanarWorld,
        robot_index: int,
        broadcasts: Sequence[tuple[float, ...]],
    ) -> LocalSwarmObservation: ...

    def after_step(
        self,
        world: PlanarWorld,
        actions: Sequence[SwarmAction],
        world_step: WorldStep,
    ) -> None: ...

    @property
    def complete(self) -> bool: ...

    def results(self, world: PlanarWorld) -> tuple[bool, Mapping[str, float]]: ...


class LocalController(Protocol):
    def reset(self, seed: int) -> None: ...

    def act(self, observation: LocalSwarmObservation) -> SwarmAction: ...


ControllerFactory = Callable[[int], LocalController]


def run_episode(
    task: EpisodeTask,
    controller_factory: ControllerFactory,
    *,
    controller_name: str,
    robot_count: int,
    seed: int,
    duration_s: float,
) -> EpisodeResult:
    """Run one synchronous decentralized episode.

    A controller factory receives only a private random seed. All controllers
    observe the same state snapshot, then act independently. Global task state
    is consulted only after actions have been selected.
    """

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    world = PlanarWorld(task.world_config, robot_count)
    task.reset(world, np.random.default_rng(seed))
    controllers = []
    for index in range(robot_count):
        controller_seed = seed + 1009 * (index + 1)
        controller = controller_factory(controller_seed)
        controller.reset(controller_seed)
        controllers.append(controller)
    broadcasts: list[tuple[float, ...]] = [()] * robot_count
    maximum_steps = int(np.ceil(duration_s / world.config.timestep_s))
    steps = 0
    while steps < maximum_steps and not task.complete:
        observations = tuple(
            task.observation(world, index, broadcasts)
            for index in range(robot_count)
        )
        actions = tuple(
            controller.act(observation).clipped()
            for controller, observation in zip(controllers, observations)
        )
        world_step = world.step(actions)
        task.after_step(world, actions, world_step)
        broadcasts = [action.broadcast for action in actions]
        steps += 1
    success, metrics = task.results(world)
    return EpisodeResult(
        task=task.name,
        controller=controller_name,
        seed=seed,
        robot_count=robot_count,
        steps=steps,
        duration_s=float(world.time_s),
        success=success,
        metrics=dict(metrics),
    )
