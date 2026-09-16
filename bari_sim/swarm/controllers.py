"""Decentralized controllers using only :mod:`bari_sim.swarm.types`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import atan2, cos, pi, sin
from pathlib import Path

import numpy as np

from .types import LocalSwarmObservation, SwarmAction


def _wall_avoidance(
    observation: LocalSwarmObservation, threshold_m: float = 0.55
) -> np.ndarray:
    directions = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)))
    force = np.zeros(2, dtype=np.float64)
    for distance, ray_direction in zip(
        observation.boundary_ranges_m, directions
    ):
        if distance < threshold_m:
            force -= ray_direction * (threshold_m - distance) / threshold_m
    return force


def _separation(
    observation: LocalSwarmObservation, preferred_distance_m: float
) -> np.ndarray:
    force = np.zeros(2, dtype=np.float64)
    for neighbor in observation.neighbors:
        if neighbor.distance_m >= preferred_distance_m:
            continue
        vector = np.asarray(neighbor.relative_position_m, dtype=np.float64)
        force -= (
            vector
            / max(neighbor.distance_m, 1e-9)
            * (preferred_distance_m - neighbor.distance_m)
            / preferred_distance_m
        )
    return force


def _steer(vector_body: np.ndarray, speed: float = 0.8) -> SwarmAction:
    norm = float(np.linalg.norm(vector_body))
    if norm <= 1e-9:
        return SwarmAction()
    turn = float(np.clip(atan2(vector_body[1], vector_body[0]) / 1.25, -1.0, 1.0))
    forward = speed * max(0.12, 1.0 - 0.72 * abs(turn))
    return SwarmAction(forward_speed=forward, turn_rate=turn)


@dataclass
class AggregationController:
    """Local spring/alignment rule with deterministic private wandering."""

    preferred_distance_m: float = 0.46
    _phase: float = field(init=False, default=0.0)

    def reset(self, seed: int) -> None:
        self._phase = float(np.random.default_rng(seed).uniform(-pi, pi))

    def act(self, observation: LocalSwarmObservation) -> SwarmAction:
        wall = _wall_avoidance(observation)
        if not observation.neighbors:
            wander = np.asarray(
                (
                    1.0,
                    0.55 * sin(0.55 * observation.simulation_time_s + self._phase),
                )
            )
            return _steer(wander + 2.4 * wall, speed=0.72)

        spring = np.zeros(2, dtype=np.float64)
        alignment = np.zeros(2, dtype=np.float64)
        for neighbor in observation.neighbors:
            vector = np.asarray(neighbor.relative_position_m, dtype=np.float64)
            direction = vector / max(neighbor.distance_m, 1e-9)
            spring += direction * (neighbor.distance_m - self.preferred_distance_m)
            alignment += np.asarray(
                (cos(neighbor.relative_heading_rad), sin(neighbor.relative_heading_rad))
            )
        spring /= len(observation.neighbors)
        alignment /= len(observation.neighbors)
        close_separation = _separation(observation, self.preferred_distance_m)
        desired = (
            1.8 * spring
            + 2.2 * close_separation
            + 0.30 * alignment
            + 2.5 * wall
        )
        return _steer(desired, speed=0.74)


@dataclass
class CoverageController:
    """Repulsive local rule that disperses a compact swarm through an arena."""

    separation_range_m: float = 1.05
    _phase: float = field(init=False, default=0.0)

    def reset(self, seed: int) -> None:
        self._phase = float(np.random.default_rng(seed).uniform(-pi, pi))

    def act(self, observation: LocalSwarmObservation) -> SwarmAction:
        separation = _separation(observation, self.separation_range_m)
        wall = _wall_avoidance(observation, threshold_m=0.72)
        wander = np.asarray(
            (
                0.65,
                0.34 * sin(0.43 * observation.simulation_time_s + self._phase),
            ),
            dtype=np.float64,
        )
        desired = wander + 3.0 * separation + 3.0 * wall
        return _steer(desired, speed=0.82)


@dataclass
class TransportController:
    """Stage behind a sensed load, align to the beacon, then push."""

    staging_offset_m: float = 0.48
    _phase: float = field(init=False, default=0.0)

    def reset(self, seed: int) -> None:
        self._phase = float(np.random.default_rng(seed).uniform(-pi, pi))

    def act(self, observation: LocalSwarmObservation) -> SwarmAction:
        detected, object_x, object_y, goal_x, goal_y = observation.task_cue
        goal = np.asarray((goal_x, goal_y), dtype=np.float64)
        separation = _separation(observation, 0.34)
        wall = _wall_avoidance(observation)
        if observation.task_contact:
            desired = 2.8 * goal + 0.55 * separation + 2.5 * wall
            return _steer(desired, speed=1.0)
        if detected > 0.5:
            # Cue coordinates are normalized by sensor range. Direction is all
            # that matters; the offset keeps robots behind the load until they
            # are aligned rather than pushing from opposing sides.
            object_vector = np.asarray((object_x, object_y), dtype=np.float64)
            staging = object_vector - 0.31 * goal
            if np.linalg.norm(staging) < 0.12:
                staging = object_vector
            desired = 2.1 * staging + 1.6 * separation + 2.0 * wall
            return _steer(desired, speed=0.92)
        search = np.asarray(
            (goal_x, goal_y + 0.25 * sin(observation.simulation_time_s + self._phase))
        )
        return _steer(search + 2.0 * wall, speed=0.72)


@dataclass
class RandomWalkController:
    """Weak baseline: persistent random turn commands plus wall avoidance."""

    decision_interval_s: float = 1.2
    _rng: np.random.Generator = field(init=False)
    _next_decision_s: float = field(init=False, default=0.0)
    _turn: float = field(init=False, default=0.0)
    _speed: float = field(init=False, default=0.5)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._next_decision_s = 0.0
        self._turn = 0.0
        self._speed = 0.5

    def act(self, observation: LocalSwarmObservation) -> SwarmAction:
        if observation.simulation_time_s + 1e-12 >= self._next_decision_s:
            self._turn = float(self._rng.uniform(-0.85, 0.85))
            self._speed = float(self._rng.uniform(0.30, 0.85))
            self._next_decision_s += self.decision_interval_s
        wall = _wall_avoidance(observation)
        if np.linalg.norm(wall) > 0.05:
            return _steer(np.asarray((0.45, 0.0)) + 3.0 * wall, speed=self._speed)
        return SwarmAction(forward_speed=self._speed, turn_rate=self._turn)


@dataclass
class StationaryController:
    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: LocalSwarmObservation) -> SwarmAction:
        del observation
        return SwarmAction()


def local_features(observation: LocalSwarmObservation) -> np.ndarray:
    """Fixed-size local summary used by the lightweight learned controller."""

    cohesion = np.zeros(2, dtype=np.float64)
    separation = np.zeros(2, dtype=np.float64)
    alignment = np.zeros(2, dtype=np.float64)
    nearest = 1.0
    if observation.neighbors:
        for neighbor in observation.neighbors:
            vector = np.asarray(neighbor.relative_position_m, dtype=np.float64)
            cohesion += vector
            if neighbor.distance_m < 0.50:
                separation -= vector / max(neighbor.distance_m**2, 1e-9)
            alignment += np.asarray(
                (cos(neighbor.relative_heading_rad), sin(neighbor.relative_heading_rad))
            )
        cohesion /= len(observation.neighbors)
        alignment /= len(observation.neighbors)
        nearest = min(1.0, observation.neighbors[0].distance_m / 1.25)
    wall = _wall_avoidance(observation, threshold_m=0.75)
    return np.asarray(
        (
            1.0,
            cohesion[0],
            cohesion[1],
            separation[0],
            separation[1],
            alignment[0],
            alignment[1],
            wall[0],
            wall[1],
            min(1.0, len(observation.neighbors) / 6.0),
            nearest,
            1.0 if not observation.neighbors else 0.0,
        ),
        dtype=np.float64,
    )


@dataclass
class LinearLocalController:
    """Shared memoryless policy trainable by derivative-free search."""

    parameters: np.ndarray
    FEATURE_COUNT = 12
    PARAMETER_COUNT = 2 * FEATURE_COUNT

    def __post_init__(self) -> None:
        values = np.asarray(self.parameters, dtype=np.float64)
        if values.shape != (self.PARAMETER_COUNT,):
            raise ValueError(
                f"linear policy requires {self.PARAMETER_COUNT} parameters"
            )
        self.parameters = values.copy()

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: LocalSwarmObservation) -> SwarmAction:
        weights = self.parameters.reshape(2, self.FEATURE_COUNT)
        logits = weights @ local_features(observation)
        speed = 0.52 + 0.46 * np.tanh(logits[0])
        turn = np.tanh(logits[1])
        return SwarmAction(forward_speed=float(speed), turn_rate=float(turn))

    def save(self, path: Path, *, metadata: dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "controller": "linear_local",
            "feature_count": self.FEATURE_COUNT,
            "parameters": self.parameters.tolist(),
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinearLocalController:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("controller") != "linear_local":
            raise ValueError("unsupported controller artifact")
        if payload.get("feature_count") != cls.FEATURE_COUNT:
            raise ValueError("controller feature contract does not match this version")
        return cls(np.asarray(payload["parameters"], dtype=np.float64))


RULE_CONTROLLERS = {
    "aggregation": AggregationController,
    "coverage": CoverageController,
    "transport": TransportController,
}


def make_controller_factory(
    task_name: str,
    controller_name: str,
    *,
    learned_parameters: np.ndarray | None = None,
):
    controller_key = controller_name.lower()
    if controller_key == "rules":
        try:
            controller_type = RULE_CONTROLLERS[task_name.lower()]
        except KeyError as error:
            raise ValueError(f"no rule controller for task {task_name!r}") from error
        return lambda seed: controller_type()
    if controller_key == "random":
        return lambda seed: RandomWalkController()
    if controller_key == "stationary":
        return lambda seed: StationaryController()
    if controller_key == "learned":
        if learned_parameters is None:
            raise ValueError("learned controller requires policy parameters")
        parameters = np.asarray(learned_parameters, dtype=np.float64).copy()
        return lambda seed: LinearLocalController(parameters.copy())
    raise ValueError(
        f"unknown controller {controller_name!r}; choose rules, random, stationary, or learned"
    )
