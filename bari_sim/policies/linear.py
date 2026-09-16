"""Small shared policy over only the declared local observation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..robot.actions import GripAction, LiftAction, MotionAction, RobotAction
from ..robot.observation import RobotObservation
from ..robot.specification import DEFAULT_ROBOT

FEATURE_NAMES = (
    "bias",
    "front_proximity",
    "down_proximity",
    "left_proximity",
    "right_proximity",
    "nearby_robot_fraction",
    "strain_fraction",
    "is_curled",
    "is_front_lifted",
    "is_possible_to_attach",
    "is_attaching",
    "is_detached",
)


@dataclass(frozen=True)
class PolicyMetadata:
    task: str
    difficulty: int
    robots: str
    seed: int
    training_score: float | None = None


class LinearPolicy:
    """A deterministic shared multi-head discrete policy.

    It intentionally has no access to global pose, target coordinates, or maps.
    """

    VERSION = 1
    PARAMETER_COUNT = (len(MotionAction) + len(LiftAction) + len(GripAction)) * len(
        FEATURE_NAMES
    )

    def __init__(
        self,
        motion_weights: np.ndarray,
        lift_weights: np.ndarray,
        grip_weights: np.ndarray,
        metadata: PolicyMetadata,
    ):
        feature_count = len(FEATURE_NAMES)
        expected = (
            (len(MotionAction), feature_count),
            (len(LiftAction), feature_count),
            (len(GripAction), feature_count),
        )
        actual = (motion_weights.shape, lift_weights.shape, grip_weights.shape)
        if actual != expected:
            raise ValueError(f"policy weight shapes must be {expected}, got {actual}")
        self.motion_weights = np.asarray(motion_weights, dtype=np.float64)
        self.lift_weights = np.asarray(lift_weights, dtype=np.float64)
        self.grip_weights = np.asarray(grip_weights, dtype=np.float64)
        self.metadata = metadata

    def act(self, observation: RobotObservation) -> RobotAction:
        features = observation_features(observation)
        return RobotAction.from_indices(
            int(np.argmax(self.motion_weights @ features)),
            int(np.argmax(self.lift_weights @ features)),
            int(np.argmax(self.grip_weights @ features)),
        )

    def parameters(self) -> np.ndarray:
        return np.concatenate(
            (
                self.motion_weights.ravel(),
                self.lift_weights.ravel(),
                self.grip_weights.ravel(),
            )
        )

    @classmethod
    def from_parameters(
        cls, parameters: np.ndarray, metadata: PolicyMetadata
    ) -> LinearPolicy:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (cls.PARAMETER_COUNT,):
            raise ValueError(f"expected {cls.PARAMETER_COUNT} policy parameters")
        feature_count = len(FEATURE_NAMES)
        motion_end = len(MotionAction) * feature_count
        lift_end = motion_end + len(LiftAction) * feature_count
        return cls(
            values[:motion_end].reshape(len(MotionAction), feature_count),
            values[motion_end:lift_end].reshape(len(LiftAction), feature_count),
            values[lift_end:].reshape(len(GripAction), feature_count),
            metadata,
        )

    @classmethod
    def idle(cls, metadata: PolicyMetadata) -> LinearPolicy:
        parameters = np.zeros(cls.PARAMETER_COUNT, dtype=np.float64)
        policy = cls.from_parameters(parameters, metadata)
        policy.motion_weights[int(MotionAction.STOP), 0] = 1.0
        policy.lift_weights[int(LiftAction.UNLIFT_FRONT), 0] = 1.0
        policy.grip_weights[int(GripAction.DETACH), 0] = 1.0
        return policy

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "format": "barisimulation-linear-policy",
            "version": self.VERSION,
            "feature_names": list(FEATURE_NAMES),
            "metadata": {
                "task": self.metadata.task,
                "difficulty": self.metadata.difficulty,
                "robots": self.metadata.robots,
                "seed": self.metadata.seed,
                "training_score": self.metadata.training_score,
            },
            "weights": {
                "motion": self.motion_weights.tolist(),
                "lift": self.lift_weights.tolist(),
                "grip": self.grip_weights.tolist(),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinearPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "barisimulation-linear-policy":
            raise ValueError("unsupported policy file")
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported policy version")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("policy observation features do not match this build")
        metadata = PolicyMetadata(**payload["metadata"])
        weights = payload["weights"]
        return cls(
            np.asarray(weights["motion"], dtype=np.float64),
            np.asarray(weights["lift"], dtype=np.float64),
            np.asarray(weights["grip"], dtype=np.float64),
            metadata,
        )


def observation_features(observation: RobotObservation) -> np.ndarray:
    maximum_range = DEFAULT_ROBOT.sensor_range_m
    proximity = [
        1.0 - min(max(distance, 0.0), maximum_range) / maximum_range
        for distance in observation.distances
    ]
    return np.asarray(
        (
            1.0,
            *proximity,
            min(len(observation.nearby_robot_ids) / 29.0, 1.0),
            min(
                max(observation.strain_value / DEFAULT_ROBOT.maximum_strain_g, 0.0), 1.0
            ),
            float(observation.is_curled),
            float(observation.is_front_lifted),
            float(observation.is_possible_to_attach),
            float(observation.is_attaching),
            float(observation.is_detached),
        ),
        dtype=np.float64,
    )
