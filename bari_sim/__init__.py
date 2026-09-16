"""BARI articulated swarm research simulator."""

from .config import ExperimentConfig, load_experiment_config
from .types import LocalObservation, RobotAction

__all__ = [
    "ExperimentConfig",
    "LocalObservation",
    "RobotAction",
    "load_experiment_config",
]

__version__ = "0.1.0"
