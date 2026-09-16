"""Fast decentralized swarm experiments.

The planar backend complements, rather than replaces, the detailed MuJoCo
robot.  It is intended for controller/task iteration over many random seeds.
"""

from .core import EpisodeResult, PlanarWorld, WorldConfig
from .types import LocalSwarmObservation, NeighborReading, SwarmAction

__all__ = [
    "EpisodeResult",
    "LocalSwarmObservation",
    "NeighborReading",
    "PlanarWorld",
    "SwarmAction",
    "WorldConfig",
]
