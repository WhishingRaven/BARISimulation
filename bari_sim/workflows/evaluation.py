"""Headless inference and evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..policies import LinearPolicy
from ..simulation import SceneRequest, Simulation
from ..tasks import RobotGrid, TaskDefinition, TaskResult
from .inference import run_inference

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: tuple[TaskResult, ...]

    @property
    def success_rate(self) -> float:
        return sum(result.success for result in self.episodes) / len(self.episodes)

    @property
    def mean_score(self) -> float:
        return sum(float(result.metrics["score"]) for result in self.episodes) / len(
            self.episodes
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_count": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_score": self.mean_score,
            "episodes": [result.as_dict() for result in self.episodes],
        }


def evaluate_policy(
    policy: LinearPolicy,
    grid: RobotGrid,
    task: TaskDefinition,
    *,
    duration_s: float,
    episodes: int = 1,
    render: bool = False,
    progress: ProgressCallback | None = None,
) -> EvaluationSummary:
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    simulation = Simulation(
        SceneRequest(grid=grid, environment=task.environment, task=task)
    )
    results: list[TaskResult] = []
    for episode in range(episodes):
        simulation.reset()
        result = run_inference(
            simulation,
            policy,
            duration_s=duration_s,
            viewer_enabled=render,
        )
        results.append(result)
        if progress is not None:
            progress(
                f"episode {episode + 1}/{episodes}: "
                f"score={float(result.metrics['score']):.3f}, success={result.success}"
            )
    return EvaluationSummary(tuple(results))
