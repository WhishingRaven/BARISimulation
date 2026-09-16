"""Explicitly invoked lightweight policy training; never run implicitly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..policies import LinearPolicy, PolicyMetadata
from ..simulation import SceneRequest, Simulation
from ..tasks import RobotGrid, TaskDefinition


@dataclass(frozen=True)
class TrainingSettings:
    generations: int = 5
    population_size: int = 8
    elite_fraction: float = 0.25
    duration_s: float = 60.0
    seed: int = 7

    def validate(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be at least one")
        if self.population_size < 2:
            raise ValueError("population size must be at least two")
        if not 0.0 < self.elite_fraction <= 0.5:
            raise ValueError("elite fraction must be in (0, 0.5]")
        if self.duration_s <= 0.0:
            raise ValueError("duration must be positive")


@dataclass(frozen=True)
class TrainingSummary:
    model_path: Path
    best_score: float
    generation_best_scores: tuple[float, ...]


def train_policy(
    grid: RobotGrid,
    task: TaskDefinition,
    output: Path,
    settings: TrainingSettings,
) -> TrainingSummary:
    """Fit one shared linear policy with the cross-entropy method.

    This function is only reached by ``barisimulation train``.  Importing the
    package, manual control, inference, and evaluation never start training.
    """

    settings.validate()
    rng = np.random.default_rng(settings.seed)
    metadata = PolicyMetadata(
        task=task.name.value,
        difficulty=task.difficulty,
        robots=str(grid),
        seed=settings.seed,
    )
    idle = LinearPolicy.idle(metadata)
    mean = idle.parameters()
    scale = np.full(LinearPolicy.PARAMETER_COUNT, 0.75, dtype=np.float64)
    elite_count = max(1, round(settings.population_size * settings.elite_fraction))
    simulation = Simulation(
        SceneRequest(grid=grid, environment=task.environment, task=task)
    )
    best_parameters = mean.copy()
    best_score = float("-inf")
    generation_bests: list[float] = []

    for _generation in range(settings.generations):
        population = rng.normal(
            mean,
            scale,
            size=(settings.population_size, LinearPolicy.PARAMETER_COUNT),
        )
        population[0] = mean
        scores = np.empty(settings.population_size, dtype=np.float64)
        for candidate_index, parameters in enumerate(population):
            simulation.reset()
            policy = LinearPolicy.from_parameters(parameters, metadata)
            result = simulation.run(
                lambda _robot_id, observation, selected=policy: selected.act(
                    observation
                ),
                settings.duration_s,
            )
            assert result is not None
            scores[candidate_index] = float(result.metrics["score"])
        ranked = np.argsort(scores)[::-1]
        elites = population[ranked[:elite_count]]
        mean = np.mean(elites, axis=0)
        scale = np.maximum(np.std(elites, axis=0), 0.05)
        generation_best = float(scores[ranked[0]])
        generation_bests.append(generation_best)
        if generation_best > best_score:
            best_score = generation_best
            best_parameters = population[ranked[0]].copy()

    trained_metadata = replace(metadata, training_score=best_score)
    LinearPolicy.from_parameters(best_parameters, trained_metadata).save(output)
    return TrainingSummary(output, best_score, tuple(generation_bests))
