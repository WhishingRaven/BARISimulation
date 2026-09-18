"""Task-agnostic cross-entropy method optimizer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FitnessEvaluation(Protocol):
    fitness: float
    components: Mapping[str, float]


CandidateEvaluator = Callable[[np.ndarray, int], FitnessEvaluation]


@dataclass(frozen=True)
class CEMSettings:
    generations: int = 5
    population_size: int = 8
    elite_fraction: float = 0.25
    initial_std: float = 0.75
    min_std: float = 0.05
    evaluation_episodes: int = 1
    seed: int = 7

    def validate(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be at least one")
        if self.population_size < 2:
            raise ValueError("population size must be at least two")
        if not 0.0 < self.elite_fraction <= 0.5:
            raise ValueError("elite fraction must be in (0, 0.5]")
        if self.initial_std <= 0.0:
            raise ValueError("initial standard deviation must be positive")
        if self.min_std <= 0.0:
            raise ValueError("minimum standard deviation must be positive")
        if self.min_std > self.initial_std:
            raise ValueError(
                "minimum standard deviation cannot exceed initial standard deviation"
            )
        if self.evaluation_episodes < 1:
            raise ValueError("evaluation episodes must be at least one")


@dataclass(frozen=True)
class CandidateFitness:
    fitness: float
    components: dict[str, float]


@dataclass(frozen=True)
class CEMGenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    elite_mean_fitness: float
    overall_best_fitness: float
    parameter_std: float
    best_components: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "elite_mean_fitness": self.elite_mean_fitness,
            "overall_best_fitness": self.overall_best_fitness,
            "parameter_std": self.parameter_std,
            "best_components": self.best_components,
        }


@dataclass(frozen=True)
class CEMResult:
    best_parameters: np.ndarray
    best_fitness: float
    best_components: dict[str, float]
    generations: tuple[CEMGenerationStats, ...]


ProgressCallback = Callable[[CEMGenerationStats], None]


def optimize(
    initial_mean: np.ndarray,
    evaluate: CandidateEvaluator,
    settings: CEMSettings,
    *,
    progress: ProgressCallback | None = None,
) -> CEMResult:
    """Maximize scalar fitness without knowing task or policy semantics."""

    settings.validate()
    mean = np.asarray(initial_mean, dtype=np.float64).copy()
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("initial mean must be a non-empty one-dimensional array")

    rng = np.random.default_rng(settings.seed)
    scale = np.full(mean.shape, settings.initial_std, dtype=np.float64)
    elite_count = max(
        1, int(np.ceil(settings.population_size * settings.elite_fraction))
    )
    best_parameters = mean.copy()
    best_evaluation = CandidateFitness(float("-inf"), {})
    generation_stats: list[CEMGenerationStats] = []

    for generation_index in range(settings.generations):
        population = rng.normal(
            mean,
            scale,
            size=(settings.population_size, mean.size),
        )
        population[0] = mean
        evaluations = [
            _evaluate_candidate(parameters, evaluate, settings.evaluation_episodes)
            for parameters in population
        ]
        scores = np.asarray(
            [evaluation.fitness for evaluation in evaluations], dtype=np.float64
        )
        ranking = np.argsort(-scores, kind="stable")
        elite_indices = ranking[:elite_count]
        elites = population[elite_indices]
        mean = np.mean(elites, axis=0)
        scale = np.maximum(np.std(elites, axis=0), settings.min_std)

        generation_best_index = int(ranking[0])
        generation_best = evaluations[generation_best_index]
        if generation_best.fitness > best_evaluation.fitness:
            best_evaluation = generation_best
            best_parameters = population[generation_best_index].copy()

        stats = CEMGenerationStats(
            generation=generation_index + 1,
            best_fitness=generation_best.fitness,
            mean_fitness=float(np.mean(scores)),
            elite_mean_fitness=float(np.mean(scores[elite_indices])),
            overall_best_fitness=best_evaluation.fitness,
            parameter_std=float(np.mean(scale)),
            best_components=dict(generation_best.components),
        )
        generation_stats.append(stats)
        if progress is not None:
            progress(stats)

    return CEMResult(
        best_parameters=best_parameters,
        best_fitness=best_evaluation.fitness,
        best_components=dict(best_evaluation.components),
        generations=tuple(generation_stats),
    )


def _evaluate_candidate(
    parameters: np.ndarray,
    evaluate: CandidateEvaluator,
    episode_count: int,
) -> CandidateFitness:
    fitness_sum = 0.0
    component_sums: dict[str, float] = {}
    component_names: set[str] | None = None
    for episode_index in range(episode_count):
        evaluation = evaluate(parameters, episode_index)
        if not np.isfinite(evaluation.fitness):
            raise ValueError("candidate fitness must be finite")
        current_names = set(evaluation.components)
        if component_names is None:
            component_names = current_names
        elif current_names != component_names:
            raise ValueError("fitness components must be consistent across episodes")
        fitness_sum += float(evaluation.fitness)
        for name, value in evaluation.components.items():
            if not np.isfinite(value):
                raise ValueError(f"fitness component {name!r} must be finite")
            component_sums[name] = component_sums.get(name, 0.0) + float(value)
    return CandidateFitness(
        fitness=fitness_sum / episode_count,
        components={
            name: value / episode_count for name, value in component_sums.items()
        },
    )
