"""Repeatable benchmarks and lightweight decentralized policy search."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .controllers import LinearLocalController, make_controller_factory
from .core import EpisodeResult, run_episode
from .tasks import make_task


@dataclass(frozen=True)
class Scenario:
    robot_count: int
    duration_s: float


SCENARIOS: Mapping[str, Scenario] = {
    "aggregation": Scenario(robot_count=12, duration_s=45.0),
    "coverage": Scenario(robot_count=16, duration_s=60.0),
    "transport": Scenario(robot_count=12, duration_s=35.0),
}


@dataclass(frozen=True)
class TrainingGeneration:
    generation: int
    best_score: float
    elite_mean_score: float
    population_mean_score: float
    parameter_std: float


@dataclass(frozen=True)
class TrainingResult:
    parameters: np.ndarray
    best_score: float
    history: tuple[TrainingGeneration, ...]


def benchmark(
    *,
    tasks: Sequence[str],
    controllers: Sequence[str],
    seeds: Iterable[int],
    learned_parameters: np.ndarray | None = None,
) -> list[EpisodeResult]:
    """Evaluate matched seeds across task/controller pairs."""

    results: list[EpisodeResult] = []
    seed_values = tuple(seeds)
    for task_name in tasks:
        try:
            scenario = SCENARIOS[task_name]
        except KeyError as error:
            raise ValueError(f"no scenario defaults for task {task_name!r}") from error
        for controller_name in controllers:
            if controller_name == "learned" and task_name != "aggregation":
                continue
            factory = make_controller_factory(
                task_name,
                controller_name,
                learned_parameters=learned_parameters,
            )
            for seed in seed_values:
                results.append(
                    run_episode(
                        make_task(task_name),
                        factory,
                        controller_name=controller_name,
                        robot_count=scenario.robot_count,
                        seed=seed,
                        duration_s=scenario.duration_s,
                    )
                )
    return results


def scaling_benchmark(
    *,
    task_name: str,
    controller_name: str,
    robot_counts: Sequence[int],
    seeds: Iterable[int],
    duration_s: float | None = None,
    learned_parameters: np.ndarray | None = None,
) -> list[EpisodeResult]:
    """Run a robot-count ablation with all other conditions fixed."""

    try:
        scenario = SCENARIOS[task_name]
    except KeyError as error:
        raise ValueError(f"no scenario defaults for task {task_name!r}") from error
    if any(count < 1 for count in robot_counts):
        raise ValueError("robot counts must be positive")
    factory = make_controller_factory(
        task_name,
        controller_name,
        learned_parameters=learned_parameters,
    )
    return [
        run_episode(
            make_task(task_name),
            factory,
            controller_name=controller_name,
            robot_count=robot_count,
            seed=seed,
            duration_s=duration_s or scenario.duration_s,
        )
        for robot_count in robot_counts
        for seed in seeds
    ]


def summarize(
    results: Sequence[EpisodeResult], *, by_robot_count: bool = False
) -> list[dict[str, float | str]]:
    grouped: dict[tuple[object, ...], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        key: tuple[object, ...] = (result.task, result.controller)
        if by_robot_count:
            key += (result.robot_count,)
        grouped[key].append(result)
    rows: list[dict[str, float | str]] = []
    for key, episodes in sorted(grouped.items()):
        task, controller = str(key[0]), str(key[1])
        row: dict[str, float | str] = {
            "task": task,
            "controller": controller,
            "episodes": float(len(episodes)),
            "success_rate": float(np.mean([episode.success for episode in episodes])),
            "mean_duration_s": float(
                np.mean([episode.duration_s for episode in episodes])
            ),
        }
        if by_robot_count:
            row["robot_count"] = float(episodes[0].robot_count)
        metric_names = sorted(
            set().union(*(episode.metrics.keys() for episode in episodes))
        )
        for metric_name in metric_names:
            values = np.asarray(
                [episode.metrics[metric_name] for episode in episodes],
                dtype=np.float64,
            )
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return rows


def train_aggregation_cem(
    *,
    seed: int = 2026,
    generations: int = 12,
    population_size: int = 24,
    elite_fraction: float = 0.20,
    evaluation_seeds: Sequence[int] = (101, 202, 303),
    robot_count: int = 12,
    duration_s: float = 32.0,
) -> TrainingResult:
    """Fit one shared linear policy by the cross-entropy method.

    Training may use privileged episode scores; execution remains decentralized
    because the resulting policy consumes only ``LocalSwarmObservation``.
    """

    if generations < 1 or population_size < 4:
        raise ValueError("training needs at least one generation and four candidates")
    if not 0.0 < elite_fraction < 1.0:
        raise ValueError("elite_fraction must be between zero and one")
    elite_count = max(2, int(np.ceil(population_size * elite_fraction)))
    rng = np.random.default_rng(seed)
    mean = np.zeros(LinearLocalController.PARAMETER_COUNT, dtype=np.float64)
    deviation = np.full_like(mean, 0.70)
    history: list[TrainingGeneration] = []
    best_parameters = mean.copy()
    best_score = float("-inf")

    for generation in range(generations):
        candidates = rng.normal(mean, deviation, (population_size, len(mean)))
        scores = np.asarray(
            [
                _aggregation_policy_score(
                    candidate,
                    seeds=evaluation_seeds,
                    robot_count=robot_count,
                    duration_s=duration_s,
                )
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        elite_indices = np.argsort(scores)[-elite_count:]
        elites = candidates[elite_indices]
        elite_scores = scores[elite_indices]
        generation_best_index = int(np.argmax(scores))
        if scores[generation_best_index] > best_score:
            best_score = float(scores[generation_best_index])
            best_parameters = candidates[generation_best_index].copy()
        # A small variance floor avoids prematurely freezing a mediocre turn rule.
        mean = elites.mean(axis=0)
        deviation = np.maximum(elites.std(axis=0), 0.045)
        history.append(
            TrainingGeneration(
                generation=generation,
                best_score=float(scores.max()),
                elite_mean_score=float(elite_scores.mean()),
                population_mean_score=float(scores.mean()),
                parameter_std=float(deviation.mean()),
            )
        )
    return TrainingResult(
        parameters=best_parameters,
        best_score=best_score,
        history=tuple(history),
    )


def _aggregation_policy_score(
    parameters: np.ndarray,
    *,
    seeds: Sequence[int],
    robot_count: int,
    duration_s: float,
) -> float:
    factory = make_controller_factory(
        "aggregation", "learned", learned_parameters=parameters
    )
    results = [
        run_episode(
            make_task("aggregation"),
            factory,
            controller_name="learned",
            robot_count=robot_count,
            seed=seed,
            duration_s=duration_s,
        )
        for seed in seeds
    ]
    scores = [
        result.metrics["aggregation_score"] + (0.25 if result.success else 0.0)
        for result in results
    ]
    return float(np.mean(scores))


def write_episode_csv(path: Path, results: Sequence[EpisodeResult]) -> None:
    metric_names = sorted(set().union(*(result.metrics for result in results)))
    fields = [
        "task",
        "controller",
        "seed",
        "robot_count",
        "steps",
        "duration_s",
        "success",
        *metric_names,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "task": result.task,
                    "controller": result.controller,
                    "seed": result.seed,
                    "robot_count": result.robot_count,
                    "steps": result.steps,
                    "duration_s": result.duration_s,
                    "success": int(result.success),
                    **result.metrics,
                }
            )


def write_summary_csv(
    path: Path, rows: Sequence[Mapping[str, float | str]]
) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    ordered = [
        name
        for name in (
            "task",
            "controller",
            "robot_count",
            "episodes",
            "success_rate",
        )
        if name in fields
    ]
    ordered.extend(name for name in fields if name not in ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_training_history(
    path: Path, history: Sequence[TrainingGeneration]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "generation",
        "best_score",
        "elite_mean_score",
        "population_mean_score",
        "parameter_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for generation in history:
            writer.writerow(
                {field: getattr(generation, field) for field in fields}
            )
