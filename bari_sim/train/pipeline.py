"""Training orchestration between task objectives, policies, and algorithms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ..policies import LinearPolicy, PolicyMetadata
from ..simulation import SceneRequest, Simulation
from ..tasks import ObjectiveResult, RobotGrid, TaskDefinition, objective_for_task
from ..workflows.inference import run_inference
from .algorithms import CEMGenerationStats, CEMResult, CEMSettings, optimize

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TrainingSettings:
    generations: int = 5
    population_size: int = 8
    elite_fraction: float = 0.25
    duration_s: float = 60.0
    seed: int = 7
    initial_std: float = 0.75
    min_std: float = 0.05
    evaluation_episodes: int = 1

    def validate(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration must be positive")
        self.cem_settings().validate()

    def cem_settings(self) -> CEMSettings:
        return CEMSettings(
            generations=self.generations,
            population_size=self.population_size,
            elite_fraction=self.elite_fraction,
            initial_std=self.initial_std,
            min_std=self.min_std,
            evaluation_episodes=self.evaluation_episodes,
            seed=self.seed,
        )


@dataclass(frozen=True)
class TrainingSummary:
    model_path: Path
    algorithm: str
    best_score: float
    best_components: dict[str, float]
    generations: tuple[CEMGenerationStats, ...]

    @property
    def generation_best_scores(self) -> tuple[float, ...]:
        return tuple(generation.best_fitness for generation in self.generations)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": str(self.model_path.resolve()),
            "algorithm": self.algorithm,
            "best_score": self.best_score,
            "best_components": self.best_components,
            "generation_best_scores": self.generation_best_scores,
            "generations": [generation.as_dict() for generation in self.generations],
        }


def _run_cem(
    initial_parameters,
    evaluate,
    settings: TrainingSettings,
    progress,
) -> CEMResult:
    return optimize(
        initial_parameters,
        evaluate,
        settings.cem_settings(),
        progress=progress,
    )


_ALGORITHM_RUNNERS = {"cem": _run_cem}
SUPPORTED_ALGORITHMS = tuple(_ALGORITHM_RUNNERS)


def train_policy(
    grid: RobotGrid,
    task: TaskDefinition,
    output: Path,
    settings: TrainingSettings,
    *,
    algorithm: str = "cem",
    render: bool = False,
    progress: ProgressCallback | None = None,
) -> TrainingSummary:
    """Train one shared policy through a selected task-agnostic optimizer."""

    settings.validate()
    runner = _ALGORITHM_RUNNERS.get(algorithm)
    if runner is None:
        raise ValueError(f"unsupported training algorithm: {algorithm}")

    metadata = PolicyMetadata(
        task=task.name.value,
        difficulty=task.difficulty,
        robots=str(grid),
        seed=settings.seed,
    )
    initial_parameters = LinearPolicy.idle(metadata).parameters()
    simulation = Simulation(
        SceneRequest(grid=grid, environment=task.environment, task=task)
    )
    objective = objective_for_task(task)

    def evaluate(parameters, _episode_index: int) -> ObjectiveResult:
        simulation.reset()
        policy = LinearPolicy.from_parameters(parameters, metadata)
        result = run_inference(
            simulation,
            policy,
            duration_s=settings.duration_s,
            viewer_enabled=render,
        )
        return objective.evaluate(
            result,
            episode_time_limit_s=settings.duration_s,
        )

    def report(stats: CEMGenerationStats) -> None:
        if progress is None:
            return
        fields = {
            "G": f"{stats.generation}/{settings.generations}",
            "best_F": f"{stats.best_fitness:.3f}",
            "mean_F": f"{stats.mean_fitness:.3f}",
            "elite_mean_F": f"{stats.elite_mean_fitness:.3f}",
            "overall_best_F": f"{stats.overall_best_fitness:.3f}",
            "P_std": f"{stats.parameter_std:.3f}",
            **{
                {
                    "progress_reward": "progress_R",
                    "cohesion_penalty": "cohesion_P",
                    "time_penalty": "time_P",
                    "collision_penalty": "collision_P",
                    "flipped_penalty": "flipped_P",
                    "total_fitness": "total_F",
                }.get(name, name): f"{value:.3f}"
                for name, value in stats.best_components.items()
            },
        }
        headers = tuple(fields)
        widths = tuple(max(len(name), 10) for name in headers)
        if not generation_header_sent[0]:
            progress(" | ".join(f"{name:<{width}}" for name, width in zip(headers, widths)))
            generation_header_sent[0] = True
        progress(
            " | ".join(
                f"{value:>{width}}" for value, width in zip(fields.values(), widths)
            )
        )

    generation_header_sent = [False]

    # Task reward logic stays in tasks/objectives.py; runners only consume fitness.
    result = runner(
        initial_parameters,
        evaluate,
        settings,
        report,
    )
    trained_metadata = replace(metadata, training_score=result.best_fitness)
    LinearPolicy.from_parameters(result.best_parameters, trained_metadata).save(output)
    return TrainingSummary(
        model_path=output,
        algorithm=algorithm,
        best_score=result.best_fitness,
        best_components=result.best_components,
        generations=result.generations,
    )
