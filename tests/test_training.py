from __future__ import annotations

import numpy as np
import pytest

from bari_sim.policies import LinearPolicy, PolicyMetadata
from bari_sim.simulation import SceneRequest, Simulation
from bari_sim.tasks import (
    CollisionAvoidanceObjective,
    ObjectiveResult,
    RobotGrid,
    TaskResult,
    task_definition,
)
from bari_sim.train.algorithms import CEMSettings, optimize
from bari_sim.workflows import run_inference


def collision_episode(
    *,
    success: bool = False,
    progress: float = 0.0,
    successful_fraction: float = 0.0,
    variance: float = 0.01,
    elapsed: float = 100.0,
    collisions: int = 0,
    flipped: int = 0,
) -> TaskResult:
    return TaskResult(
        task="collision-avoidance",
        difficulty=1,
        success=success,
        elapsed_time_s=elapsed,
        metrics={
            "robot_count": 10,
            "successful_robot_count": round(successful_fraction * 10),
            "successful_fraction": successful_fraction,
            "required_successful_robot_count": 10,
            "position_variance_m2": variance,
            "mean_position_variance_m2": variance,
            "collision_count": collisions,
            "flipped_immobile_robot_count": flipped,
            "progress_fraction": progress,
        },
    )


def fitness(result: TaskResult) -> float:
    return (
        CollisionAvoidanceObjective()
        .evaluate(result, episode_time_limit_s=100.0)
        .fitness
    )


def test_progress_beats_stationary_collision_free_group() -> None:
    moving_away = collision_episode(progress=-0.2)
    stationary = collision_episode(progress=0.0)
    moving = collision_episode(progress=0.5)
    assert fitness(moving_away) < fitness(stationary)
    assert fitness(moving) - fitness(stationary) >= 1.0


def test_success_beats_unsuccessful_collision_free_trajectory() -> None:
    successful = collision_episode(
        success=True,
        progress=0.9,
        successful_fraction=1.0,
        elapsed=70.0,
    )
    unsuccessful = collision_episode(progress=0.95, elapsed=100.0)
    assert fitness(successful) > fitness(unsuccessful)


@pytest.mark.parametrize(
    ("preferred", "penalized"),
    [
        (
            collision_episode(success=True, progress=1.0, collisions=0),
            collision_episode(success=True, progress=1.0, collisions=3),
        ),
        (
            collision_episode(success=True, progress=1.0, flipped=0),
            collision_episode(success=True, progress=1.0, flipped=2),
        ),
        (
            collision_episode(success=True, progress=1.0, variance=0.01),
            collision_episode(success=True, progress=1.0, variance=0.09),
        ),
        (
            collision_episode(success=True, progress=1.0, elapsed=30.0),
            collision_episode(success=True, progress=1.0, elapsed=90.0),
        ),
    ],
    ids=["fewer-collisions", "fewer-flipped", "more-cohesive", "faster"],
)
def test_objective_prefers_safer_cohesive_faster_success(
    preferred: TaskResult, penalized: TaskResult
) -> None:
    assert fitness(preferred) > fitness(penalized)


def test_failed_early_termination_gets_no_time_advantage() -> None:
    early_failure = collision_episode(progress=0.2, elapsed=5.0)
    full_failure = collision_episode(progress=0.2, elapsed=100.0)
    assert fitness(early_failure) == fitness(full_failure)


def test_collision_metric_counts_contact_transitions_not_contact_steps() -> None:
    task = task_definition("collision-avoidance", 1)
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat", task))
    assert simulation.evaluator is not None

    simulation.evaluator.update({(0, 1)})
    simulation.evaluator.update({(0, 1)})
    simulation.evaluator.update(set())
    simulation.evaluator.update({(0, 1)})

    assert simulation.evaluator.result().metrics["collision_count"] == 2


def test_reset_training_rollout_matches_fresh_inference_rollout() -> None:
    task = task_definition("collision-avoidance", 1)
    request = SceneRequest(RobotGrid(1, 1), "flat", task)
    policy = LinearPolicy.idle(PolicyMetadata(task.name.value, 1, "1*1", 7))
    fresh = run_inference(
        Simulation(request), policy, duration_s=0.5, viewer_enabled=False
    )
    reused_simulation = Simulation(request)
    reused_simulation.reset()
    reused = run_inference(
        reused_simulation, policy, duration_s=0.5, viewer_enabled=False
    )

    assert reused.metrics["progress_fraction"] == pytest.approx(
        fresh.metrics["progress_fraction"], abs=1e-12
    )
    assert reused.metrics["score"] == pytest.approx(fresh.metrics["score"], abs=1e-12)


def test_cem_is_seeded_generic_and_averages_episode_fitness() -> None:
    calls = 0

    def evaluate(parameters: np.ndarray, episode_index: int) -> ObjectiveResult:
        nonlocal calls
        calls += 1
        value = -float(np.sum(parameters**2)) + episode_index * 0.1
        return ObjectiveResult(value, {"total_fitness": value})

    settings = CEMSettings(
        generations=3,
        population_size=12,
        elite_fraction=0.25,
        initial_std=1.0,
        min_std=0.05,
        evaluation_episodes=2,
        seed=11,
    )
    first = optimize(np.asarray([3.0, -3.0]), evaluate, settings)
    second = optimize(np.asarray([3.0, -3.0]), evaluate, settings)

    assert calls == 2 * 3 * 12 * 2
    assert first.best_fitness > -18.0
    assert first.best_fitness == second.best_fitness
    assert first.best_parameters == pytest.approx(second.best_parameters)
    assert len(first.generations) == 3
