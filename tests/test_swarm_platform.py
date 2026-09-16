from dataclasses import fields
from math import cos, sin

import numpy as np

from bari_sim.swarm.controllers import (
    LinearLocalController,
    make_controller_factory,
)
from bari_sim.swarm.core import PlanarWorld, WorldConfig, run_episode
from bari_sim.swarm.experiments import SCENARIOS, summarize, train_aggregation_cem
from bari_sim.swarm.metrics import largest_component_fraction
from bari_sim.swarm.tasks import make_task
from bari_sim.swarm.types import LocalSwarmObservation, NeighborReading


def _episode(task_name: str, *, robot_count: int | None = None, seed: int = 1000):
    scenario = SCENARIOS[task_name]
    return run_episode(
        make_task(task_name),
        make_controller_factory(task_name, "rules"),
        controller_name="rules",
        robot_count=robot_count or scenario.robot_count,
        seed=seed,
        duration_s=scenario.duration_s,
    )


def test_policy_contract_contains_local_egocentric_information_only() -> None:
    assert {field.name for field in fields(LocalSwarmObservation)} == {
        "simulation_time_s",
        "neighbors",
        "boundary_ranges_m",
        "task_cue",
        "task_contact",
    }
    assert "robot_id" not in {field.name for field in fields(NeighborReading)}

    config = WorldConfig(width_m=20.0, height_m=20.0)
    first = PlanarWorld(config, 2)
    first.reset(np.asarray(((0.0, 0.0), (0.7, 0.2))), np.asarray((0.3, -0.4)))

    rotation_angle = 0.9
    rotation = np.asarray(
        (
            (cos(rotation_angle), -sin(rotation_angle)),
            (sin(rotation_angle), cos(rotation_angle)),
        )
    )
    transformed = PlanarWorld(config, 2)
    transformed.reset(
        first.positions @ rotation.T + np.asarray((2.0, -1.5)),
        first.headings + rotation_angle,
    )
    original_neighbor = first.local_observation(0, ((), ())).neighbors[0]
    transformed_neighbor = transformed.local_observation(0, ((), ())).neighbors[0]
    assert np.allclose(
        original_neighbor.relative_position_m,
        transformed_neighbor.relative_position_m,
    )
    assert np.isclose(
        original_neighbor.relative_heading_rad,
        transformed_neighbor.relative_heading_rad,
    )


def test_component_metric_distinguishes_clusters() -> None:
    positions = np.asarray(((0.0, 0.0), (0.2, 0.0), (2.0, 0.0)))
    assert largest_component_fraction(positions, 0.3) == 2.0 / 3.0
    assert largest_component_fraction(positions, 3.0) == 1.0


def test_seeded_episode_is_deterministic() -> None:
    first = _episode("aggregation", seed=41)
    second = _episode("aggregation", seed=41)
    assert first == second


def test_local_rules_solve_all_three_reference_tasks() -> None:
    for task_name in ("aggregation", "coverage", "transport"):
        result = _episode(task_name)
        assert result.success, (task_name, result.metrics)


def test_transport_is_physically_thresholded_above_one_robot() -> None:
    result = _episode("transport", robot_count=1)
    assert not result.success
    assert result.metrics["transport_progress"] == 0.0
    assert result.metrics["maximum_simultaneous_pushers"] <= 1.0


def test_lightweight_trainer_and_summary_contract() -> None:
    training = train_aggregation_cem(
        seed=9,
        generations=1,
        population_size=4,
        evaluation_seeds=(12,),
        robot_count=6,
        duration_s=1.0,
    )
    assert training.parameters.shape == (LinearLocalController.PARAMETER_COUNT,)
    assert np.isfinite(training.parameters).all()
    result = _episode("aggregation", seed=18)
    rows = summarize((result,))
    assert rows[0]["task"] == "aggregation"
    assert rows[0]["success_rate"] in {0.0, 1.0}
    scaling_rows = summarize((result,), by_robot_count=True)
    assert scaling_rows[0]["robot_count"] == result.robot_count
