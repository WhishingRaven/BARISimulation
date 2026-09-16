from __future__ import annotations

import mujoco
import pytest

from bari_sim.cli import build_parser, main
from bari_sim.simulation import SceneBuilder, SceneRequest
from bari_sim.tasks import (
    DIFFICULTY_VALUES,
    RobotGrid,
    TaskName,
    parse_robot_grid,
    task_definition,
)
from bari_sim.workflows.manual import ManualController


def test_robot_grid_parser() -> None:
    assert parse_robot_grid("2*5") == RobotGrid(2, 5)
    assert parse_robot_grid("4x5") == RobotGrid(4, 5)
    with pytest.raises(ValueError):
        parse_robot_grid("10")


def test_all_task_difficulties_build_valid_scenes() -> None:
    expected = {
        TaskName.COLLISION_AVOIDANCE: (2.0, 4.0, 6.0, 8.0, 10.0),
        TaskName.GAP: (0.10, 0.15, 0.20, 0.25, 0.30),
        TaskName.STEP: (0.03, 0.05, 0.08, 0.10, 0.15),
    }
    assert DIFFICULTY_VALUES == expected
    for task_name, values in expected.items():
        for difficulty, value in enumerate(values, start=1):
            task = task_definition(task_name, difficulty)
            assert task.value == value
            scene = SceneBuilder(
                SceneRequest(RobotGrid(1, 1), task.environment, task)
            ).build()
            mujoco.MjModel.from_xml_string(scene.xml)


def test_help_command_and_required_command_shapes(capsys) -> None:
    assert main(["help"]) == 0
    assert "collision-avoidance" in capsys.readouterr().out
    parser = build_parser()
    manual = parser.parse_args(["manual", "--robots", "1*1", "--environment", "flat"])
    assert manual.robots == RobotGrid(1, 1)
    train = parser.parse_args(
        [
            "train",
            "--robots",
            "2*5",
            "--task",
            "gap",
            "--difficulty",
            "3",
        ]
    )
    assert train.difficulty == 3


def test_manual_actions_latch_independently() -> None:
    controller = ManualController(2)
    controller.handle_key("w")
    controller.handle_key("r")
    controller.handle_key(" ")
    action = controller.actions()[0]
    assert action.motion.name == "CURL_BODY"
    assert action.lift.name == "LIFT_FRONT"
    assert action.grip.name == "ATTACH"
    assert controller.actions()[1].grip.name == "DETACH"
    controller.handle_key("n")
    controller.handle_key("d")
    assert controller.actions()[1].motion.name == "TURN_RIGHT"
