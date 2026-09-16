from __future__ import annotations

import mujoco
import pytest

from bari_sim.cli import build_parser, main
from bari_sim.simulation import SceneBuilder, SceneRequest, Simulation
from bari_sim.tasks import (
    DIFFICULTY_VALUES,
    RobotGrid,
    TaskName,
    parse_robot_grid,
    task_definition,
)
from bari_sim.workflows.manual import (
    ManualController,
    ManualShortcutDefaults,
    manual_camera_pose,
    manual_overlay_text,
    manual_selection_overlay_text,
)


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


def test_flat_scene_includes_non_colliding_floor_grid() -> None:
    scene = SceneBuilder(SceneRequest(RobotGrid(1, 1), "flat")).build()
    assert 'name="floor_grid_x_+0"' in scene.xml
    assert 'name="floor_grid_y_+0"' in scene.xml
    assert 'contype="0"' in scene.xml
    assert 'conaffinity="0"' in scene.xml


def test_help_command_and_required_command_shapes(capsys) -> None:
    assert main(["help"]) == 0
    assert "collision-avoidance" in capsys.readouterr().out
    assert "quote it in zsh" in build_parser()._bari_subparsers["manual"].format_help()
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
    controller.handle_key("r")
    assert controller.actions()[0].motion.name == "STOP"
    controller.handle_key(" ")
    assert controller.take_pending_actions() is not None
    controller.handle_key("w")
    action = controller.actions()[0]
    assert action.motion.name == "CURL_BODY"
    assert action.lift.name == "LIFT_FRONT"
    assert action.grip.name == "ATTACH"
    assert controller.actions()[1].grip.name == "DETACH"
    assert controller.take_pending_actions()[0].motion.name == "CURL_BODY"
    assert controller.take_pending_actions() is None
    controller.refresh_held_motion({"W"})
    assert controller.take_pending_actions()[0].motion.name == "CURL_BODY"
    controller.refresh_held_motion(set())
    assert controller.take_halt_request()
    assert controller.take_pending_actions() is None
    controller.handle_key("n")
    controller.handle_key("d")
    assert controller.actions()[1].motion.name == "TURN_RIGHT"
    controller.handle_key("x")
    assert controller.actions()[1].motion.name == "STOP"


def test_manual_number_keys_select_robots_one_through_ten() -> None:
    controller = ManualController(10)
    for keycode, expected in ((49, 0), (57, 8), (48, 9), (321, 0), (320, 9)):
        controller.key_callback(keycode)
        controller.process_keys()
        assert controller.state.selected_robot_id == expected


def test_manual_overlay_shows_controls_and_active_action() -> None:
    controller = ManualController(1)
    controller.handle_key("w")
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    controls, status = manual_overlay_text(controller, simulation)
    assert "Hold W/S" in controls
    assert "0→R10" in manual_selection_overlay_text()
    assert "● CURL_BODY" in status
    assert "ROBOT 1" in status
    assert "STRAIN" in status
    assert "RANGE" in status
    assert "ATTACH" in status
    assert "NEARBY" in status


def test_manual_camera_tightly_frames_one_robot_and_scales_with_formation() -> None:
    single = manual_camera_pose(Simulation(SceneRequest(RobotGrid(1, 1), "flat")))
    swarm = manual_camera_pose(Simulation(SceneRequest(RobotGrid(6, 5), "flat")))
    assert single.lookat == pytest.approx((0.045, 0.0, 0.005))
    assert single.distance == pytest.approx(0.28)
    assert single.azimuth == 90.0
    assert single.elevation == -65.0
    assert swarm.distance > single.distance
    assert swarm.distance < 1.6


def test_robot_shortcuts_override_native_visualization_shortcuts() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))

    class Viewer:
        opt = mujoco.MjvOption()
        user_scn = mujoco.MjvScene(simulation.model, 100)

    viewer = Viewer()
    defaults = ManualShortcutDefaults.capture(viewer)
    viewer.opt.geomgroup[:] = 0
    viewer.opt.flags[:] = 1 - viewer.opt.flags
    viewer.user_scn.flags[:] = 1 - viewer.user_scn.flags
    defaults.restore(viewer)
    assert tuple(int(value) for value in viewer.opt.geomgroup) == defaults.geom_groups
    assert ManualShortcutDefaults.capture(viewer) == defaults
