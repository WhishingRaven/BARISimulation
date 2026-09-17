from __future__ import annotations

from math import atan2, pi

import mujoco
import pytest

from bari_sim.cli import build_parser, main
from bari_sim.robot import GripAction, LiftAction, MotionAction, RobotAction
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
    _sync_robot_labels,
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


def test_manual_labels_include_strain_at_active_attachment() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    simulation.step({0: RobotAction(grip=GripAction.ATTACH)})
    viewer = type("Viewer", (), {})()
    viewer.user_scn = mujoco.MjvScene(simulation.model, 4)
    _sync_robot_labels(viewer, simulation)
    assert viewer.user_scn.ngeom == 2
    assert viewer.user_scn.geoms[1].label.startswith("strain: ")
    assert viewer.user_scn.geoms[1].label.endswith(" g")


def test_manual_strain_label_is_hidden_after_attachment_releases() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(grip=GripAction.ATTACH)})
    simulation.step({0: RobotAction(grip=GripAction.DETACH)})
    viewer = type("Viewer", (), {})()
    viewer.user_scn = mujoco.MjvScene(simulation.model, 4)
    _sync_robot_labels(viewer, simulation)
    assert viewer.user_scn.ngeom == 1


def test_manual_strain_is_hidden_after_overload_detach() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(grip=GripAction.ATTACH)})
    result = simulation.step(
        {0: RobotAction(grip=GripAction.ATTACH)},
        external_forces_n={0: (1.0, 0.0, 0.0)},
    )
    assert not result.observations[0].is_attaching
    assert any(event.event == "overload_detached" for event in result.attachment_events)
    viewer = type("Viewer", (), {})()
    viewer.user_scn = mujoco.MjvScene(simulation.model, 4)
    _sync_robot_labels(viewer, simulation)
    _, status = manual_overlay_text(ManualController(1), simulation)

    assert viewer.user_scn.ngeom == 1
    assert "STRAIN" not in status


def test_manual_overlay_separates_grip_action_from_attachment_observation() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(grip=GripAction.ATTACH)})
    simulation.step(
        {0: RobotAction(grip=GripAction.ATTACH)},
        external_forces_n={0: (1.0, 0.0, 0.0)},
    )
    controller = ManualController(1)
    controller.handle_key(" ")
    controller.take_pending_actions()

    _, status = manual_overlay_text(controller, simulation)

    assert "ACTION" in status
    assert "GRIP    ● ATTACH" in status
    assert "OBSERVATION" in status
    assert "ATTACH  possible=" in status
    assert "active=0" in status

    controller.mark_step_complete()
    _, status_after_step = manual_overlay_text(controller, simulation)
    assert "MOTION  ● STOP" in status_after_step
    assert "LIFT    ● STOP" in status_after_step
    assert "GRIP    ● STOP" in status_after_step


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


def test_manual_actions_reset_to_neutral_after_each_step() -> None:
    controller = ManualController(2)
    controller.handle_key("r")
    assert controller.actions()[0].motion.name == "STOP"
    controller.handle_key(" ")
    assert controller.take_pending_actions() is not None
    controller.handle_key("w")
    action = controller.actions()[0]
    assert action.motion.name == "CURL_BODY"
    assert action.lift is LiftAction.STOP
    assert action.grip is GripAction.STOP
    assert controller.actions()[1].grip is GripAction.STOP
    dispatched = controller.take_pending_actions()[0]
    assert dispatched.motion.name == "CURL_BODY"
    assert dispatched.lift is LiftAction.STOP
    assert dispatched.grip is GripAction.STOP
    assert controller.take_pending_actions() is None
    controller.refresh_held_motion({"W"})
    assert controller.take_pending_actions()[0].motion.name == "CURL_BODY"
    controller.refresh_held_motion(set())
    controller.refresh_held_motion(set())
    controller.refresh_held_motion(set())
    assert controller.take_halt_request()
    assert controller.take_pending_actions() is None
    controller.handle_key("n")
    controller.handle_key("d")
    assert controller.actions()[1].motion.name == "TURN_RIGHT"
    controller.handle_key("x")
    assert controller.actions()[1].motion.name == "STOP"


def test_manual_turn_keydown_runs_one_segment_then_stops_at_completion() -> None:
    controller = ManualController(1)
    controller.handle_key("a")
    assert controller.take_pending_actions() is not None

    # A/D repeats policy steps without depending on platform held-key polling.
    assert controller.take_pending_actions()[0].motion is MotionAction.TURN_LEFT
    controller.mark_turn_complete(0)
    assert controller.take_pending_actions() is None
    assert controller.actions()[0].motion is MotionAction.STOP

    # A new keydown starts the next independent five-degree segment.
    controller.handle_key("a")
    assert controller.take_pending_actions()[0].motion is MotionAction.TURN_LEFT
    controller.handle_key("c")
    assert controller.take_pending_actions() is None


def test_manual_new_action_preempts_an_in_progress_turn() -> None:
    controller = ManualController(1)
    controller.handle_key("a")
    assert controller.take_pending_actions()[0].motion is MotionAction.TURN_LEFT
    assert not controller.has_pending_action()

    controller.handle_key("r")

    assert controller.has_pending_action()
    action = controller.take_pending_actions()[0]
    assert action.motion is MotionAction.STOP
    assert action.lift is LiftAction.LIFT_FRONT


def test_one_manual_turn_keydown_physically_stops_at_five_degrees() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    controller = ManualController(1)
    rotation = simulation.data.xmat[simulation.root_body_ids[0]].reshape(3, 3)
    initial_heading = atan2(rotation[1, 0], rotation[0, 0])
    controller.handle_key("a")

    for _ in range(60):
        actions = controller.take_pending_actions()
        assert actions is not None
        simulation.step(actions)
        if simulation.turn_is_settled(0):
            simulation.end_turn_sessions()
            controller.mark_turn_complete(0)
            break
    else:
        pytest.fail("manual turn did not settle")

    rotation = simulation.data.xmat[simulation.root_body_ids[0]].reshape(3, 3)
    final_heading = atan2(rotation[1, 0], rotation[0, 0])
    assert (final_heading - initial_heading) * 180.0 / pi == pytest.approx(
        5.0, abs=0.15
    )
    assert controller.take_pending_actions() is None
    assert controller.actions()[0].motion is MotionAction.STOP


def test_manual_held_motion_ignores_one_false_poll_but_stops_on_release() -> None:
    controller = ManualController(1)
    controller.handle_key("w")
    assert controller.take_pending_actions() is not None

    controller.refresh_held_motion(set())
    controller.refresh_held_motion({"W"})
    assert controller.take_pending_actions()[0].motion is MotionAction.CURL_BODY
    assert not controller.take_halt_request()

    controller.refresh_held_motion(set())
    controller.refresh_held_motion(set())
    controller.refresh_held_motion(set())
    assert controller.take_halt_request()


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
    assert "ACTION" in status
    assert "OBSERVATION" in status
    assert "ROBOT 1" in status
    assert "STRAIN" not in status
    assert "RANGE" in status
    assert "ATTACH" in status
    assert "NEARBY" in status


def test_manual_camera_tightly_frames_one_robot_and_scales_with_formation() -> None:
    single = manual_camera_pose(Simulation(SceneRequest(RobotGrid(1, 1), "flat")))
    swarm = manual_camera_pose(Simulation(SceneRequest(RobotGrid(6, 5), "flat")))
    assert single.lookat == pytest.approx((0.045, 0.0, 0.0025), abs=1e-6)
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
