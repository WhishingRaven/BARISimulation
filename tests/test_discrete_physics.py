from __future__ import annotations

from math import atan2, pi

import mujoco
import numpy as np
import pytest

from bari_sim.robot import GripAction, LiftAction, MotionAction, RobotAction
from bari_sim.simulation import SceneRequest, Simulation
from bari_sim.tasks import RobotGrid


def _heading(simulation: Simulation) -> float:
    rotation = simulation.data.xmat[simulation.root_body_ids[0]].reshape(3, 3)
    return atan2(rotation[1, 0], rotation[0, 0])


def test_public_step_is_half_a_second_and_actions_reach_posture() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    result = simulation.step(
        {
            0: RobotAction(
                MotionAction.CURL_BODY,
                LiftAction.LIFT_FRONT,
                GripAction.DETACH,
            )
        }
    )
    assert simulation.time_s == pytest.approx(1.0)
    assert result.observations[0].is_curled
    assert result.observations[0].is_front_lifted


def test_curl_flatten_cycle_moves_forward_from_contact_friction() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    initial_x = float(simulation.data.xpos[simulation.root_body_ids[0], 0])
    for _ in range(4):
        simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
        simulation.step({0: RobotAction(MotionAction.FLATTEN_BODY)})
    final_x = float(simulation.data.xpos[simulation.root_body_ids[0], 0])
    assert final_x - initial_x > 0.0


def test_alternating_cleats_limit_reverse_slip_during_wsws() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    positions: list[float] = []

    def capture_frame(current: Simulation) -> bool:
        positions.append(float(current.data.xpos[current.root_body_ids[0], 0]))
        return True

    for motion in (MotionAction.CURL_BODY, MotionAction.FLATTEN_BODY) * 4:
        simulation.step({0: RobotAction(motion)}, frame_callback=capture_frame)
    deltas = np.diff(positions)
    assert positions[-1] - positions[0] > 0.05
    assert float(deltas[deltas < 0.0].sum()) > -0.002


def test_gait_cleat_on_a_robot_latches_to_that_robot_not_world() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    lower_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    upper_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[1]])
    simulation.data.qpos[upper_address : upper_address + 3] = simulation.data.qpos[
        lower_address : lower_address + 3
    ]
    simulation.data.qpos[upper_address + 2] = 0.012
    mujoco.mj_forward(simulation.model, simulation.data)

    simulation.step({0: RobotAction(), 1: RobotAction(MotionAction.CURL_BODY)})

    world_latch = simulation.model.equality("robot_1_gait_front_latch").id
    lower_front_latch = simulation.model.equality("robot_1_gait_front_on_0_front").id
    assert not simulation.data.eq_active[world_latch]
    assert simulation.data.eq_active[lower_front_latch]


def test_repeating_an_already_reached_posture_does_not_creep() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    simulation.halt_motion()
    pose = simulation.data.qpos.copy()
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    assert simulation.data.qpos[:7] == pytest.approx(pose[:7])


def test_locking_unselected_root_prevents_previous_robot_drift() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    simulation.step({0: RobotAction(MotionAction.CURL_BODY), 1: RobotAction()})
    pose = simulation.data.qpos.copy()
    simulation.halt_motion()
    simulation.step(
        {0: RobotAction(), 1: RobotAction(MotionAction.CURL_BODY)},
        lock_root_motion=(0,),
    )
    assert simulation.data.qpos[:7] == pytest.approx(pose[:7])


def test_turn_rotates_whole_free_body_by_one_tuned_increment() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    initial = _heading(simulation)
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    assert (_heading(simulation) - initial) * 180.0 / pi == pytest.approx(8.0, abs=0.2)


def test_turn_after_curl_keeps_root_translation_in_place() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    simulation.halt_motion()
    pose = simulation.data.qpos.copy()
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    assert simulation.data.qpos[:3] == pytest.approx(pose[:3])


def test_manual_halt_clears_momentum_and_preserves_pose() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    pose = simulation.data.qpos.copy()
    simulation.halt_motion()
    assert simulation.data.qpos == pytest.approx(pose)
    assert simulation.data.qvel == pytest.approx(0.0)
    mujoco.mj_forward(simulation.model, simulation.data)
    assert simulation.data.qpos == pytest.approx(pose)


def test_lift_action_can_lock_free_body_pose() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    pose = simulation.data.qpos.copy()
    simulation.step(
        {0: RobotAction(lift=LiftAction.LIFT_FRONT)},
        lock_root_motion=(0,),
    )
    root = simulation.root_joint_ids[0]
    address = int(simulation.model.jnt_qposadr[root])
    assert simulation.data.qpos[address : address + 3] == pytest.approx(pose[:3])
    assert simulation.data.qvel[int(simulation.model.jnt_dofadr[root]) :][
        :3
    ] == pytest.approx(0.0)


def test_rear_attachment_holds_then_breaks_above_50_grams_force() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    attached = simulation.step({0: RobotAction(grip=GripAction.ATTACH)})
    assert attached.observations[0].is_attaching
    pose = simulation.data.qpos.copy()
    motion_while_attached = simulation.step(
        {0: RobotAction(MotionAction.CURL_BODY, grip=GripAction.ATTACH)}
    )
    assert motion_while_attached.observations[0].is_attaching
    assert simulation.data.qpos[:7] == pytest.approx(pose[:7])
    held = simulation.step(
        {0: RobotAction(grip=GripAction.ATTACH)},
        external_forces_n={0: (0.1, 0.0, 0.0)},
    )
    assert held.observations[0].is_attaching
    assert simulation.data.qpos[:7] == pytest.approx(pose[:7])
    overloaded = simulation.step(
        {0: RobotAction(grip=GripAction.ATTACH)}, external_forces_n={0: (1.0, 0.0, 0.0)}
    )
    assert not overloaded.observations[0].is_attaching
    assert overloaded.observations[0].strain_value == pytest.approx(100.0)
    assert simulation.data.qpos[0] > pose[0]
    assert any(
        event.event == "overload_detached" for event in overloaded.attachment_events
    )


def test_stable_curl_and_flatten_keep_floor_attachment_available() -> None:
    for motion in (MotionAction.CURL_BODY, MotionAction.FLATTEN_BODY):
        simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
        simulation.step({0: RobotAction()})
        simulation.step({0: RobotAction(motion)})
        assert simulation.observations()[0].is_possible_to_attach


def test_nearby_ids_are_refreshed_each_step() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    result = simulation.step({0: RobotAction(), 1: RobotAction()})
    assert result.observations[0].nearby_robot_ids == (1,)
    assert result.observations[1].nearby_robot_ids == (0,)
