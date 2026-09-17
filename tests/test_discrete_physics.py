from __future__ import annotations

from math import atan2, pi

import mujoco
import numpy as np
import pytest

from bari_sim.robot import GripAction, LiftAction, MotionAction, RobotAction
from bari_sim.simulation import SceneRequest, Simulation
from bari_sim.tasks import RobotGrid


def _heading(simulation: Simulation, robot_id: int = 0) -> float:
    rotation = simulation.data.xmat[simulation.root_body_ids[robot_id]].reshape(3, 3)
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


def test_turn_uses_a_rear_cleat_and_yaw_motor_for_smooth_rotation() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    initial = _heading(simulation)
    positions: list[np.ndarray] = []

    def capture_frame(current: Simulation) -> bool:
        positions.append(current.data.xpos[current.root_body_ids[0], :2].copy())
        return True

    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    simulation.step(
        {0: RobotAction(MotionAction.TURN_LEFT)}, frame_callback=capture_frame
    )
    yaw_change_deg = (_heading(simulation) - initial) * 180.0 / pi
    rear_latch = simulation.model.equality("robot_0_gait_rear_latch").id
    assert simulation.data.eq_active[rear_latch]
    assert simulation.data.ctrl[simulation.turn_actuator_ids[0]] > 0.0
    assert yaw_change_deg > 0.3
    assert (
        max(np.linalg.norm(position - positions[0]) for position in positions) < 0.001
    )


def test_first_turn_is_settled_and_has_no_latch_impulse() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    initial = _heading(simulation)

    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    yaw_change_deg = (_heading(simulation) - initial) * 180.0 / pi
    assert yaw_change_deg == pytest.approx(0.3, abs=0.15)
    assert simulation.time_s == pytest.approx(0.5)


def test_turn_resumes_after_manual_stop() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    simulation.halt_motion()
    simulation.end_turn_sessions()
    heading = _heading(simulation)

    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    assert _heading(simulation) - heading > np.deg2rad(0.1)
    assert simulation.data.ctrl[simulation.turn_actuator_ids[0]] > 0.0


def test_held_turn_repeats_five_degree_increments_without_stopping() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    for _ in range(30):
        simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    assert _heading(simulation) * 180.0 / pi > 4.0
    assert simulation.data.ctrl[simulation.turn_actuator_ids[0]] > 0.0


def test_turn_after_curl_begins_physical_flattening() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    simulation.halt_motion()
    rear_before = simulation._joint_position(0, 0)
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    assert abs(simulation._joint_position(0, 0)) < abs(rear_before)


def test_held_turn_flattens_a_curled_rear_then_rotates() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    initial_heading = _heading(simulation)

    for _ in range(18):
        simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    assert abs(simulation._joint_position(0, 0)) < np.deg2rad(8.0)
    assert _heading(simulation) - initial_heading > np.deg2rad(0.5)


def test_turn_on_another_robot_latches_to_it_and_transfers_reaction() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    lower_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    upper_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[1]])
    simulation.data.qpos[upper_address : upper_address + 3] = simulation.data.qpos[
        lower_address : lower_address + 3
    ]
    simulation.data.qpos[upper_address + 2] = 0.012
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation.step({0: RobotAction(), 1: RobotAction()})
    lower_before = simulation.data.qpos[lower_address : lower_address + 2].copy()
    upper_before = _heading(simulation, 1)

    for _ in range(3):
        simulation.step({0: RobotAction(), 1: RobotAction(MotionAction.TURN_LEFT)})

    lower_rear_latch = simulation.model.equality("robot_1_gait_rear_on_0_rear").id
    assert simulation.data.eq_active[lower_rear_latch]
    assert _heading(simulation, 1) - upper_before > np.deg2rad(0.8)
    assert (
        np.linalg.norm(
            simulation.data.qpos[lower_address : lower_address + 2] - lower_before
        )
        > 0.0001
    )


def test_turn_on_step_top_uses_physical_rear_latch() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "step"))
    root_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    simulation.data.qpos[root_address : root_address + 3] = (0.12, 0.0, 0.0345)
    mujoco.mj_forward(simulation.model, simulation.data)
    for _ in range(simulation.physics_steps_per_control):
        mujoco.mj_step(simulation.model, simulation.data)
    simulation.data.qvel[:] = 0.0
    simulation.data.time = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation.halt_motion()
    origin = simulation.data.qpos[root_address : root_address + 2].copy()
    heading = _heading(simulation)

    for _ in range(3):
        simulation.halt_motion()
        simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    rear_latch = simulation.model.equality("robot_0_gait_rear_latch").id
    assert simulation.data.eq_active[rear_latch]
    assert _heading(simulation) - heading > np.deg2rad(0.4)
    assert (
        np.linalg.norm(simulation.data.qpos[root_address : root_address + 2] - origin)
        < 0.001
    )


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
