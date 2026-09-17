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


def _finish_turn(
    simulation: Simulation,
    actions: dict[int, RobotAction],
    *,
    robot_id: int = 0,
    headings: list[float] | None = None,
    torques: list[float] | None = None,
    positions: list[np.ndarray] | None = None,
    maximum_steps: int = 30,
) -> int:
    def capture_frame(current: Simulation) -> bool:
        if headings is not None:
            headings.append(_heading(current, robot_id))
        if torques is not None:
            torques.append(float(current.data.ctrl[current.turn_actuator_ids[robot_id]]))
        if positions is not None:
            positions.append(current.data.xpos[current.root_body_ids[robot_id], :2].copy())
        return True

    for step_count in range(1, maximum_steps + 1):
        simulation.step(
            actions,
            frame_callback=capture_frame,
            render_hz=1.0 / simulation.timestep_s,
        )
        if simulation.turn_is_settled(robot_id):
            return step_count
    pytest.fail(f"robot {robot_id} did not settle on its yaw target")


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


def test_halting_unselected_robots_preserves_selected_turn_momentum() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    simulation.step(
        {0: RobotAction(MotionAction.TURN_LEFT), 1: RobotAction()}
    )
    selected_dof = int(simulation.model.jnt_dofadr[simulation.root_joint_ids[0]])
    unselected_dof = int(simulation.model.jnt_dofadr[simulation.root_joint_ids[1]])
    selected_velocity = simulation.data.qvel[selected_dof : selected_dof + 6].copy()

    simulation.halt_robots((1,))

    assert simulation.data.qvel[selected_dof : selected_dof + 6] == pytest.approx(
        selected_velocity
    )
    assert simulation.data.qvel[unselected_dof : unselected_dof + 6] == pytest.approx(
        0.0
    )


def test_turn_command_sets_yaw_torque_without_writing_the_root_pose() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    pose = simulation.data.qpos.copy()

    simulation._set_action(0, RobotAction(MotionAction.TURN_LEFT))
    simulation.gait_anchors.apply_motion(0, MotionAction.TURN_LEFT)
    simulation._apply_joint_control()

    rear_latch = simulation.model.equality("robot_0_gait_rear_latch").id
    front_latch = simulation.model.equality("robot_0_gait_front_latch").id
    assert simulation.data.qpos == pytest.approx(pose)
    assert not simulation.data.eq_active[rear_latch]
    assert not simulation.data.eq_active[front_latch]
    assert simulation.data.ctrl[simulation.turn_actuator_ids[0]] > 0.0


@pytest.mark.parametrize(
    ("motion", "expected_degrees"),
    [
        (MotionAction.TURN_LEFT, 5.0),
        (MotionAction.TURN_RIGHT, -5.0),
    ],
)
def test_turn_reaches_five_degrees_smoothly_without_a_cleat(
    motion: MotionAction, expected_degrees: float
) -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    initial = _heading(simulation)
    headings = [initial]
    torques: list[float] = []

    _finish_turn(
        simulation,
        {0: RobotAction(motion)},
        headings=headings,
        torques=torques,
    )

    yaw_change_deg = (_heading(simulation) - initial) * 180.0 / pi
    per_physics_step = np.diff(np.unwrap(headings))
    assert yaw_change_deg == pytest.approx(expected_degrees, abs=0.15)
    assert max(abs(per_physics_step)) < np.deg2rad(0.25)
    assert max(abs(np.asarray(torques))) <= simulation.robot.turn_torque_nm
    assert any(abs(torque) > 0.0 for torque in torques)
    for anchor in ("rear", "front"):
        latch = simulation.model.equality(f"robot_0_gait_{anchor}_latch").id
        assert not simulation.data.eq_active[latch]


@pytest.mark.parametrize(
    ("motion", "expected_degrees"),
    [
        (MotionAction.TURN_LEFT, 5.0),
        (MotionAction.TURN_RIGHT, -5.0),
    ],
)
def test_turn_finishes_in_one_control_step(
    motion: MotionAction, expected_degrees: float
) -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    initial = _heading(simulation)

    simulation.step({0: RobotAction(motion)})

    assert simulation.turn_is_settled(0)
    assert (_heading(simulation) - initial) * 180.0 / pi == pytest.approx(
        expected_degrees, abs=0.15
    )


def test_turn_keeps_the_same_five_degree_target_until_it_settles() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    initial = _heading(simulation)

    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    target = simulation._turn_target_headings[0]
    assert simulation.turn_is_settled(0)
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    assert simulation._turn_target_headings[0] == pytest.approx(
        target + np.deg2rad(5.0)
    )
    assert (_heading(simulation) - initial) * 180.0 / pi == pytest.approx(
        10.0, abs=0.15
    )


def test_turn_resumes_after_manual_stop() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    simulation.halt_motion()
    simulation.end_turn_sessions()
    heading = _heading(simulation)

    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})

    assert _heading(simulation) - heading > np.deg2rad(0.1)
    assert simulation.turn_is_settled(0)


def test_held_turn_repeats_five_degree_increments_without_stopping() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    initial = _heading(simulation)
    action = {0: RobotAction(MotionAction.TURN_LEFT)}

    simulation.step(action)
    simulation.step(action)

    assert (_heading(simulation) - initial) * 180.0 / pi == pytest.approx(
        10.0, abs=0.2
    )


def test_turn_preserves_a_curled_rear_while_reaching_its_target() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(MotionAction.CURL_BODY)})
    rear_before = simulation._joint_position(0, 0)
    initial_heading = _heading(simulation)

    _finish_turn(simulation, {0: RobotAction(MotionAction.TURN_LEFT)})

    assert simulation._joint_position(0, 0) > np.deg2rad(40.0)
    assert rear_before - simulation._joint_position(0, 0) < np.deg2rad(8.0)
    assert (_heading(simulation) - initial_heading) * 180.0 / pi == pytest.approx(
        5.0, abs=0.15
    )


@pytest.mark.parametrize(
    "gait",
    [
        (MotionAction.FLATTEN_BODY,),
        (MotionAction.CURL_BODY,),
        (MotionAction.CURL_BODY, MotionAction.FLATTEN_BODY),
    ],
)
@pytest.mark.parametrize(
    ("turn", "expected_degrees"),
    [
        (MotionAction.TURN_LEFT, 5.0),
        (MotionAction.TURN_RIGHT, -5.0),
    ],
)
@pytest.mark.parametrize("halt_between_inputs", [False, True])
def test_turn_after_ws_input_does_not_shake_or_slide(
    gait: tuple[MotionAction, ...],
    turn: MotionAction,
    expected_degrees: float,
    halt_between_inputs: bool,
) -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    for motion in gait:
        simulation.step({0: RobotAction(motion)})
        if halt_between_inputs:
            # A released W/S key takes this manual-control path before A/D;
            # the false case covers pressing A/D before release polling wins.
            simulation.halt_motion()
            simulation.end_turn_sessions()
    heading = _heading(simulation)
    headings = [heading]
    positions = [simulation.data.xpos[simulation.root_body_ids[0], :2].copy()]

    _finish_turn(
        simulation,
        {0: RobotAction(turn)},
        headings=headings,
        positions=positions,
        maximum_steps=60,
    )

    yaw_change_deg = (_heading(simulation) - heading) * 180.0 / pi
    assert yaw_change_deg == pytest.approx(expected_degrees, abs=0.2)
    assert max(abs(np.diff(np.unwrap(headings)))) < np.deg2rad(0.27)
    assert np.linalg.norm(positions[-1] - positions[0]) < 0.01
    for anchor in ("rear", "front"):
        latch = simulation.model.equality(f"robot_0_gait_{anchor}_latch").id
        assert not simulation.data.eq_active[latch]


def test_turn_reaches_five_degrees_while_supported_by_another_robot() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    lower_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    upper_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[1]])
    simulation.data.qpos[upper_address : upper_address + 3] = simulation.data.qpos[
        lower_address : lower_address + 3
    ]
    # Offset the upper robot so that part of it overhangs the lower robot and
    # its support can change while it turns.
    simulation.data.qpos[upper_address] += 0.07
    simulation.data.qpos[upper_address + 2] = 0.015
    mujoco.mj_forward(simulation.model, simulation.data)
    for _ in range(2 * simulation.physics_steps_per_control):
        mujoco.mj_step(simulation.model, simulation.data)
    simulation.data.qvel[:] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)
    upper_before = _heading(simulation, 1)
    headings = [upper_before]

    _finish_turn(
        simulation,
        {0: RobotAction(), 1: RobotAction(MotionAction.TURN_LEFT)},
        robot_id=1,
        headings=headings,
        maximum_steps=60,
    )

    lower_rear_latch = simulation.model.equality("robot_1_gait_rear_on_0_rear").id
    yaw_change_deg = (_heading(simulation, 1) - upper_before) * 180.0 / pi
    assert not simulation.data.eq_active[lower_rear_latch]
    assert yaw_change_deg == pytest.approx(5.0, abs=0.2)
    assert max(abs(np.diff(np.unwrap(headings)))) < np.deg2rad(0.30)


def test_turn_on_step_top_reaches_five_degrees_without_a_cleat() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "step"))
    root_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    simulation.data.qpos[root_address : root_address + 3] = (0.12, 0.0, 0.0345)
    mujoco.mj_forward(simulation.model, simulation.data)
    for _ in range(simulation.physics_steps_per_control):
        mujoco.mj_step(simulation.model, simulation.data)
    simulation.data.qvel[:] = 0.0
    simulation.data.time = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)
    heading = _heading(simulation)

    _finish_turn(simulation, {0: RobotAction(MotionAction.TURN_LEFT)})

    rear_latch = simulation.model.equality("robot_0_gait_rear_latch").id
    assert not simulation.data.eq_active[rear_latch]
    assert (_heading(simulation) - heading) * 180.0 / pi == pytest.approx(
        5.0, abs=0.2
    )


def test_turn_settles_smoothly_while_straddling_a_step_edge() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "step"))
    root_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    simulation.data.qpos[root_address : root_address + 3] = (-0.05, 0.0, 0.04)
    mujoco.mj_forward(simulation.model, simulation.data)
    for _ in range(4 * simulation.physics_steps_per_control):
        mujoco.mj_step(simulation.model, simulation.data)
    simulation.data.qvel[:] = 0.0
    simulation.data.time = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)
    rotation = simulation.data.xmat[simulation.root_body_ids[0]].reshape(3, 3).copy()
    heading = _heading(simulation)
    headings = [heading]

    _finish_turn(
        simulation,
        {0: RobotAction(MotionAction.TURN_LEFT)},
        headings=headings,
        maximum_steps=60,
    )

    assert abs(rotation[2, 0]) > 0.4
    assert (_heading(simulation) - heading) * 180.0 / pi == pytest.approx(
        5.0, abs=0.25
    )
    assert max(abs(np.diff(np.unwrap(headings)))) < np.deg2rad(0.40)


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


def test_attachment_ignores_a_brief_overload_spike() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction(grip=GripAction.ATTACH)})
    root_body_id = simulation.root_body_ids[0]
    simulation.data.xfrc_applied[root_body_id, :3] = (1.0, 0.0, 0.0)

    for _ in range(24):  # 48 ms at the fixed 2 ms physics timestep
        simulation.attachments.post_physics_step()

    assert simulation.observations()[0].is_attaching
    simulation.data.xfrc_applied[root_body_id, :3] = 0.0
    simulation.attachments.post_physics_step()
    assert simulation.observations()[0].is_attaching


def test_robot_attachment_settles_when_placed_on_another_robot() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    lower_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[0]])
    upper_address = int(simulation.model.jnt_qposadr[simulation.root_joint_ids[1]])
    simulation.data.qpos[upper_address : upper_address + 3] = simulation.data.qpos[
        lower_address : lower_address + 3
    ]
    # This is inside the 12 mm attachment range, but initially above the
    # lower robot's top surface.  The equality must be allowed to settle it.
    simulation.data.qpos[upper_address + 2] = 0.012
    mujoco.mj_forward(simulation.model, simulation.data)

    attached = simulation.step(
        {0: RobotAction(), 1: RobotAction(grip=GripAction.ATTACH)}
    )
    held = simulation.step({0: RobotAction(), 1: RobotAction(grip=GripAction.ATTACH)})
    overloaded = simulation.step(
        {0: RobotAction(), 1: RobotAction(grip=GripAction.ATTACH)},
        external_forces_n={1: (1.0, 0.0, 0.0)},
    )

    assert attached.observations[1].is_attaching
    assert held.observations[1].is_attaching
    assert not any(event.event == "overload_detached" for event in attached.attachment_events)
    assert not overloaded.observations[1].is_attaching
    assert overloaded.observations[1].strain_value == pytest.approx(100.0)


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
