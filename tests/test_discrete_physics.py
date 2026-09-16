from __future__ import annotations

from math import atan2, pi

import mujoco
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
    assert final_x - initial_x > 0.10


def test_turn_rotates_whole_free_body_by_one_tuned_increment() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 1), "flat"))
    simulation.step({0: RobotAction()})
    initial = _heading(simulation)
    simulation.step({0: RobotAction(MotionAction.TURN_LEFT)})
    assert (_heading(simulation) - initial) * 180.0 / pi == pytest.approx(8.0, abs=0.2)


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
    overloaded = simulation.step(
        {0: RobotAction(grip=GripAction.ATTACH)},
        external_forces_n={0: (0.0, 0.0, 1.0)},
    )
    assert not overloaded.observations[0].is_attaching
    assert any(
        event.event == "overload_detached" for event in overloaded.attachment_events
    )


def test_nearby_ids_are_refreshed_each_step() -> None:
    simulation = Simulation(SceneRequest(RobotGrid(1, 2), "flat"))
    result = simulation.step({0: RobotAction(), 1: RobotAction()})
    assert result.observations[0].nearby_robot_ids == (1,)
    assert result.observations[1].nearby_robot_ids == (0,)
