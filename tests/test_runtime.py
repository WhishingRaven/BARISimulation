from __future__ import annotations

from contextlib import nullcontext

import mujoco
import numpy as np
import pytest

from bari_sim.attachment import CombinedFailureCriterion, ForceFailureCriterion
from bari_sim.config import load_experiment_config
from bari_sim.controllers.gait import TravelingWaveGait
from bari_sim.controllers.manual import ManualTeleop
from bari_sim.environment import EnvironmentEvaluator
from bari_sim.model_builder import root_joint_name
from bari_sim.simulator import SwarmSimulator
from bari_sim.types import GlobalRobotState, RobotAction
from bari_sim.visualization import DebugVisualizer


def test_model_compiles_and_steps_finitely() -> None:
    config = load_experiment_config().with_overrides(robot_count=2, environment="flat")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        assert np.isclose(
            mujoco.mj_getTotalmass(simulator.model),
            2.0 * config.robot.total_mass_kg,
        )
        for _ in range(100):
            simulator.step({0: RobotAction(joint_targets_rad=(0.2, -0.2))})
        assert np.isfinite(simulator.data.qpos).all()
        assert np.isfinite(simulator.data.qvel).all()
        observation = simulator.robot(0).get_observation()
        assert len(observation.joint_positions_rad) == 2
        assert observation.joint_positions_rad[0] > 0.05
        assert observation.joint_positions_rad[1] < -0.05
        assert set(observation.force_sensors) == {
            f"link_{link_id}.{location}"
            for link_id in range(3)
            for location in ("top_rear", "top_front", "left", "right")
        }
        assert all(
            reading.force_n >= 0.0 for reading in observation.force_sensors.values()
        )
        assert set(observation.ranges) == {
            "forward",
            "forward_left",
            "forward_right",
            "forward_down",
        }
        simulator.robot(0).set_joint_torque(0, 100.0)
        simulator.robot(0).apply_control(simulator.timestep_s)
        assert np.isclose(
            simulator.data.ctrl[simulator.robot(0).actuator_ids[0]],
            config.robot.joint.max_torque_nm,
        )
        controlled_joint_id = simulator.robot(0).joint_ids[0]
        dof_address = int(simulator.model.jnt_dofadr[controlled_joint_id])
        simulator.data.qvel[dof_address] = 1.1 * config.robot.joint.max_speed_rad_s
        simulator.robot(0).apply_control(simulator.timestep_s)
        assert simulator.data.ctrl[simulator.robot(0).actuator_ids[0]] == 0.0


def test_top_force_sensors_measure_load_without_side_cross_talk() -> None:
    config = load_experiment_config().with_overrides(robot_count=1, environment="flat")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        root_joint_id = simulator.model.joint(root_joint_name(0)).id
        qpos_address = int(simulator.model.jnt_qposadr[root_joint_id])
        simulator.data.qpos[qpos_address : qpos_address + 3] = (-0.55, 0.0, 0.014)
        simulator.data.qpos[qpos_address + 3 : qpos_address + 7] = (
            0.0,
            1.0,
            0.0,
            0.0,
        )
        simulator.data.qvel[:] = 0.0
        mujoco.mj_forward(simulator.model, simulator.data)
        for _ in range(200):
            simulator.step()

        readings = simulator.robot(0).get_observation().force_sensors
        top_load = sum(
            reading.force_n for name, reading in readings.items() if ".top_" in name
        )
        side_load = sum(
            reading.force_n
            for name, reading in readings.items()
            if name.endswith((".left", ".right"))
        )
        assert np.isclose(
            top_load,
            config.robot.total_mass_kg * config.simulation.gravity_m_s2,
            rtol=0.12,
        )
        assert side_load < 1e-9


@pytest.mark.parametrize("environment", ("flat", "gap", "step"))
def test_each_environment_compiles_and_steps(environment: str) -> None:
    config = load_experiment_config().with_overrides(
        robot_count=2, environment=environment
    )
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        for _ in range(20):
            simulator.step()
        assert np.isfinite(simulator.data.qpos).all()


def test_privileged_evaluator_detects_gap_and_step_crossings() -> None:
    base = load_experiment_config()
    state = GlobalRobotState(
        robot_id=4,
        position_m=(2.0, 0.0, 0.25),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity_m_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    )
    for name in ("gap", "step"):
        config = load_experiment_config().with_overrides(environment=name)
        evaluator = EnvironmentEvaluator(config.environment, base.robot)
        assert evaluator.update((state,)) == {4}
        assert evaluator.update((state,)) == set()


def test_arbitrary_contact_attachment_stores_target_local_point() -> None:
    config = load_experiment_config().with_overrides(robot_count=2, environment="flat")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        z = 0.15
        rear0 = np.asarray((-0.55, 0.0, z), dtype=np.float64)
        rear1 = rear0 + np.asarray(
            (
                sum(config.robot.link_lengths_m)
                + config.robot.gripper.length_m
                - 0.0001,
                0.0,
                0.0,
            )
        )
        for robot_id, position in ((0, rear0), (1, rear1)):
            joint_id = simulator.model.joint(root_joint_name(robot_id)).id
            address = int(simulator.model.jnt_qposadr[joint_id])
            simulator.data.qpos[address : address + 3] = position
            simulator.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        simulator.data.qvel[:] = 0.0
        mujoco.mj_forward(simulator.model, simulator.data)

        assert simulator.data.ncon > 0
        assert simulator.robot(0).attach()
        attachment = simulator.attachments.get_attachment(0)
        assert attachment is not None
        assert attachment.target_robot_id == 1
        assert simulator.data.eq_active[attachment.equality_id] == 1
        world_point = simulator.attachments.target_world_point(attachment)
        source_world_point = simulator.data.site_xpos[attachment.source_site_id]
        assert np.isfinite(world_point).all()
        assert np.allclose(source_world_point, world_point, atol=1e-8)
        for _ in range(5):
            simulator.step()
        assert simulator.robot(0).get_attachment_state().active
        assert simulator.robot(0).detach()


def test_forward_range_locally_identifies_another_robot() -> None:
    config = load_experiment_config().with_overrides(robot_count=2, environment="flat")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        rear0 = np.asarray((-0.55, 0.0, 0.15), dtype=np.float64)
        rear1 = rear0 + np.asarray(
            (
                sum(config.robot.link_lengths_m) + config.robot.gripper.length_m + 0.10,
                0.0,
                0.0,
            )
        )
        for robot_id, position in ((0, rear0), (1, rear1)):
            joint_id = simulator.model.joint(root_joint_name(robot_id)).id
            address = int(simulator.model.jnt_qposadr[joint_id])
            simulator.data.qpos[address : address + 3] = position
            simulator.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        simulator.data.qvel[:] = 0.0
        mujoco.mj_forward(simulator.model, simulator.data)
        simulator.sensors[0].reset()

        reading = simulator.robot(0).get_observation().ranges["forward"]
        assert reading.detected
        assert reading.hit_robot_id == 1
        assert 0.05 < reading.distance_m < 0.15


def test_failure_criteria_leave_moment_extension_point() -> None:
    force_only = ForceFailureCriterion(2.0)
    combined = CombinedFailureCriterion(2.0, 1.0)
    assert force_only.utilization(1.0, 100.0) == 0.5
    assert combined.utilization(2.0, 0.0) == 1.0
    assert combined.utilization(0.0, 1.0) == 1.0


def test_manual_callback_and_debug_scene_work_without_external_keyboard_package() -> (
    None
):
    config = load_experiment_config().with_overrides(robot_count=2, environment="gap")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        teleop = ManualTeleop(simulator.robot_count, config.robot.joint_count)
        teleop.key_callback(ord("2"))
        teleop.key_callback(ord("2"))
        teleop.key_callback(ord("Q"))
        actions = teleop.update(simulator, simulator.get_observations())
        assert teleop.state.selected_robot_id == 1
        assert set(actions) == {0, 1}
        assert actions[1].joint_targets_rad is not None
        assert actions[1].joint_targets_rad[0] > 0

        ordered = ManualTeleop(simulator.robot_count, config.robot.joint_count)
        ordered.key_callback(ord(" "))
        ordered.key_callback(ord("2"))
        ordered_actions = ordered.update(simulator, simulator.get_observations())
        assert ordered_actions[0].attach
        assert not ordered_actions[1].attach

        class FakeViewer:
            def __init__(self):
                self.user_scn = mujoco.MjvScene(simulator.model, 200)
                self.opt = mujoco.MjvOption()
                self.opt.geomgroup[:] = 0
                self.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
                self.user_scn.flags[:] = 1
                self.texts = None

            def lock(self):
                return nullcontext()

            def set_texts(self, texts):
                self.texts = texts

        viewer = FakeViewer()
        visualizer = DebugVisualizer()
        observations = simulator.get_observations()
        visualizer.update(viewer, simulator, selected_robot_id=1)
        visualizer.set_text(viewer, teleop, observations[1], simulator)
        assert viewer.user_scn.ngeom >= simulator.robot_count
        stable_geometry_count = viewer.user_scn.ngeom
        assert tuple(viewer.opt.geomgroup) == (1, 1, 1, 0, 0, 0)
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] == 0
        assert viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] == 0
        assert viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] == 1
        assert viewer.texts is not None
        assert any("Gap width" in item[2] for item in viewer.texts)

        teleop.key_callback(ord("V"))
        teleop.update(simulator, observations)
        visualizer.update(
            viewer,
            simulator,
            selected_robot_id=1,
            show_sensor_rays=teleop.state.debug_visible,
        )
        assert viewer.user_scn.ngeom > stable_geometry_count


@pytest.mark.parametrize("key,sign", (("W", 1.0), ("S", -1.0)))
def test_manual_wave_keys_move_in_labeled_direction(key: str, sign: float) -> None:
    config = load_experiment_config().with_overrides(robot_count=1, environment="flat")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        simulator.run_headless(0.8)
        initial_x = simulator.robot(0).get_global_state().position_m[0]
        teleop = ManualTeleop(simulator.robot_count, config.robot.joint_count)
        teleop.key_callback(ord(key))
        observations = simulator.get_observations()
        for _ in range(int(4.0 / simulator.timestep_s)):
            actions = teleop.update(simulator, observations)
            observations = simulator.step(actions).observations
        final_x = simulator.robot(0).get_global_state().position_m[0]
        signed_displacement = sign * (final_x - initial_x)
        assert signed_displacement > 0.03


def test_default_wave_physically_uses_expanded_joint_stroke() -> None:
    config = load_experiment_config().with_overrides(robot_count=1, environment="flat")
    with SwarmSimulator(config, logging_enabled=False) as simulator:
        observations = simulator.get_observations()
        gait = TravelingWaveGait()
        positions: list[tuple[float, ...]] = []
        for _ in range(int(5.0 / simulator.timestep_s)):
            observations = simulator.step({0: gait.act(observations[0])}).observations
            positions.append(observations[0].joint_positions_rad)
        measured = np.asarray(positions)
        # The stability gait keeps a meaningful physical stroke while avoiding
        # the previous high-amplitude roll-inducing wave.
        assert np.all(np.ptp(measured, axis=0) > np.deg2rad(50.0))


def test_attachment_breaks_and_records_solver_load(tmp_path) -> None:
    config = load_experiment_config().with_overrides(robot_count=1, environment="flat")
    with SwarmSimulator(
        config, logging_enabled=True, log_directory=tmp_path
    ) as simulator:
        logger = simulator.logger
        assert logger is not None
        observation = simulator.get_observations()[0]
        posture = (0.50,) * config.robot.joint_count
        for _ in range(int(1.5 / simulator.timestep_s)):
            if observation.gripper_contact:
                break
            observation = simulator.step(
                {0: RobotAction(joint_targets_rad=posture)}
            ).observations[0]
        assert observation.gripper_contact
        for _ in range(int(0.8 / simulator.timestep_s)):
            simulator.step({0: RobotAction(joint_targets_rad=posture)})
        assert simulator.robot(0).attach()

        failure = None
        maximum = 4.0 * config.robot.attachment.break_force_n
        for index in range(1500):
            force = maximum * (index + 1) / 1500
            result = simulator.step(external_forces_n={0: (0.0, 0.0, force)})
            failure = next(
                (
                    event
                    for event in result.attachment_events
                    if event.event == "attachment_failure"
                ),
                None,
            )
            if failure is not None:
                break
        assert failure is not None
        assert failure.force_n > config.robot.attachment.break_force_n
        assert failure.force_n < 1.15 * config.robot.attachment.break_force_n
        assert not simulator.robot(0).get_attachment_state().active
    assert "attachment_failure" in logger.event_path.read_text(encoding="utf-8")


def test_csv_logger_writes_state_schema(tmp_path) -> None:
    config = load_experiment_config().with_overrides(robot_count=1, environment="flat")
    simulator = SwarmSimulator(config, logging_enabled=True, log_directory=tmp_path)
    logger = simulator.logger
    assert logger is not None
    simulator.step()
    simulator.close()

    state_text = logger.state_path.read_text(encoding="utf-8")
    event_text = logger.event_path.read_text(encoding="utf-8")
    assert "maximum_structural_load_n" in state_text.splitlines()[0]
    assert "force_sensor_readings_n" in state_text.splitlines()[0]
    assert len(state_text.splitlines()) >= 2
    assert "event" in event_text.splitlines()[0]
