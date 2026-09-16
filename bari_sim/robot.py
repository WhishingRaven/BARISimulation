"""Per-robot actuator API and strict local-observation boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from .attachment import AttachmentManager
from .config import RobotConfig
from .model_builder import actuator_name, link_body_name, pitch_joint_name
from .sensors import SensorSuite
from .types import GlobalRobotState, LocalObservation, RobotAction

if TYPE_CHECKING:
    from collections.abc import Sequence


class Robot:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_id: int,
        config: RobotConfig,
        sensors: SensorSuite,
        attachments: AttachmentManager,
    ):
        self.model = model
        self.data = data
        self.robot_id = robot_id
        self.config = config
        self.sensors = sensors
        self.attachments = attachments
        self.root_body_id = model.body(link_body_name(robot_id, 0)).id
        self.joint_ids = tuple(
            model.joint(pitch_joint_name(robot_id, joint_id)).id
            for joint_id in range(config.joint_count)
        )
        self.actuator_ids = tuple(
            model.actuator(actuator_name(robot_id, joint_id)).id
            for joint_id in range(config.joint_count)
        )
        self._desired_targets = np.zeros(config.joint_count, dtype=np.float64)
        self._limited_targets = np.zeros(config.joint_count, dtype=np.float64)
        self._torque_commands = np.zeros(config.joint_count, dtype=np.float64)
        self._control_modes = ["position"] * config.joint_count
        self.reset_control()

    @property
    def joint_count(self) -> int:
        return self.config.joint_count

    @property
    def actuator_commands_nm(self) -> tuple[float, ...]:
        return tuple(
            float(self.data.ctrl[actuator_id]) for actuator_id in self.actuator_ids
        )

    def set_joint_target(self, joint_id: int, angle_rad: float) -> None:
        self._validate_joint_id(joint_id)
        low, high = self.config.joint.limits_rad[joint_id]
        self._desired_targets[joint_id] = float(np.clip(angle_rad, low, high))
        self._control_modes[joint_id] = "position"

    def set_joint_targets(self, angles_rad: Sequence[float]) -> None:
        self._validate_vector(angles_rad)
        for joint_id, angle in enumerate(angles_rad):
            self.set_joint_target(joint_id, float(angle))

    def set_joint_torque(self, joint_id: int, torque_nm: float) -> None:
        self._validate_joint_id(joint_id)
        maximum = self.config.joint.max_torque_nm
        self._torque_commands[joint_id] = float(np.clip(torque_nm, -maximum, maximum))
        self._control_modes[joint_id] = "torque"

    def set_joint_torques(self, torques_nm: Sequence[float]) -> None:
        self._validate_vector(torques_nm)
        for joint_id, torque in enumerate(torques_nm):
            self.set_joint_torque(joint_id, float(torque))

    def apply_action(self, action: RobotAction) -> None:
        if action.detach:
            self.detach()
        if action.joint_targets_rad is not None:
            self.set_joint_targets(action.joint_targets_rad)
        elif action.joint_torques_nm is not None:
            self.set_joint_torques(action.joint_torques_nm)
        if action.attach:
            self.attach()

    def apply_control(self, timestep_s: float) -> None:
        maximum_torque = self.config.joint.max_torque_nm
        maximum_target_step = self.config.joint.max_speed_rad_s * timestep_s
        for joint_id, (model_joint_id, actuator_id) in enumerate(
            zip(self.joint_ids, self.actuator_ids)
        ):
            qpos_address = int(self.model.jnt_qposadr[model_joint_id])
            dof_address = int(self.model.jnt_dofadr[model_joint_id])
            position = float(self.data.qpos[qpos_address])
            velocity = float(self.data.qvel[dof_address])
            if self._control_modes[joint_id] == "position":
                delta = (
                    self._desired_targets[joint_id] - self._limited_targets[joint_id]
                )
                self._limited_targets[joint_id] += float(
                    np.clip(delta, -maximum_target_step, maximum_target_step)
                )
                torque = (
                    self.config.joint.position_kp_nm_rad
                    * (self._limited_targets[joint_id] - position)
                    - self.config.joint.velocity_kd_nms_rad * velocity
                )
            else:
                torque = self._torque_commands[joint_id]
            maximum_speed = self.config.joint.max_speed_rad_s
            if (velocity >= maximum_speed and torque > 0.0) or (
                velocity <= -maximum_speed and torque < 0.0
            ):
                torque = 0.0
            self.data.ctrl[actuator_id] = float(
                np.clip(torque, -maximum_torque, maximum_torque)
            )

    def attach(self) -> bool:
        return self.attachments.attach(self.robot_id)

    def detach(self) -> bool:
        return self.attachments.detach(self.robot_id)

    def get_attachment_state(self):
        return self.attachments.get_state(self.robot_id)

    def get_observation(self) -> LocalObservation:
        forces = self.sensors.read_force_sensors()
        ranges = self.sensors.read_ranges(float(self.data.time))
        neighbors = set(forces.neighboring_robot_ids)
        neighbors.update(
            reading.hit_robot_id
            for reading in ranges.values()
            if reading.hit_robot_id is not None
        )
        return LocalObservation(
            robot_id=self.robot_id,
            simulation_time_s=float(self.data.time),
            joint_positions_rad=self.joint_positions_rad(),
            joint_velocities_rad_s=self.joint_velocities_rad_s(),
            force_sensors=forces.readings,
            ranges=ranges,
            attachment=self.get_attachment_state(),
            gripper_contact=forces.gripper_contact,
            locally_detected_robot_ids=tuple(sorted(neighbors)),
        )

    def get_global_state(self) -> GlobalRobotState:
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body_id,
            spatial_velocity,
            0,
        )
        return GlobalRobotState(
            robot_id=self.robot_id,
            position_m=tuple(
                float(value) for value in self.data.xpos[self.root_body_id]
            ),
            orientation_wxyz=tuple(
                float(value) for value in self.data.xquat[self.root_body_id]
            ),
            linear_velocity_m_s=tuple(float(value) for value in spatial_velocity[3:]),
            angular_velocity_rad_s=tuple(
                float(value) for value in spatial_velocity[:3]
            ),
        )

    def joint_positions_rad(self) -> tuple[float, ...]:
        return tuple(
            float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
            for joint_id in self.joint_ids
        )

    def joint_velocities_rad_s(self) -> tuple[float, ...]:
        return tuple(
            float(self.data.qvel[int(self.model.jnt_dofadr[joint_id])])
            for joint_id in self.joint_ids
        )

    def reset_control(self) -> None:
        positions = np.asarray(self.joint_positions_rad(), dtype=np.float64)
        self._desired_targets[:] = positions
        self._limited_targets[:] = positions
        self._torque_commands[:] = 0.0
        self._control_modes[:] = ["position"] * self.config.joint_count
        for actuator_id in self.actuator_ids:
            self.data.ctrl[actuator_id] = 0.0

    def _validate_joint_id(self, joint_id: int) -> None:
        if joint_id < 0 or joint_id >= self.joint_count:
            raise IndexError(f"joint_id must be in [0, {self.joint_count - 1}]")

    def _validate_vector(self, values: Sequence[float]) -> None:
        if len(values) != self.joint_count:
            raise ValueError(
                f"expected {self.joint_count} joint commands, got {len(values)}"
            )
