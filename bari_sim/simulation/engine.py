"""One 0.5-second discrete control step over fine-grained MuJoCo physics."""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..robot.actions import GripAction, LiftAction, MotionAction, RobotAction
from ..robot.observation import RobotObservation
from ..robot.specification import DEFAULT_ROBOT, RobotSpecification
from ..tasks.evaluation import TaskEvaluator, TaskResult
from .attachments import AttachmentEvent, AttachmentManager
from .gait_anchors import GaitAnchorSystem
from .scene import (
    BuiltScene,
    SceneBuilder,
    SceneRequest,
    actuator_name,
    body_name,
    hinge_name,
    turn_actuator_name,
)
from .sensing import SensorSystem

FrameCallback = Callable[["Simulation"], bool | None]


@dataclass(frozen=True)
class StepResult:
    observations: Mapping[int, RobotObservation]
    attachment_events: tuple[AttachmentEvent, ...]
    task_result: TaskResult | None


class Simulation:
    """MuJoCo runtime whose public ``step`` is exactly one policy interval."""

    def __init__(
        self,
        request: SceneRequest,
        *,
        robot: RobotSpecification = DEFAULT_ROBOT,
    ):
        self.request = request
        self.robot = robot
        self.scene: BuiltScene = SceneBuilder(request, robot).build()
        self.model = mujoco.MjModel.from_xml_string(self.scene.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.robot_count = request.grid.count
        ratio = robot.control_interval_s / float(self.model.opt.timestep)
        self.physics_steps_per_control = round(ratio)
        if abs(ratio - self.physics_steps_per_control) > 1e-9:
            raise ValueError(
                "control interval must be an integer number of physics steps"
            )
        # Spawn poses leave a small clearance for the body collision geoms.
        # Let gravity/contact settle that clearance before accepting a manual
        # latch command; otherwise the first equality activation corrects the
        # entire gap in one step and produces a large, one-off yaw impulse.
        for _ in range(self.physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.root_body_ids = tuple(
            self.model.body(body_name(robot_id, "rear")).id
            for robot_id in range(self.robot_count)
        )
        self.joint_ids = np.asarray(
            [
                [
                    self.model.joint(hinge_name(robot_id, hinge)).id
                    for hinge in ("rear", "front")
                ]
                for robot_id in range(self.robot_count)
            ],
            dtype=np.int32,
        )
        self.actuator_ids = np.asarray(
            [
                [
                    self.model.actuator(actuator_name(robot_id, hinge)).id
                    for hinge in ("rear", "front")
                ]
                for robot_id in range(self.robot_count)
            ],
            dtype=np.int32,
        )
        self.root_joint_ids = tuple(
            self.model.joint(f"robot_{robot_id}_root").id
            for robot_id in range(self.robot_count)
        )
        self.turn_actuator_ids = np.asarray(
            [
                self.model.actuator(turn_actuator_name(robot_id)).id
                for robot_id in range(self.robot_count)
            ],
            dtype=np.int32,
        )
        self.attachments = AttachmentManager(
            self.model, self.data, self.robot_count, robot
        )
        self.gait_anchors = GaitAnchorSystem(
            self.model, self.data, self.robot_count, robot
        )
        self.sensors = SensorSystem(
            self.model, self.data, self.robot_count, self.attachments, robot
        )
        self.evaluator = (
            TaskEvaluator(request.task, self.scene, self.model, self.data, robot)
            if request.task is not None
            else None
        )
        self._desired_targets = np.zeros((self.robot_count, 2), dtype=np.float64)
        self._limited_targets = np.zeros((self.robot_count, 2), dtype=np.float64)
        self._turn_torques = np.zeros(self.robot_count, dtype=np.float64)
        self._turn_start_headings = np.full(self.robot_count, np.nan)
        self._turn_directions = np.zeros(self.robot_count, dtype=np.int8)
        self._last_actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        mujoco.mj_forward(self.model, self.data)

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    def observations(self) -> Mapping[int, RobotObservation]:
        return self.sensors.read_all()

    def step(
        self,
        actions: Mapping[int, RobotAction],
        *,
        external_forces_n: Mapping[int, tuple[float, float, float]] | None = None,
        frame_callback: FrameCallback | None = None,
        realtime: bool = False,
        render_hz: float = 60.0,
        lock_root_motion: Collection[int] | None = None,
        lock_root_translation: Collection[int] | None = None,
    ) -> StepResult:
        self._validate_actions(actions)
        self.attachments.begin_control_step()
        locked_roots = set(lock_root_motion or ())
        locked_translations = set(lock_root_translation or ())
        for robot_id in range(self.robot_count):
            action = actions[robot_id]
            if self._rear_posture_is_reached(robot_id, action.motion):
                locked_roots.add(robot_id)
            self._set_action(robot_id, action)
        mujoco.mj_forward(self.model, self.data)
        for robot_id in range(self.robot_count):
            action = actions[robot_id]
            gait_motion = (
                MotionAction.STOP
                if action.grip is GripAction.ATTACH
                or self.attachments.is_attaching(robot_id)
                else action.motion
            )
            self.gait_anchors.apply_motion(robot_id, gait_motion)
            self.attachments.apply_command(robot_id, action.grip)
            self._last_actions[robot_id] = action

        # Gait/attachment commands can both move target sites and activate an
        # equality.  Refresh kinematics before the first mj_step so its
        # constraint solver sees the new target point, rather than the prior
        # frame's (often origin) site transform and injecting an impulse.
        mujoco.mj_forward(self.model, self.data)

        collision_pairs: set[tuple[int, int]] = set()
        locked_roots = tuple(locked_roots)
        locked_translations = tuple(locked_translations)
        locked_poses = {
            robot_id: (
                self.data.qpos[
                    int(self.model.jnt_qposadr[self.root_joint_ids[robot_id]]) :
                ][:7].copy(),
                self.data.qvel[
                    int(self.model.jnt_dofadr[self.root_joint_ids[robot_id]]) :
                ][:6].copy(),
            )
            for robot_id in locked_roots
        }
        locked_positions = {
            robot_id: (
                self.data.qpos[
                    int(self.model.jnt_qposadr[self.root_joint_ids[robot_id]]) :
                ][:3].copy(),
                self.data.qvel[
                    int(self.model.jnt_dofadr[self.root_joint_ids[robot_id]]) :
                ][:3].copy(),
            )
            for robot_id in locked_translations
        }
        callback_stride = max(1, round(1.0 / (render_hz * self.timestep_s)))
        wall_start = time.monotonic()
        for executed_steps, physics_index in enumerate(
            range(self.physics_steps_per_control), start=1
        ):
            self.data.qfrc_applied[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            self._apply_joint_control()
            if external_forces_n:
                for robot_id, force in external_forces_n.items():
                    self.data.xfrc_applied[self.root_body_ids[robot_id], :3] = force
            mujoco.mj_step(self.model, self.data)
            for robot_id, (qpos, qvel) in locked_poses.items():
                qpos_address = int(
                    self.model.jnt_qposadr[self.root_joint_ids[robot_id]]
                )
                dof_address = int(self.model.jnt_dofadr[self.root_joint_ids[robot_id]])
                self.data.qpos[qpos_address : qpos_address + 7] = qpos
                self.data.qvel[dof_address : dof_address + 6] = qvel
            for robot_id, (position, velocity) in locked_positions.items():
                qpos_address = int(
                    self.model.jnt_qposadr[self.root_joint_ids[robot_id]]
                )
                dof_address = int(self.model.jnt_dofadr[self.root_joint_ids[robot_id]])
                self.data.qpos[qpos_address : qpos_address + 3] = position
                self.data.qvel[dof_address : dof_address + 3] = velocity
            if locked_poses or locked_positions:
                mujoco.mj_forward(self.model, self.data)
            self.attachments.post_physics_step()
            if self.attachments.lock_active_roots():
                mujoco.mj_forward(self.model, self.data)
            collision_pairs.update(self._robot_collision_pairs())
            if frame_callback is not None and (
                physics_index % callback_stride == 0
                or physics_index == self.physics_steps_per_control - 1
            ):
                if realtime:
                    target = executed_steps * self.timestep_s
                    delay = target - (time.monotonic() - wall_start)
                    if delay > 0.0:
                        time.sleep(delay)
                if frame_callback(self) is False:
                    break

        if self.evaluator is not None:
            self.evaluator.update(collision_pairs)
        return StepResult(
            observations=self.observations(),
            attachment_events=self.attachments.drain_events(),
            task_result=None if self.evaluator is None else self.evaluator.result(),
        )

    def run(
        self,
        policy: Callable[[int, RobotObservation], RobotAction],
        duration_s: float,
        *,
        stop_on_complete: bool = True,
        frame_callback: FrameCallback | None = None,
        realtime: bool = False,
    ) -> TaskResult | None:
        if duration_s <= 0.0:
            raise ValueError("duration must be positive")
        observations = self.observations()
        end_time = self.time_s + duration_s
        while self.time_s + 1e-12 < end_time:
            actions = {
                robot_id: policy(robot_id, observations[robot_id])
                for robot_id in range(self.robot_count)
            }
            result = self.step(
                actions, frame_callback=frame_callback, realtime=realtime
            )
            observations = result.observations
            if (
                stop_on_complete
                and self.evaluator is not None
                and self.evaluator.complete
            ):
                break
            if frame_callback is not None and result.task_result is not None:
                # Viewer closure is handled inside step.  A closed viewer makes
                # the next callback return false without mutating policy state.
                pass
        return None if self.evaluator is None else self.evaluator.result()

    def reset(self) -> Mapping[int, RobotObservation]:
        mujoco.mj_resetData(self.model, self.data)
        self.attachments.reset()
        self.gait_anchors.reset()
        self._desired_targets[:] = 0.0
        self._limited_targets[:] = 0.0
        self._turn_torques[:] = 0.0
        self._turn_start_headings[:] = np.nan
        self._turn_directions[:] = 0
        self._last_actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        mujoco.mj_forward(self.model, self.data)
        if self.request.task is not None:
            self.evaluator = TaskEvaluator(
                self.request.task, self.scene, self.model, self.data, self.robot
            )
        return self.observations()

    def halt_motion(self) -> None:
        """Freeze the current pose at a manual-control action boundary."""

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        for robot_id in range(self.robot_count):
            for hinge_index in range(2):
                position = self._joint_position(robot_id, hinge_index)
                self._desired_targets[robot_id, hinge_index] = position
                self._limited_targets[robot_id, hinge_index] = position
        mujoco.mj_forward(self.model, self.data)

    def end_turn_sessions(self) -> None:
        """Clear the current manual turn segment when its key is released."""

        self._turn_start_headings[:] = np.nan
        self._turn_directions[:] = 0

    def export_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.scene.xml, encoding="utf-8")

    def _set_action(self, robot_id: int, action: RobotAction) -> None:
        # A latched rear spike is a physical anchor.  Motion commands may
        # still update the requested flap state, but must not drive the body
        # away from its attachment point.
        motion = (
            MotionAction.STOP
            if self.attachments.is_attaching(robot_id)
            else action.motion
        )
        if motion is MotionAction.CURL_BODY:
            self._set_rear_target_if_needed(robot_id, self.robot.curl_angle_rad)
        elif motion is MotionAction.FLATTEN_BODY:
            self._set_rear_target_if_needed(robot_id, 0.0)
        elif motion in {MotionAction.TURN_LEFT, MotionAction.TURN_RIGHT}:
            # A/D must be usable from the current manual posture.  Flatten
            # the rear through its real pitch actuator, then enable yaw once
            # its rear cleat has a stable planar support.
            self._set_rear_target_if_needed(robot_id, 0.0)
        elif motion is MotionAction.STOP:
            self._desired_targets[robot_id, 0] = self._joint_position(robot_id, 0)
        # A curled rear segment leaves its cleat without a stable planar
        # support.  Applying yaw torque then tips the body instead of turning
        # it, so only apply the motor once the rear segment is flat.
        can_turn = abs(self._joint_position(robot_id, 0)) <= np.deg2rad(8.0)
        direction = {
            MotionAction.TURN_LEFT: 1,
            MotionAction.TURN_RIGHT: -1,
        }.get(motion, 0)
        if direction != self._turn_directions[robot_id]:
            self._turn_directions[robot_id] = direction
            self._turn_start_headings[robot_id] = (
                self._heading(robot_id) if direction else np.nan
            )
        self._turn_torques[robot_id] = (
            direction * self.robot.turn_torque_nm if can_turn else 0.0
        )
        self._desired_targets[robot_id, 1] = (
            self.robot.front_lift_angle_rad
            if action.lift is LiftAction.LIFT_FRONT
            else 0.0
        )

    def _set_rear_target_if_needed(self, robot_id: int, target: float) -> None:
        """Avoid re-driving a posture that is already reached.

        Reapplying curl/flatten at the target creates contact impulses that
        make a stationary manual robot creep across the floor.
        """

        current = self._joint_position(robot_id, 0)
        if abs(current - target) <= np.deg2rad(8.0):
            self._desired_targets[robot_id, 0] = current
        else:
            self._desired_targets[robot_id, 0] = target

    def _rear_posture_is_reached(self, robot_id: int, motion: MotionAction) -> bool:
        target = {
            MotionAction.CURL_BODY: self.robot.curl_angle_rad,
            MotionAction.FLATTEN_BODY: 0.0,
        }.get(motion)
        return target is not None and abs(
            self._joint_position(robot_id, 0) - target
        ) <= np.deg2rad(8.0)

    def _apply_joint_control(self) -> None:
        maximum_step = self.robot.joint_speed_rad_s * self.timestep_s
        delta = self._desired_targets - self._limited_targets
        self._limited_targets += np.clip(delta, -maximum_step, maximum_step)
        for robot_id in range(self.robot_count):
            for hinge_index in range(2):
                joint_id = int(self.joint_ids[robot_id, hinge_index])
                actuator_id = int(self.actuator_ids[robot_id, hinge_index])
                qpos = float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
                qvel = float(self.data.qvel[int(self.model.jnt_dofadr[joint_id])])
                torque = (
                    self.robot.joint_kp_nm_rad
                    * (self._limited_targets[robot_id, hinge_index] - qpos)
                    - self.robot.joint_kd_nms_rad * qvel
                )
                if abs(qvel) >= self.robot.joint_speed_rad_s and torque * qvel > 0.0:
                    torque = 0.0
                self.data.ctrl[actuator_id] = float(
                    np.clip(
                        torque,
                        -self.robot.joint_torque_nm,
                        self.robot.joint_torque_nm,
                    )
                )
            torque = self._turn_torques[robot_id]
            if torque and self._turn_limit_reached(robot_id):
                # Five degrees is the manual turn increment, not a hold-key
                # stop.  Rebase the physical yaw motor immediately so A/D
                # continues through consecutive 5° increments while held.
                self._turn_start_headings[robot_id] = self._heading(robot_id)
            if torque:
                root_dof_address = int(
                    self.model.jnt_dofadr[self.root_joint_ids[robot_id]]
                )
                yaw_rate = float(self.data.qvel[root_dof_address + 5])
                desired_yaw_rate = np.copysign(self.robot.turn_speed_rad_s, torque)
                torque = float(
                    np.clip(
                        self.robot.turn_rate_kp_nms_rad * (desired_yaw_rate - yaw_rate),
                        -self.robot.turn_torque_nm,
                        self.robot.turn_torque_nm,
                    )
                )
            self.data.ctrl[self.turn_actuator_ids[robot_id]] = torque

    def _joint_position(self, robot_id: int, hinge_index: int) -> float:
        joint_id = int(self.joint_ids[robot_id, hinge_index])
        return float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])

    def _heading(self, robot_id: int) -> float:
        rotation = self.data.xmat[self.root_body_ids[robot_id]].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _turn_limit_reached(self, robot_id: int) -> bool:
        start = self._turn_start_headings[robot_id]
        direction = self._turn_directions[robot_id]
        if direction == 0 or np.isnan(start):
            return False
        delta = np.arctan2(
            np.sin(self._heading(robot_id) - start),
            np.cos(self._heading(robot_id) - start),
        )
        return direction * delta >= self.robot.turn_angle_rad

    def _robot_collision_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.attachments.geom_to_robot.get(int(contact.geom1))
            second = self.attachments.geom_to_robot.get(int(contact.geom2))
            if first is None or second is None or first == second:
                continue
            pairs.add((min(first, second), max(first, second)))
        return pairs

    def _validate_actions(self, actions: Mapping[int, RobotAction]) -> None:
        expected = set(range(self.robot_count))
        actual = set(actions)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"actions must contain every robot exactly once; missing={missing}, extra={extra}"
            )
