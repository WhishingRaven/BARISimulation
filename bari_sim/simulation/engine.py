"""One 0.5-second discrete control step over fine-grained MuJoCo physics."""

from __future__ import annotations

import os
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
from ..tasks.objectives import score_task_result
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


def _env_is_on(name: str) -> bool:
    """Windows에서 값에 따옴표·공백이 섞여도 켜진 것으로 읽는다."""

    value = os.environ.get(name, "").strip().strip("'\"").lower()
    return value in {"1", "true", "yes", "on", "y"}


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
        self._settle_initial_pose()

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
        self._turn_target_headings = np.full(self.robot_count, np.nan)
        self._turn_rate_targets = np.zeros(self.robot_count, dtype=np.float64)
        self._turn_directions = np.zeros(self.robot_count, dtype=np.int8)
        self._last_actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        if _env_is_on("BARI_STRAIN_DEBUG") or _env_is_on("BARI_CONTACT_DEBUG"):
            from . import attachments as _attachments_module
            from . import scene as _scene_module

            margin_m = float(self.model.geom_margin.max())
            gap_m = float(self.model.geom_gap.max())
            print(
                f"[bari] 디버그 출력 켜짐\n"
                f"       engine.py      : {__file__}\n"
                f"       attachments.py : {_attachments_module.__file__}\n"
                f"       scene.py       : {_scene_module.__file__}\n"
                f"       접촉: margin {margin_m * 1000:.3f} mm, gap {gap_m * 1000:.3f} mm"
                f"  →  힘이 걸리기 시작하는 거리 {(margin_m - gap_m) * 1000:.3f} mm"
                + ("   ※ 0 이 아니면 떨어져 있어도 서로 밉니다" if margin_m > gap_m else "")
                + f"\n       과도구간 {AttachmentManager._ACTIVATION_SETTLE_S * 1000:.0f} ms, "
                f"게이지 시상수 {AttachmentManager._STRAIN_TAU_S * 1000:.0f} ms",
                flush=True,
            )
        mujoco.mj_forward(self.model, self.data)

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    def observations(self) -> Mapping[int, RobotObservation]:
        return self.sensors.read_all()

    def turn_is_settled(self, robot_id: int) -> bool:
        """Return whether the active five-degree yaw segment has stopped."""

        return not np.isnan(self._turn_target_headings[robot_id]) and self._turn_is_settled(
            robot_id
        )

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
        reached_postures: set[int] = set()
        for robot_id in range(self.robot_count):
            action = actions[robot_id]
            if self._rear_posture_is_reached(robot_id, action.motion):
                locked_roots.add(robot_id)
                reached_postures.add(robot_id)
            self._set_action(robot_id, action)
        mujoco.mj_forward(self.model, self.data)
        for robot_id in range(self.robot_count):
            action = actions[robot_id]
            gait_motion = (
                MotionAction.STOP
                if action.grip is GripAction.ATTACH
                or self.attachments.is_attaching(robot_id)
                or robot_id in reached_postures
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
        # BARI_POSTURE_LOCK_ALWAYS=1 로 두면 예전(접촉 무시) 동작으로 되돌린다.
        release_disabled = _env_is_on("BARI_POSTURE_LOCK_ALWAYS")
        # 접촉만으로 막지 못한 깊은 침투는 이 한계(m)를 넘는 만큼 직접 떼어 놓는다.
        #
        # 겹침을 막는 주된 수단은 접촉 강성(scene.py의 solref/solimp)이고, 이
        # 보정은 눈에 띄는 관통만 잡는 최후의 안전장치다.  한계를 1 mm처럼 좁게
        # 두면 서로 올라탈 때의 정상적인 눌림까지 침투로 보고 밀어내어, 올라타려는
        # 로봇을 계속 떠민다.  그래서 기본값을 5 mm로 둔다(로봇 두께 5 mm).
        # BARI_CONTACT_LIMIT_MM=0 으로 두면 이 보정을 끈다.
        contact_limit = float(os.environ.get("BARI_CONTACT_LIMIT_MM", "5.0")) / 1000.0
        worst_overlap = 0.0
        worst_world_overlap = 0.0
        corrections = 0
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
            # Posture holds must never win over a contact with another robot:
            # restoring the pose would undo the solver's separation and let the
            # two bodies sink into each other.  Release the hold while touching
            # and adopt the contact-resolved pose as the new hold.
            touching = (
                self._robots_in_foreign_contact()
                if reached_postures and not release_disabled
                else frozenset()
            )
            # 결합한 로봇의 자세는 구속이 책임진다.  여기서 제어 스텝 시작 시점의
            # qpos/qvel 을 2 ms마다 다시 써 넣으면, 구속은 그 속도를 매번 취소해야
            # 하고 그 취소력이 곧바로 수 N(=strain 100)으로 찍힌다.
            latched = self.attachments.constrained_robots()
            for robot_id, (qpos, qvel) in locked_poses.items():
                if robot_id in latched:
                    continue
                qpos_address = int(
                    self.model.jnt_qposadr[self.root_joint_ids[robot_id]]
                )
                dof_address = int(self.model.jnt_dofadr[self.root_joint_ids[robot_id]])
                if robot_id in reached_postures and robot_id in touching:
                    locked_poses[robot_id] = (
                        self.data.qpos[qpos_address : qpos_address + 7].copy(),
                        np.zeros(6, dtype=np.float64),
                    )
                    continue
                self.data.qpos[qpos_address : qpos_address + 7] = qpos
                self.data.qvel[dof_address : dof_address + 6] = qvel
            for robot_id, (position, velocity) in locked_positions.items():
                if robot_id in latched:
                    continue
                qpos_address = int(
                    self.model.jnt_qposadr[self.root_joint_ids[robot_id]]
                )
                dof_address = int(self.model.jnt_dofadr[self.root_joint_ids[robot_id]])
                self.data.qpos[qpos_address : qpos_address + 3] = position
                self.data.qvel[dof_address : dof_address + 3] = velocity
            if locked_poses or locked_positions:
                mujoco.mj_forward(self.model, self.data)
            if self.attachments.lock_active_roots():
                mujoco.mj_forward(self.model, self.data)
            if contact_limit > 0.0:
                corrections += self._separate_deep_contacts(contact_limit, locked_roots)
            # 스트레인은 자세 고정·강제 분리가 모두 끝난 뒤의 상태에서 읽는다.
            # 중간 상태에서 읽으면 보정 중의 순간값이 하중으로 둔갑한다.
            self.attachments.post_physics_step()
            collision_pairs.update(self._robot_collision_pairs())
            pair_overlap, world_overlap = self._worst_robot_overlap()
            worst_overlap = min(worst_overlap, pair_overlap)
            worst_world_overlap = min(worst_world_overlap, world_overlap)
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

        if _env_is_on("BARI_STRAIN_DEBUG"):
            breakdown = self.attachments.strain_breakdown()
            if breakdown:
                body = "  ".join(
                    f"로봇{robot_id}: 구속 {constraint:.3f} N / 외력 {external:.3f} N"
                    f" / 구속오차 {error * 1000:.2f} mm"
                    f" → {self.attachments.strain_value(robot_id):5.1f} g"
                    for robot_id, (constraint, external, error) in sorted(
                        breakdown.items()
                    )
                )
            else:
                # 이 제어 스텝 동안 결합이 한 번도 없었다는 뜻이다.  줄 자체는
                # 찍어서, 설정이 켜졌는지와 결합이 없는지를 구분할 수 있게 한다.
                body = "결합 없음 (이 스텝 동안 붙은 로봇이 없습니다)"
            events = self.attachments.event_summary()
            if events:
                body += f"   | {events}"
            print(f"[strain]  t={self.time_s:6.2f}s  {body}", flush=True)
        if _env_is_on("BARI_CONTACT_DEBUG"):
            print(
                f"[contact] t={self.time_s:6.2f}s  로봇끼리 "
                f"{-worst_overlap * 1000:6.3f} mm  플랫폼 "
                f"{-worst_world_overlap * 1000:6.3f} mm  강제 분리 {corrections}회  "
                f"접촉 쌍 {sorted(collision_pairs)}",
                flush=True,
            )
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
        if self.evaluator is None:
            return None
        assert self.request.task is not None
        return score_task_result(
            self.evaluator.result(),
            self.request.task,
            episode_time_limit_s=duration_s,
        )

    def reset(self) -> Mapping[int, RobotObservation]:
        mujoco.mj_resetData(self.model, self.data)
        self.attachments.reset()
        self.gait_anchors.reset()
        self._desired_targets[:] = 0.0
        self._limited_targets[:] = 0.0
        self._turn_target_headings[:] = np.nan
        self._turn_rate_targets[:] = 0.0
        self._turn_directions[:] = 0
        self._last_actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        self._settle_initial_pose()
        if self.request.task is not None:
            self.evaluator = TaskEvaluator(
                self.request.task, self.scene, self.model, self.data, self.robot
            )
        return self.observations()

    def _settle_initial_pose(self) -> None:
        """Apply the same contact settling on construction and every reset."""

        # Spawn poses leave a small clearance for the body collision geoms.
        # Settling it avoids a one-off impulse and keeps training rollouts
        # identical to fresh inference simulations.
        for _ in range(self.physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def halt_motion(self) -> None:
        """Freeze the current pose at a manual-control action boundary."""

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._turn_rate_targets[:] = 0.0
        for robot_id in range(self.robot_count):
            for hinge_index in range(2):
                position = self._joint_position(robot_id, hinge_index)
                self._desired_targets[robot_id, hinge_index] = position
                self._limited_targets[robot_id, hinge_index] = position
        mujoco.mj_forward(self.model, self.data)

    def halt_robots(self, robot_ids: Collection[int]) -> None:
        """Stop only the requested robots while preserving all other momentum."""

        for robot_id in robot_ids:
            root_dof_address = int(
                self.model.jnt_dofadr[self.root_joint_ids[robot_id]]
            )
            self.data.qvel[root_dof_address : root_dof_address + 6] = 0.0
            self.data.ctrl[self.turn_actuator_ids[robot_id]] = 0.0
            self._turn_rate_targets[robot_id] = 0.0
            for hinge_index in range(2):
                joint_id = int(self.joint_ids[robot_id, hinge_index])
                actuator_id = int(self.actuator_ids[robot_id, hinge_index])
                self.data.qvel[int(self.model.jnt_dofadr[joint_id])] = 0.0
                self.data.ctrl[actuator_id] = 0.0
                position = self._joint_position(robot_id, hinge_index)
                self._desired_targets[robot_id, hinge_index] = position
                self._limited_targets[robot_id, hinge_index] = position
        mujoco.mj_forward(self.model, self.data)

    def end_turn_sessions(self) -> None:
        """Clear the current manual turn segment when its key is released."""

        self._turn_target_headings[:] = np.nan
        self._turn_rate_targets[:] = 0.0
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
        elif motion is MotionAction.STOP:
            self._desired_targets[robot_id, 0] = self._joint_position(robot_id, 0)
        direction = {
            MotionAction.TURN_LEFT: 1,
            MotionAction.TURN_RIGHT: -1,
        }.get(motion, 0)
        target = self._turn_target_headings[robot_id]
        if direction == 0:
            self._turn_target_headings[robot_id] = np.nan
            self._turn_rate_targets[robot_id] = 0.0
        elif direction != self._turn_directions[robot_id] or np.isnan(target):
            self._turn_target_headings[robot_id] = self._wrapped_angle(
                self._heading(robot_id) + direction * self.robot.turn_angle_rad
            )
        elif self._turn_is_settled(robot_id):
            # One public action advances one exact five-degree segment.  If a
            # segment needed extra policy intervals because contact was poor,
            # repeated actions keep its original target until it settles.
            self._turn_target_headings[robot_id] = self._wrapped_angle(
                target + direction * self.robot.turn_angle_rad
            )
        if direction != self._turn_directions[robot_id]:
            self._turn_directions[robot_id] = direction
        if action.lift is LiftAction.LIFT_FRONT:
            self._desired_targets[robot_id, 1] = self.robot.front_lift_angle_rad
        elif action.lift is LiftAction.UNLIFT_FRONT:
            self._desired_targets[robot_id, 1] = 0.0
        elif action.lift is LiftAction.STOP:
            position = self._joint_position(robot_id, 1)
            joint_id = int(self.joint_ids[robot_id, 1])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self._desired_targets[robot_id, 1] = position
            self._limited_targets[robot_id, 1] = position
            self.data.qvel[dof_address] = 0.0

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
            turning = not np.isnan(self._turn_target_headings[robot_id])
            joint_kp = (
                self.robot.turn_joint_kp_nm_rad
                if turning
                else self.robot.joint_kp_nm_rad
            )
            # Velocity feedback is modelled as implicit joint damping in the
            # MJCF (see SceneBuilder._add_hinge).  Only the turn posture uses
            # a different kd; apply just the explicit difference, if any.
            joint_kd = (
                self.robot.turn_joint_kd_nms_rad - self.robot.joint_kd_nms_rad
                if turning
                else 0.0
            )
            joint_torque_limit = (
                self.robot.turn_joint_torque_nm
                if turning
                else self.robot.joint_torque_nm
            )
            for hinge_index in range(2):
                joint_id = int(self.joint_ids[robot_id, hinge_index])
                actuator_id = int(self.actuator_ids[robot_id, hinge_index])
                qpos = float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
                qvel = float(self.data.qvel[int(self.model.jnt_dofadr[joint_id])])
                torque = (
                    joint_kp * (self._limited_targets[robot_id, hinge_index] - qpos)
                    - joint_kd * qvel
                )
                if abs(qvel) >= self.robot.joint_speed_rad_s and torque * qvel > 0.0:
                    torque = 0.0
                self.data.ctrl[actuator_id] = float(
                    np.clip(
                        torque,
                        -joint_torque_limit,
                        joint_torque_limit,
                    )
                )
            target = self._turn_target_headings[robot_id]
            torque = 0.0
            if not np.isnan(target):
                error = self._turn_error(robot_id)
                yaw_rate = self._yaw_rate(robot_id)
                if (
                    abs(error) <= self.robot.turn_angle_tolerance_rad
                    and abs(yaw_rate) <= self.robot.turn_rate_tolerance_rad_s
                ):
                    self._turn_rate_targets[robot_id] = 0.0
                else:
                    braking_rate = np.sqrt(
                        2.0 * self.robot.turn_acceleration_rad_s2 * abs(error)
                    )
                    requested_rate = np.copysign(
                        min(self.robot.turn_speed_rad_s, braking_rate), error
                    )
                    maximum_rate_step = (
                        self.robot.turn_acceleration_rad_s2 * self.timestep_s
                    )
                    rate_delta = requested_rate - self._turn_rate_targets[robot_id]
                    self._turn_rate_targets[robot_id] += float(
                        np.clip(rate_delta, -maximum_rate_step, maximum_rate_step)
                    )
                    torque = float(
                        np.clip(
                            self.robot.turn_rate_kp_nms_rad
                            * (self._turn_rate_targets[robot_id] - yaw_rate),
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

    def _yaw_rate(self, robot_id: int) -> float:
        body_id = self.root_body_ids[robot_id]
        forward = self.data.xmat[body_id].reshape(3, 3)[:, 0]
        horizontal_length_squared = float(forward[0] ** 2 + forward[1] ** 2)
        angular_velocity = self.data.cvel[body_id, :3]
        if horizontal_length_squared <= 1e-12:
            # Heading itself is singular when the body's forward axis is
            # vertical.  World-z angular velocity is the least surprising
            # bounded fallback until a horizontal projection exists again.
            return float(angular_velocity[2])
        # d/dt atan2(forward_y, forward_x), derived from
        # forward_dot = angular_velocity x forward.  Unlike free-joint qvel-z,
        # this remains the actual heading rate while the robot is pitched or
        # rolling over a step or another robot.
        return float(
            angular_velocity[2]
            - forward[2]
            * (
                forward[0] * angular_velocity[0]
                + forward[1] * angular_velocity[1]
            )
            / horizontal_length_squared
        )

    def _turn_error(self, robot_id: int) -> float:
        return self._wrapped_angle(
            self._turn_target_headings[robot_id] - self._heading(robot_id)
        )

    def _turn_is_settled(self, robot_id: int) -> bool:
        return (
            abs(self._turn_error(robot_id)) <= self.robot.turn_angle_tolerance_rad
            and abs(self._yaw_rate(robot_id)) <= self.robot.turn_rate_tolerance_rad_s
        )

    @staticmethod
    def _wrapped_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _robots_in_foreign_contact(self) -> frozenset[int]:
        """Robot ids that currently touch a different robot."""

        touching: set[int] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.attachments.geom_to_robot.get(int(contact.geom1))
            second = self.attachments.geom_to_robot.get(int(contact.geom2))
            if first is None or second is None or first == second:
                continue
            touching.add(first)
            touching.add(second)
        return frozenset(touching)

    def _separate_deep_contacts(
        self, limit_m: float, locked_roots: Collection[int]
    ) -> int:
        """Push bodies apart when a contact is deeper than ``limit_m``.

        Covers robot-to-robot and robot-to-environment (platform, floor, step)
        contacts.  The environment never moves, so the robot takes the whole
        correction there.

        The contact solver is the primary defence; this is a last-resort cap so
        a single violent step can never leave two bodies visibly merged.
        """

        # 결합(connect 구속)에 묶인 로봇은 양쪽 모두 손으로 옮기면 안 된다.  한쪽
        # 끝을 밀어내는 순간 구속 오차가 생기고, 그것을 되돌리려는 힘이 곧바로
        # strain 한계(100 g)로 찍힌다.  붙은 쪽과 붙인 대상 모두 보정에서 뺀다.
        frozen = set(locked_roots) | set(self.attachments.constrained_robots())
        # 한 물리 스텝(2 ms)에 밀어낼 수 있는 최대량 (m).
        push_limit = float(os.environ.get("BARI_CONTACT_PUSH_MM", "0.5")) / 1000.0
        corrections = 0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            depth = -float(contact.dist) - limit_m
            if depth <= 0.0:
                continue
            first = self.attachments.geom_to_robot.get(int(contact.geom1))
            second = self.attachments.geom_to_robot.get(int(contact.geom2))
            if first is None and second is None:
                continue                      # 환경끼리는 움직일 것이 없다
            if first is not None and first == second:
                continue                      # 같은 로봇의 조각끼리는 건드리지 않는다
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            movable = [
                robot_id
                for robot_id in (first, second)
                if robot_id is not None
                and robot_id not in frozen
                and not self.attachments.is_attaching(robot_id)
            ]
            if not movable:
                continue
            # 한 번에 다 밀어내면 그 자체가 순간이동이라 기체가 뒤집힌다.
            # 매 물리 스텝 조금씩 밀어, 여러 스텝에 걸쳐 빠져나오게 한다.
            depth = min(depth, push_limit)
            # 플랫폼·바닥(환경)은 움직이지 않으므로, 로봇 혼자 침투분을 모두 물러난다.
            share = depth / len(movable)
            for robot_id in movable:
                address = int(self.model.jnt_qposadr[self.root_joint_ids[robot_id]])
                direction = -1.0 if robot_id == first else 1.0
                self.data.qpos[address : address + 3] += direction * share * normal
            corrections += 1
        if corrections:
            mujoco.mj_forward(self.model, self.data)
        return corrections

    def _worst_robot_overlap(self) -> tuple[float, float]:
        """Deepest (robot-robot, robot-environment) contact distance."""

        worst_pair = 0.0
        worst_world = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.attachments.geom_to_robot.get(int(contact.geom1))
            second = self.attachments.geom_to_robot.get(int(contact.geom2))
            if first is None and second is None:
                continue
            if first is not None and first == second:
                continue
            if first is None or second is None:
                worst_world = min(worst_world, float(contact.dist))
            else:
                worst_pair = min(worst_pair, float(contact.dist))
        return worst_pair, worst_world

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
