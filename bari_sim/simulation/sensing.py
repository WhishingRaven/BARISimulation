"""Four idealized ultrasonic rays and local ID communication."""

from __future__ import annotations

import mujoco
import numpy as np

from ..robot.observation import RobotObservation
from ..robot.specification import DEFAULT_ROBOT, RobotSpecification
from .attachments import AttachmentManager
from .scene import body_name, hinge_name, sensor_site_name

SENSOR_ORDER = ("front", "down", "left", "right")
SENSOR_DIRECTIONS = {
    "front": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    "down": np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    "left": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
    "right": np.asarray((0.0, -1.0, 0.0), dtype=np.float64),
}


class SensorSystem:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_count: int,
        attachments: AttachmentManager,
        robot: RobotSpecification = DEFAULT_ROBOT,
    ):
        self.model = model
        self.data = data
        self.robot_count = robot_count
        self.attachments = attachments
        self.robot = robot
        self.root_body_ids = tuple(
            model.body(body_name(robot_id, "rear")).id
            for robot_id in range(robot_count)
        )
        self.hinge_ids = {
            (robot_id, hinge): model.joint(hinge_name(robot_id, hinge)).id
            for robot_id in range(robot_count)
            for hinge in ("rear", "front")
        }
        self.site_ids = {
            (robot_id, sensor): model.site(sensor_site_name(robot_id, sensor)).id
            for robot_id in range(robot_count)
            for sensor in SENSOR_ORDER
        }
        self.geom_to_robot = attachments.geom_to_robot
        # Environment group 0 and robot collision group 3 are visible.
        self._ray_groups = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.uint8)

    def read(self, robot_id: int) -> RobotObservation:
        distances = tuple(self._read_range(robot_id, sensor) for sensor in SENSOR_ORDER)
        rear_angle = self._joint_position(robot_id, "rear")
        front_angle = self._joint_position(robot_id, "front")
        return RobotObservation(
            nearby_robot_ids=self._nearby_robot_ids(robot_id),
            strain_value=self.attachments.strain_value(robot_id),
            distance1=distances[0],
            distance2=distances[1],
            distance3=distances[2],
            distance4=distances[3],
            is_curled=bool(
                abs(rear_angle - self.robot.curl_angle_rad) <= np.deg2rad(8.0)
            ),
            is_front_lifted=bool(
                abs(front_angle - self.robot.front_lift_angle_rad) <= np.deg2rad(8.0)
            ),
            is_possible_to_attach=self.attachments.is_possible(robot_id),
            is_attaching=self.attachments.is_attaching(robot_id),
            is_detached=self.attachments.is_detached(robot_id),
        )

    def read_all(self) -> dict[int, RobotObservation]:
        return {robot_id: self.read(robot_id) for robot_id in range(self.robot_count)}

    def _read_range(self, robot_id: int, sensor: str) -> float:
        site_id = self.site_ids[(robot_id, sensor)]
        origin = np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = np.asarray(self.data.site_xmat[site_id], dtype=np.float64).reshape(
            3, 3
        )
        direction = rotation @ SENSOR_DIRECTIONS[sensor]
        # Start just inside the owning (excluded) body.  This still reports a
        # near-zero floor distance when the underside is resting on a surface.
        origin -= direction * 1e-4
        hit_geom = np.asarray((-1,), dtype=np.int32)
        site_body_id = int(self.model.site_bodyid[site_id])
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            direction,
            self._ray_groups,
            True,
            site_body_id,
            hit_geom,
        )
        if distance < 0.0:
            return self.robot.sensor_range_m
        return min(float(distance), self.robot.sensor_range_m)

    def _nearby_robot_ids(self, robot_id: int) -> tuple[int, ...]:
        source = np.asarray(self.data.xpos[self.root_body_ids[robot_id]])
        nearby = []
        for other_id, body_id in enumerate(self.root_body_ids):
            if other_id == robot_id:
                continue
            distance = float(np.linalg.norm(self.data.xpos[body_id] - source))
            if distance <= self.robot.communication_range_m:
                nearby.append(other_id)
        return tuple(nearby)

    def _joint_position(self, robot_id: int, hinge: str) -> float:
        joint_id = self.hinge_ids[(robot_id, hinge)]
        return float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
