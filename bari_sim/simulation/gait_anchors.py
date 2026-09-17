"""Physical, alternating ground latches for the manual inchworm gait."""

from __future__ import annotations

import mujoco
import numpy as np

from ..robot.actions import MotionAction
from ..robot.specification import DEFAULT_ROBOT, RobotSpecification
from .scene import (
    LINK_NAMES,
    body_name,
    gait_anchor_site_name,
    gait_equality_name,
    gait_robot_equality_name,
    gait_target_site_name,
    robot_target_site_name,
)


class GaitAnchorSystem:
    """Latch one physical end-cleat to its current ground point per body phase.

    The latch is a MuJoCo ``connect`` equality between two sites.  It does not
    overwrite free-joint positions or velocities: all motion is solved from
    contact, hinge torque, and the active latch.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_count: int,
        robot: RobotSpecification = DEFAULT_ROBOT,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.robot_count = robot_count
        self._site_ids = {
            (robot_id, anchor): model.site(gait_anchor_site_name(robot_id, anchor)).id
            for robot_id in range(robot_count)
            for anchor in ("rear", "front")
        }
        self._target_site_ids = {
            (robot_id, anchor): model.site(gait_target_site_name(robot_id, anchor)).id
            for robot_id in range(robot_count)
            for anchor in ("rear", "front")
        }
        self._equality_ids = {
            (robot_id, anchor): model.equality(gait_equality_name(robot_id, anchor)).id
            for robot_id in range(robot_count)
            for anchor in ("rear", "front")
        }
        self._robot_equality_ids = {
            (source_id, anchor, target_id, link): model.equality(
                gait_robot_equality_name(source_id, anchor, target_id, link)
            ).id
            for source_id in range(robot_count)
            for anchor in ("rear", "front")
            for target_id in range(robot_count)
            if target_id != source_id
            for link in LINK_NAMES
        }
        self._body_to_link = {
            model.body(body_name(robot_id, link)).id: (robot_id, link)
            for robot_id in range(robot_count)
            for link in LINK_NAMES
        }

    def apply_motion(self, robot_id: int, motion: MotionAction) -> None:
        anchor = {
            MotionAction.CURL_BODY: "front",
            MotionAction.FLATTEN_BODY: "rear",
        }.get(motion)
        for candidate in ("rear", "front"):
            equality_id = self._equality_ids[(robot_id, candidate)]
            self.data.eq_active[equality_id] = 0
            for target_id in range(self.robot_count):
                if target_id == robot_id:
                    continue
                for link in LINK_NAMES:
                    self.data.eq_active[
                        self._robot_equality_ids[(robot_id, candidate, target_id, link)]
                    ] = 0
        if anchor is None:
            return

        source_site_id = self._site_ids[(robot_id, anchor)]
        source_body_id = self.model.body(body_name(robot_id, anchor)).id
        candidate = self._find_surface(source_site_id, source_body_id)
        if candidate is None:
            return
        target_body_id, target_robot_id, point = candidate
        if target_robot_id is not None:
            _, target_link = self._body_to_link[target_body_id]
            target_site_id = self.model.site(
                robot_target_site_name(robot_id, target_robot_id, target_link)
            ).id
            self.model.site_pos[target_site_id] = self._world_to_local(
                target_body_id, point
            )
            self.model.site_sameframe[target_site_id] = int(
                mujoco.mjtSameFrame.mjSAMEFRAME_NONE
            )
            equality_id = self._robot_equality_ids[
                (robot_id, anchor, target_robot_id, target_link)
            ]
            self.data.eq_active[equality_id] = 1
            return
        target_site_id = self._target_site_ids[(robot_id, anchor)]
        self.model.site_pos[target_site_id] = point
        self.model.site_sameframe[target_site_id] = int(
            mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        )
        self.data.eq_active[self._equality_ids[(robot_id, anchor)]] = 1

    def reset(self) -> None:
        for equality_id in self._equality_ids.values():
            self.data.eq_active[equality_id] = 0
        for equality_id in self._robot_equality_ids.values():
            self.data.eq_active[equality_id] = 0

    def _find_surface(
        self, source_site_id: int, source_body_id: int
    ) -> tuple[int, int | None, np.ndarray] | None:
        origin = np.asarray(self.data.site_xpos[source_site_id], dtype=np.float64)
        ray_origin = origin + np.asarray(
            (0.0, 0.0, self.robot.attachment_contact_tolerance_m), dtype=np.float64
        )
        hit_geom = np.asarray((-1,), dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            ray_origin,
            np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
            np.asarray((1, 1, 0, 0, 0, 0), dtype=np.uint8),
            True,
            source_body_id,
            hit_geom,
        )
        if distance < 0.0:
            return None
        target_geom_id = int(hit_geom[0])
        target_body_id = int(self.model.geom_bodyid[target_geom_id])
        target = self._body_to_link.get(target_body_id)
        point = ray_origin + np.asarray((0.0, 0.0, -distance), dtype=np.float64)
        if abs(point[2] - origin[2]) > self.robot.attachment_contact_tolerance_m:
            return None
        if target is None:
            return 0, None, point
        target_robot_id, _ = target
        return target_body_id, target_robot_id, point

    def _world_to_local(self, body_id: int, point: np.ndarray) -> np.ndarray:
        rotation = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        return rotation.T @ (point - self.data.xpos[body_id])
