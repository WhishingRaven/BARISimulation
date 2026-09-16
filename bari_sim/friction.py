"""Explicit fore-aft asymmetric anchor-pad contact model.

When enabled, ModelBuilder zeros MuJoCo sliding friction and this model supplies
anisotropic, smooth Coulomb forces at actual anchor-pad contact points. The
pad resists rearward slip more than forward recovery slip, making the mechanism
visible and configurable rather than a hidden root-body propulsion force.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from .config import DirectionalFrictionConfig


class DirectionalFrictionModel:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: DirectionalFrictionConfig,
        body_to_robot: Mapping[int, int],
        geom_to_link: Mapping[int, int],
    ):
        self.model = model
        self.data = data
        self.config = config
        self.body_to_robot = body_to_robot
        self.geom_to_link = geom_to_link
        self._anchor_direction_by_robot = {
            robot_id: 1.0 for robot_id in set(body_to_robot.values())
        }
        self._stance_links_by_robot: dict[int, frozenset[int] | None] = {
            robot_id: None for robot_id in set(body_to_robot.values())
        }

    def set_anchor_direction(self, robot_id: int, direction: float) -> None:
        """Select which way a robot's actively oriented anchor pads resist slip."""

        self._anchor_direction_by_robot[robot_id] = 1.0 if direction >= 0.0 else -1.0

    def set_stance_links(self, robot_id: int, link_ids: tuple[int, ...]) -> None:
        """Enable asymmetric grip only for the scheduled stance links."""

        self._stance_links_by_robot[robot_id] = frozenset(link_ids)

    def apply(self) -> None:
        if not self.config.enabled:
            return
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if not self._is_active_anchor_pad(
                int(contact.geom1)
            ) and not self._is_active_anchor_pad(int(contact.geom2)):
                continue
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            robot1 = body1 in self.body_to_robot
            robot2 = body2 in self.body_to_robot
            if not robot1 and not robot2:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
            normal_load = abs(float(wrench[0]))
            if normal_load <= 0:
                continue
            point = np.asarray(contact.pos, dtype=np.float64)
            velocity1 = (
                self._point_velocity(body1, point) if body1 != 0 else np.zeros(3)
            )
            velocity2 = (
                self._point_velocity(body2, point) if body2 != 0 else np.zeros(3)
            )
            relative_velocity = velocity1 - velocity2

            if robot1:
                force_on_1 = self._friction_force(body1, relative_velocity, normal_load)
            else:
                force_on_1 = -self._friction_force(
                    body2, -relative_velocity, normal_load
                )
            if robot1 and robot2:
                force_implied_by_2 = -self._friction_force(
                    body2, -relative_velocity, normal_load
                )
                force_on_1 = 0.5 * (force_on_1 + force_implied_by_2)
            if robot1:
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    force_on_1,
                    np.zeros(3),
                    point,
                    body1,
                    self.data.qfrc_applied,
                )
            if robot2:
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    -force_on_1,
                    np.zeros(3),
                    point,
                    body2,
                    self.data.qfrc_applied,
                )

    def _friction_force(
        self, body_id: int, velocity_world: np.ndarray, normal_load: float
    ) -> np.ndarray:
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        local_velocity = rotation.T @ velocity_world
        transition = max(self.config.transition_speed_m_s, 1e-6)
        robot_id = self.body_to_robot[body_id]
        directed_velocity = (
            self._anchor_direction_by_robot[robot_id] * local_velocity[0]
        )
        longitudinal_coefficient = (
            self.config.forward_sliding_coefficient
            if directed_velocity >= 0.0
            else self.config.reverse_sliding_coefficient
        )
        local_force = np.asarray(
            (
                -longitudinal_coefficient
                * normal_load
                * np.tanh(local_velocity[0] / transition),
                -self.config.lateral_coefficient
                * normal_load
                * np.tanh(local_velocity[1] / transition),
                0.0,
            )
        )
        return rotation @ local_force

    def _is_active_anchor_pad(self, geom_id: int) -> bool:
        name = self.model.geom(geom_id).name
        if name is None or "_anchor_pad_" not in name:
            return False
        body_id = int(self.model.geom_bodyid[geom_id])
        robot_id = self.body_to_robot.get(body_id)
        link_id = self.geom_to_link.get(geom_id)
        if robot_id is None or link_id is None:
            return False
        stance = self._stance_links_by_robot[robot_id]
        return stance is None or link_id in stance

    def _point_velocity(self, body_id: int, point: np.ndarray) -> np.ndarray:
        spatial = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            spatial,
            0,
        )
        angular = spatial[:3]
        linear = spatial[3:]
        center_of_mass = np.asarray(self.data.xipos[body_id], dtype=np.float64)
        return linear + np.cross(angular, point - center_of_mass)
