"""Local surface-force and range sensing.

Sensors derive readings from MuJoCo contacts/rays. They expose no global pose.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, radians, sin

import mujoco
import numpy as np

from .config import RangeSensorConfig, RobotConfig
from .model_builder import (
    FORCE_SENSOR_LOCATIONS,
    force_sensor_key,
    force_sensor_name,
    gripper_geom_name,
    link_body_name,
    range_origin_site_name,
)
from .types import ForceReading, RangeReading


@dataclass(frozen=True)
class ForceSnapshot:
    readings: Mapping[str, ForceReading]
    gripper_contact: bool
    neighboring_robot_ids: tuple[int, ...]


class SensorSuite:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_id: int,
        config: RobotConfig,
        geom_to_robot: Mapping[int, int],
        geom_to_link: Mapping[int, int],
    ):
        self.model = model
        self.data = data
        self.robot_id = robot_id
        self.config = config
        self.geom_to_robot = geom_to_robot
        self.geom_to_link = geom_to_link
        self.link_body_ids = tuple(
            model.body(link_body_name(robot_id, link_id)).id
            for link_id in range(config.link_count)
        )
        self.gripper_geom_id = model.geom(gripper_geom_name(robot_id)).id
        self.owned_geoms = frozenset(
            geom_id
            for geom_id, owner_id in self.geom_to_robot.items()
            if owner_id == self.robot_id
        )
        self.force_sensor_names = {
            force_sensor_key(link_id, location): force_sensor_name(
                robot_id, link_id, location
            )
            for link_id in range(config.link_count)
            for location in FORCE_SENSOR_LOCATIONS
        }
        self.range_origin_site_id = model.site(range_origin_site_name(robot_id)).id
        self._range_cache: dict[str, RangeReading] = {
            sensor.name: RangeReading(sensor.max_range_m, False, None)
            for sensor in config.range_sensors
        }
        self._last_update: dict[str, float] = {
            sensor.name: float("-inf") for sensor in config.range_sensors
        }
        self.last_rays: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
        # Environment geoms use group 0; collision geoms use group 3.
        self._ray_geom_groups = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.uint8)
        self.rng = np.random.default_rng(robot_id)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed + 104729 * self.robot_id)

    def reset(self) -> None:
        self._last_update = {
            sensor.name: float("-inf") for sensor in self.config.range_sensors
        }
        self._range_cache = {
            sensor.name: RangeReading(sensor.max_range_m, False, None)
            for sensor in self.config.range_sensors
        }
        self.last_rays.clear()

    def read_force_sensors(self) -> ForceSnapshot:
        force_by_sensor: dict[str, float] = defaultdict(float)
        neighbors: set[int] = set()
        gripper_contact = False

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            sides: list[tuple[int, int, float]] = []
            if contact.geom1 in self.owned_geoms:
                sides.append((int(contact.geom1), int(contact.geom2), 1.0))
            if contact.geom2 in self.owned_geoms:
                sides.append((int(contact.geom2), int(contact.geom1), -1.0))
            if not sides:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
            normal_force = abs(float(wrench[0]))
            contact_normal = np.asarray(contact.frame[:3], dtype=np.float64)
            point = np.asarray(contact.pos, dtype=np.float64)
            for owned_geom, other_geom, normal_sign in sides:
                if owned_geom == self.gripper_geom_id:
                    gripper_contact = True
                # The gripper is rigidly part of the final link, so route its
                # external load to that link's nearest surface channel too.
                sensor_key = self._surface_sensor_key(
                    owned_geom, normal_sign * contact_normal, point
                )
                if sensor_key is not None:
                    force_by_sensor[sensor_key] += normal_force
                other_robot = self.geom_to_robot.get(other_geom)
                if other_robot is not None and other_robot != self.robot_id:
                    neighbors.add(other_robot)

        readings = {
            name: ForceReading(force_n=force_by_sensor[name])
            for name in self.force_sensor_names
        }
        return ForceSnapshot(readings, gripper_contact, tuple(sorted(neighbors)))

    def _surface_sensor_key(
        self,
        owned_geom: int,
        outward_normal_world: np.ndarray,
        contact_point_world: np.ndarray,
    ) -> str | None:
        link_id = self.geom_to_link[owned_geom]
        body_id = self.link_body_ids[link_id]
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        local_normal = rotation.T @ outward_normal_world
        dominant_axis = int(np.argmax(np.abs(local_normal)))
        if dominant_axis == 2 and local_normal[2] > 0.0:
            local_point = rotation.T @ (
                contact_point_world - np.asarray(self.data.xpos[body_id])
            )
            midpoint = self.config.link_lengths_m[link_id] / 2.0
            location = "top_rear" if local_point[0] < midpoint else "top_front"
        elif dominant_axis == 1:
            location = "left" if local_normal[1] > 0.0 else "right"
        else:
            return None
        return force_sensor_key(link_id, location)

    def read_ranges(self, simulation_time_s: float) -> Mapping[str, RangeReading]:
        origin = np.asarray(
            self.data.site_xpos[self.range_origin_site_id], dtype=np.float64
        ).copy()
        front_body_id = self.link_body_ids[-1]
        rotation = np.asarray(self.data.xmat[front_body_id], dtype=np.float64).reshape(
            3, 3
        )
        for sensor in self.config.range_sensors:
            period = 1.0 / sensor.update_rate_hz
            if simulation_time_s - self._last_update[sensor.name] + 1e-12 < period:
                continue
            reading, endpoint = self._cast_sensor(
                sensor, origin, rotation, front_body_id
            )
            self._range_cache[sensor.name] = reading
            self._last_update[sensor.name] = simulation_time_s
            self.last_rays[sensor.name] = (origin.copy(), endpoint, reading.detected)
        return dict(self._range_cache)

    def _cast_sensor(
        self,
        sensor: RangeSensorConfig,
        origin: np.ndarray,
        rotation: np.ndarray,
        excluded_body_id: int,
    ) -> tuple[RangeReading, np.ndarray]:
        local = np.asarray(sensor.direction_local, dtype=np.float64)
        norm = float(np.linalg.norm(local))
        if norm <= 1e-12:
            raise ValueError(f"range sensor {sensor.name!r} has zero direction")
        local /= norm
        candidate_directions = [local]
        if sensor.field_of_view_deg > 0:
            half = radians(sensor.field_of_view_deg / 2.0)
            candidate_directions.extend(
                (self._yaw(local, half), self._yaw(local, -half))
            )

        best_distance = sensor.max_range_m
        best_geom = -1
        best_direction = rotation @ local
        for candidate in candidate_directions:
            world_direction = rotation @ candidate
            world_direction /= np.linalg.norm(world_direction)
            geom_id = np.full(1, -1, dtype=np.int32)
            distance = float(
                mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    world_direction,
                    self._ray_geom_groups,
                    1,
                    excluded_body_id,
                    geom_id,
                )
            )
            if 0.0 <= distance < best_distance:
                best_distance = distance
                best_geom = int(geom_id[0])
                best_direction = world_direction

        detected = best_geom >= 0 and best_distance <= sensor.max_range_m
        if sensor.noise_std_m > 0:
            best_distance += float(self.rng.normal(0.0, sensor.noise_std_m))
        best_distance = min(max(best_distance, 0.0), sensor.max_range_m)
        hit_robot = self.geom_to_robot.get(best_geom) if detected else None
        if hit_robot == self.robot_id:
            hit_robot = None
        endpoint = origin + best_direction * best_distance
        return RangeReading(best_distance, detected, hit_robot), endpoint

    @staticmethod
    def _yaw(vector: np.ndarray, angle: float) -> np.ndarray:
        c, s = cos(angle), sin(angle)
        return np.asarray(
            (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]),
            dtype=np.float64,
        )
