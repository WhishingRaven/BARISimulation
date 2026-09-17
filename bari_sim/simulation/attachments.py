"""Contact-gated, breakable rear spike-line attachments."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..robot.actions import GripAction
from ..robot.specification import DEFAULT_ROBOT, GRAVITY_M_S2, RobotSpecification
from .scene import (
    LINK_NAMES,
    body_name,
    geom_name,
    robot_equality_name,
    robot_target_site_name,
    spike_site_name,
    world_equality_name,
    world_target_site_name,
)


@dataclass
class Attachment:
    source_robot_id: int
    target_robot_id: int | None
    target_body_id: int
    equality_id: int
    source_site_id: int
    target_site_id: int
    surface_normal_world: np.ndarray
    root_qpos: np.ndarray
    root_qvel: np.ndarray
    force_n: float = 0.0
    overload_duration_s: float = 0.0


@dataclass(frozen=True)
class AttachmentEvent:
    event: str
    source_robot_id: int
    target_robot_id: int | None
    strain_value: float
    caused_by_other_robot: bool = False


class AttachmentManager:
    # Contact resolution can emit a one-physics-step force spike when another
    # robot climbs on.  A material overload must persist, rather than being a
    # single solver impulse.
    _OVERLOAD_DURATION_S = 0.05

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_count: int,
        robot: RobotSpecification = DEFAULT_ROBOT,
    ):
        self.model = model
        self.data = data
        self.robot_count = robot_count
        self.robot = robot
        self.rear_geom_ids = {
            robot_id: model.geom(geom_name(robot_id, "rear")).id
            for robot_id in range(robot_count)
        }
        self.geom_to_robot: dict[int, int] = {}
        self.body_to_robot: dict[int, int] = {}
        self.body_to_link: dict[int, str] = {}
        for robot_id in range(robot_count):
            for link in LINK_NAMES:
                geom_id = model.geom(geom_name(robot_id, link)).id
                body_id = model.body(body_name(robot_id, link)).id
                self.geom_to_robot[geom_id] = robot_id
                self.body_to_robot[body_id] = robot_id
                self.body_to_link[body_id] = link
        self.spike_site_ids = {
            robot_id: model.site(spike_site_name(robot_id)).id
            for robot_id in range(robot_count)
        }
        self._default_spike_positions = {
            robot_id: np.asarray(model.site_pos[site_id]).copy()
            for robot_id, site_id in self.spike_site_ids.items()
        }
        self._active: dict[int, Attachment] = {}
        self._last_strain_g = np.zeros(robot_count, dtype=np.float64)
        self._last_label_positions: list[np.ndarray | None] = [
            None for _ in range(robot_count)
        ]
        self._detached_by_other = np.zeros(robot_count, dtype=np.bool_)
        self._events: list[AttachmentEvent] = []
        self._all_equality_ids = tuple(range(model.neq))

    def begin_control_step(self) -> None:
        self._detached_by_other[:] = False
        self._events.clear()

    def apply_command(self, robot_id: int, command: GripAction) -> None:
        if command is GripAction.DETACH:
            self.detach(robot_id, caused_by_other=False, overloaded=False)
        elif command is GripAction.ATTACH:
            self.attach(robot_id)

    def attach(self, robot_id: int) -> bool:
        if robot_id in self._active:
            return True
        candidate = self._find_candidate(robot_id)
        if candidate is None:
            return False
        target_body_id, target_robot_id, world_point, surface_normal_world = candidate
        source_site_id = self.spike_site_ids[robot_id]
        source_body_id = self.model.body(body_name(robot_id, "rear")).id
        root_joint_id = int(self.model.body_jntadr[source_body_id])
        root_qpos_address = int(self.model.jnt_qposadr[root_joint_id])
        source_local = self._world_to_local(source_body_id, world_point)
        # The physical spike is a line across the rear edge.  Moving this
        # non-physical site along y selects the actual point on that line.
        source_local[0] = -self.robot.rear_length_m / 2.0
        source_local[2] = -self.robot.height_m / 2.0
        self.model.site_pos[source_site_id] = source_local

        if target_robot_id is None:
            target_site_id = self.model.site(world_target_site_name(robot_id)).id
            equality_id = self.model.equality(world_equality_name(robot_id)).id
            self.model.site_pos[target_site_id] = world_point
        else:
            target_link = self.body_to_link[target_body_id]
            target_site_id = self.model.site(
                robot_target_site_name(robot_id, target_robot_id, target_link)
            ).id
            equality_id = self.model.equality(
                robot_equality_name(robot_id, target_robot_id, target_link)
            ).id
            self.model.site_pos[target_site_id] = self._world_to_local(
                target_body_id, world_point
            )
        self.model.site_sameframe[source_site_id] = int(
            mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        )
        self.model.site_sameframe[target_site_id] = int(
            mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        )
        self.data.eq_active[equality_id] = 1
        self._active[robot_id] = Attachment(
            source_robot_id=robot_id,
            target_robot_id=target_robot_id,
            target_body_id=target_body_id,
            equality_id=equality_id,
            source_site_id=source_site_id,
            target_site_id=target_site_id,
            surface_normal_world=surface_normal_world,
            root_qpos=self.data.qpos[root_qpos_address : root_qpos_address + 7].copy(),
            root_qvel=np.zeros(6, dtype=np.float64),
        )
        self._last_strain_g[robot_id] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_label_positions[robot_id] = np.asarray(
            self.data.site_xpos[source_site_id]
        ).copy()
        self._events.append(AttachmentEvent("attached", robot_id, target_robot_id, 0.0))
        return True

    def detach(self, robot_id: int, *, caused_by_other: bool, overloaded: bool) -> bool:
        attachment = self._active.pop(robot_id, None)
        if attachment is None:
            return False
        self._last_label_positions[robot_id] = np.asarray(
            self.data.site_xpos[attachment.source_site_id]
        ).copy()
        self.data.eq_active[attachment.equality_id] = 0
        self.model.site_pos[attachment.source_site_id] = self._default_spike_positions[
            robot_id
        ]
        if not overloaded:
            self._last_strain_g[robot_id] = 0.0
        self._detached_by_other[robot_id] = caused_by_other
        self._events.append(
            AttachmentEvent(
                "overload_detached" if overloaded else "detached",
                robot_id,
                attachment.target_robot_id,
                float(self._last_strain_g[robot_id]),
                caused_by_other,
            )
        )
        return True

    def post_physics_step(self) -> None:
        failures: list[tuple[int, bool]] = []
        for robot_id, attachment in tuple(self._active.items()):
            constraint_force_n = self._normal_constraint_force(
                attachment.equality_id, attachment.surface_normal_world
            )
            external_force_n = abs(
                float(
                    np.dot(
                        self.data.xfrc_applied[attachment.target_body_id, :3],
                        attachment.surface_normal_world,
                    )
                )
            )
            source_body_id = self.model.body(
                body_name(attachment.source_robot_id, "rear")
            ).id
            external_force_n = max(
                external_force_n,
                abs(
                    float(
                        np.dot(
                            self.data.xfrc_applied[source_body_id, :3],
                            attachment.surface_normal_world,
                        )
                    )
                ),
            )
            force_n = max(constraint_force_n, external_force_n)
            attachment.force_n = force_n
            strain_g = force_n / GRAVITY_M_S2 * 1000.0
            self._last_strain_g[robot_id] = min(strain_g, self.robot.maximum_strain_g)
            self._last_label_positions[robot_id] = np.asarray(
                self.data.site_xpos[attachment.source_site_id]
            ).copy()
            if force_n > self.robot.maximum_attachment_force_n:
                attachment.overload_duration_s += self.model.opt.timestep
            else:
                attachment.overload_duration_s = 0.0
            if attachment.overload_duration_s >= self._OVERLOAD_DURATION_S:
                caused_by_other = (
                    attachment.target_robot_id is not None
                    or self._touching_other_robot(robot_id)
                )
                failures.append((robot_id, caused_by_other))
        for robot_id, caused_by_other in failures:
            self.detach(robot_id, caused_by_other=caused_by_other, overloaded=True)

    def lock_active_roots(self) -> bool:
        """Keep a non-overloaded attachment rigid at its latched world pose."""

        if not self._active:
            return False
        for attachment in self._active.values():
            # A robot-to-robot equality already constrains both bodies.  Also
            # overwriting the source free joint over-constrains an occupied
            # contact, which turns normal settling on another robot into a
            # large solver impulse.
            if attachment.target_robot_id is not None:
                continue
            source_body_id = self.model.body(
                body_name(attachment.source_robot_id, "rear")
            ).id
            root_joint_id = int(self.model.body_jntadr[source_body_id])
            qpos_address = int(self.model.jnt_qposadr[root_joint_id])
            dof_address = int(self.model.jnt_dofadr[root_joint_id])
            self.data.qpos[qpos_address : qpos_address + 7] = attachment.root_qpos
            self.data.qvel[dof_address : dof_address + 6] = attachment.root_qvel
        return True

    def is_possible(self, robot_id: int) -> bool:
        return robot_id in self._active or self._find_candidate(robot_id) is not None

    def active_labels(self) -> tuple[tuple[np.ndarray, float], ...]:
        """Return strain labels only for currently active attachments."""

        return tuple(
            (
                np.asarray(self.data.site_xpos[attachment.source_site_id]).copy(),
                float(self._last_strain_g[robot_id]),
            )
            for robot_id, attachment in sorted(self._active.items())
        )

    def is_attaching(self, robot_id: int) -> bool:
        return robot_id in self._active

    def is_detached(self, robot_id: int) -> bool:
        return bool(self._detached_by_other[robot_id])

    def strain_value(self, robot_id: int) -> float:
        if (
            robot_id not in self._active
            and not self._detached_by_other[robot_id]
            and self._last_strain_g[robot_id] == 0.0
        ):
            return 0.0
        return float(self._last_strain_g[robot_id])

    def drain_events(self) -> tuple[AttachmentEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def reset(self) -> None:
        for equality_id in self._all_equality_ids:
            self.data.eq_active[equality_id] = 0
        for robot_id, site_id in self.spike_site_ids.items():
            self.model.site_pos[site_id] = self._default_spike_positions[robot_id]
        self._active.clear()
        self._last_strain_g[:] = 0.0
        self._last_label_positions = [None for _ in range(self.robot_count)]
        self._detached_by_other[:] = False
        self._events.clear()

    def _find_candidate(
        self, robot_id: int
    ) -> tuple[int, int | None, np.ndarray, np.ndarray] | None:
        rear_geom_id = self.rear_geom_ids[robot_id]
        rear_body_id = self.model.body(body_name(robot_id, "rear")).id
        candidates: list[
            tuple[bool, float, int, int | None, np.ndarray, np.ndarray]
        ] = []
        rear_edge = -self.robot.rear_length_m / 2.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if int(contact.geom1) == rear_geom_id:
                other_geom_id = int(contact.geom2)
            elif int(contact.geom2) == rear_geom_id:
                other_geom_id = int(contact.geom1)
            else:
                continue
            target_robot_id = self.geom_to_robot.get(other_geom_id)
            if target_robot_id == robot_id:
                continue
            point = np.asarray(contact.pos, dtype=np.float64).copy()
            local = self._world_to_local(rear_body_id, point)
            if local[0] > rear_edge + self.robot.attachment_contact_tolerance_m:
                continue
            if abs(local[1]) > self.robot.width_m / 2.0 + 0.003:
                continue
            target_body_id = int(self.model.geom_bodyid[other_geom_id])
            if target_robot_id is None:
                target_body_id = 0
            candidates.append(
                (
                    target_robot_id is None,
                    float(contact.dist),
                    target_body_id,
                    target_robot_id,
                    point,
                    np.asarray(contact.frame[:3], dtype=np.float64).copy(),
                )
            )
        if not candidates:
            return self._find_nearby_surface(robot_id)
        _, _, body_id, target_id, point, normal = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        return body_id, target_id, point, normal

    def _find_nearby_surface(
        self, robot_id: int
    ) -> tuple[int, int | None, np.ndarray, np.ndarray] | None:
        """Find a floor or robot-top surface just below the rear spike."""

        site_id = self.spike_site_ids[robot_id]
        source_body_id = self.model.body(body_name(robot_id, "rear")).id
        origin = np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()
        hit_geom = np.asarray((-1,), dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
            np.asarray((1, 1, 0, 0, 0, 0), dtype=np.uint8),
            True,
            source_body_id,
            hit_geom,
        )
        if distance < 0.0 or distance > self.robot.attachment_contact_tolerance_m:
            return None
        target_geom_id = int(hit_geom[0])
        target_robot_id = self.geom_to_robot.get(target_geom_id)
        if target_robot_id == robot_id:
            return None
        target_body_id = (
            0
            if target_robot_id is None
            else int(self.model.geom_bodyid[target_geom_id])
        )
        point = origin + np.asarray((0.0, 0.0, -distance), dtype=np.float64)
        return target_body_id, target_robot_id, point, np.asarray(
            (0.0, 0.0, 1.0), dtype=np.float64
        )

    def _normal_constraint_force(
        self, equality_id: int, surface_normal_world: np.ndarray
    ) -> float:
        count = int(self.data.nefc)
        if count == 0:
            return 0.0
        rows = np.flatnonzero(
            (
                np.asarray(self.data.efc_type[:count])
                == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            )
            & (np.asarray(self.data.efc_id[:count]) == equality_id)
        )
        if rows.size == 0:
            return 0.0
        constraint_force = np.asarray(self.data.efc_force[rows], dtype=np.float64)
        normal = surface_normal_world / np.linalg.norm(surface_normal_world)
        return abs(float(np.dot(constraint_force[:3], normal)))

    def _touching_other_robot(self, robot_id: int) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.geom_to_robot.get(int(contact.geom1))
            second = self.geom_to_robot.get(int(contact.geom2))
            if first == robot_id and second is not None and second != robot_id:
                return True
            if second == robot_id and first is not None and first != robot_id:
                return True
        return False

    def _world_to_local(self, body_id: int, point: np.ndarray) -> np.ndarray:
        if body_id == 0:
            return point.copy()
        rotation = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        position = np.asarray(self.data.xpos[body_id])
        return rotation.T @ (point - position)
