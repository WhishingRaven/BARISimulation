"""Arbitrary-point, breakable gripper attachments using inactive equality slots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Protocol

import mujoco
import numpy as np

from .config import AttachmentConfig, RobotConfig
from .model_builder import (
    attachment_equality_name,
    attachment_target_site_name,
    gripper_geom_name,
    world_attachment_equality_name,
    world_attachment_site_name,
)
from .types import AttachmentEvent, AttachmentObservation


class FailureCriterion(Protocol):
    def utilization(self, force_n: float, moment_nm: float) -> float: ...


@dataclass(frozen=True)
class ForceFailureCriterion:
    maximum_force_n: float

    def utilization(self, force_n: float, moment_nm: float) -> float:
        del moment_nm
        return force_n / self.maximum_force_n


@dataclass(frozen=True)
class CombinedFailureCriterion:
    maximum_force_n: float
    maximum_moment_nm: float

    def utilization(self, force_n: float, moment_nm: float) -> float:
        return sqrt(
            (force_n / self.maximum_force_n) ** 2
            + (moment_nm / self.maximum_moment_nm) ** 2
        )


@dataclass
class Attachment:
    source_robot_id: int
    source_gripper: str
    target_robot_id: int | None
    target_body_id: int
    target_body_name: str
    target_local_point_m: tuple[float, float, float]
    source_local_point_m: tuple[float, float, float]
    equality_id: int
    source_site_id: int
    target_site_id: int
    active: bool = True
    force_n: float = 0.0
    moment_nm: float = 0.0
    maximum_force_seen_n: float = 0.0
    utilization: float = 0.0


class AttachmentManager:
    """Owns contact selection, dynamic point constraints, and failure events."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_config: RobotConfig,
        robot_count: int,
        geom_to_robot: Mapping[int, int],
        body_to_robot: Mapping[int, int],
        body_to_link: Mapping[int, int],
    ):
        self.model = model
        self.data = data
        self.config: AttachmentConfig = robot_config.attachment
        self.robot_count = robot_count
        self.geom_to_robot = geom_to_robot
        self.body_to_robot = body_to_robot
        self.body_to_link = body_to_link
        self.gripper_geom_ids = {
            robot_id: model.geom(gripper_geom_name(robot_id)).id
            for robot_id in range(robot_count)
        }
        self.gripper_site_ids = {
            robot_id: model.site(f"robot_{robot_id}_gripper_site").id
            for robot_id in range(robot_count)
        }
        self._default_gripper_site_positions = {
            robot_id: np.asarray(model.site_pos[site_id], dtype=np.float64).copy()
            for robot_id, site_id in self.gripper_site_ids.items()
        }
        self._default_gripper_sameframe = {
            robot_id: int(model.site_sameframe[site_id])
            for robot_id, site_id in self.gripper_site_ids.items()
        }
        self._active: dict[int, Attachment] = {}
        self._events: list[AttachmentEvent] = []
        self._all_equality_ids = tuple(
            eq_id
            for eq_id in range(model.neq)
            if self._equality_name(eq_id).startswith("attach_")
        )
        if self.config.failure_model == "combined":
            self.criterion: FailureCriterion = CombinedFailureCriterion(
                self.config.break_force_n,
                self.config.break_moment_nm,
            )
        else:
            self.criterion = ForceFailureCriterion(self.config.break_force_n)

    def attach(self, source_robot_id: int) -> bool:
        if source_robot_id in self._active:
            return True
        candidate = self._find_contact(source_robot_id)
        if candidate is None:
            self._events.append(
                AttachmentEvent(
                    simulation_time_s=float(self.data.time),
                    event="attach_missed",
                    source_robot_id=source_robot_id,
                    target_robot_id=None,
                    target_body="none",
                    detail="gripper had no eligible contact",
                )
            )
            return False

        target_body_id, target_robot_id, contact_point = candidate
        source_site_id = self.gripper_site_ids[source_robot_id]
        source_body_id = int(
            self.model.geom_bodyid[self.gripper_geom_ids[source_robot_id]]
        )
        source_rotation = np.asarray(
            self.data.xmat[source_body_id], dtype=np.float64
        ).reshape(3, 3)
        source_position = np.asarray(self.data.xpos[source_body_id], dtype=np.float64)
        source_local_point = source_rotation.T @ (contact_point - source_position)
        if target_robot_id is None:
            target_body_name = "world"
            site_name = world_attachment_site_name(source_robot_id)
            equality_name = world_attachment_equality_name(source_robot_id)
            local_point = contact_point
        else:
            target_body_name = self._body_name(target_body_id)
            site_name = attachment_target_site_name(source_robot_id, target_body_name)
            equality_name = attachment_equality_name(source_robot_id, target_body_name)
            rotation = np.asarray(
                self.data.xmat[target_body_id], dtype=np.float64
            ).reshape(3, 3)
            body_position = np.asarray(self.data.xpos[target_body_id], dtype=np.float64)
            local_point = rotation.T @ (contact_point - body_position)

        site_id = self.model.site(site_name).id
        equality_id = self.model.equality(equality_name).id
        self.model.site_sameframe[source_site_id] = int(
            mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        )
        self.model.site_sameframe[site_id] = int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
        self.model.site_pos[source_site_id] = source_local_point
        self.model.site_pos[site_id] = local_point
        self.model.site_rgba[site_id, 3] = 1.0
        self.data.eq_active[equality_id] = 1
        attachment = Attachment(
            source_robot_id=source_robot_id,
            source_gripper=gripper_geom_name(source_robot_id),
            target_robot_id=target_robot_id,
            target_body_id=target_body_id,
            target_body_name=target_body_name,
            target_local_point_m=tuple(float(value) for value in local_point),
            source_local_point_m=tuple(float(value) for value in source_local_point),
            equality_id=equality_id,
            source_site_id=source_site_id,
            target_site_id=site_id,
        )
        self._active[source_robot_id] = attachment
        mujoco.mj_forward(self.model, self.data)
        self._events.append(
            AttachmentEvent(
                simulation_time_s=float(self.data.time),
                event="attached",
                source_robot_id=source_robot_id,
                target_robot_id=target_robot_id,
                target_body=target_body_name,
                detail="contact point stored in target body coordinates",
            )
        )
        return True

    def detach(self, source_robot_id: int, reason: str = "commanded") -> bool:
        attachment = self._active.pop(source_robot_id, None)
        if attachment is None:
            return False
        self.data.eq_active[attachment.equality_id] = 0
        self.model.site_pos[attachment.source_site_id] = (
            self._default_gripper_site_positions[source_robot_id]
        )
        self.model.site_sameframe[attachment.source_site_id] = (
            self._default_gripper_sameframe[source_robot_id]
        )
        self.model.site_rgba[attachment.target_site_id, 3] = 0.0
        attachment.active = False
        self._events.append(
            AttachmentEvent(
                simulation_time_s=float(self.data.time),
                event="attachment_failure" if reason == "failure" else "detached",
                source_robot_id=source_robot_id,
                target_robot_id=attachment.target_robot_id,
                target_body=attachment.target_body_name,
                force_n=attachment.force_n,
                moment_nm=attachment.moment_nm,
                detail=reason,
            )
        )
        return True

    def post_step(self) -> None:
        failed: list[int] = []
        for source_robot_id, attachment in self._active.items():
            force_n, moment_nm = self._constraint_load(attachment.equality_id)
            attachment.force_n = force_n
            attachment.moment_nm = moment_nm
            attachment.maximum_force_seen_n = max(
                attachment.maximum_force_seen_n, force_n
            )
            attachment.utilization = self.criterion.utilization(force_n, moment_nm)
            if attachment.utilization > 1.0:
                failed.append(source_robot_id)
        for source_robot_id in failed:
            self.detach(source_robot_id, reason="failure")

    def get_attachment(self, source_robot_id: int) -> Attachment | None:
        return self._active.get(source_robot_id)

    def get_state(self, source_robot_id: int) -> AttachmentObservation:
        attachment = self._active.get(source_robot_id)
        if attachment is None:
            return AttachmentObservation()
        return AttachmentObservation(
            active=True,
            force_n=attachment.force_n,
            moment_nm=attachment.moment_nm,
            utilization=attachment.utilization,
            target_is_robot=attachment.target_robot_id is not None,
        )

    def target_world_point(self, attachment: Attachment) -> np.ndarray:
        if attachment.target_robot_id is None:
            return np.asarray(attachment.target_local_point_m, dtype=np.float64)
        rotation = np.asarray(
            self.data.xmat[attachment.target_body_id], dtype=np.float64
        ).reshape(3, 3)
        position = np.asarray(
            self.data.xpos[attachment.target_body_id], dtype=np.float64
        )
        return position + rotation @ np.asarray(
            attachment.target_local_point_m, dtype=np.float64
        )

    def active_attachments(self) -> tuple[Attachment, ...]:
        return tuple(self._active[source_id] for source_id in sorted(self._active))

    def drain_events(self) -> tuple[AttachmentEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def reset(self) -> None:
        for equality_id in self._all_equality_ids:
            self.data.eq_active[equality_id] = 0
        for attachment in self._active.values():
            self.model.site_rgba[attachment.target_site_id, 3] = 0.0
        for robot_id, site_id in self.gripper_site_ids.items():
            self.model.site_pos[site_id] = self._default_gripper_site_positions[
                robot_id
            ]
            self.model.site_sameframe[site_id] = self._default_gripper_sameframe[
                robot_id
            ]
        self._active.clear()
        self._events.clear()

    def _find_contact(
        self,
        source_robot_id: int,
    ) -> tuple[int, int | None, np.ndarray] | None:
        gripper_geom_id = self.gripper_geom_ids[source_robot_id]
        candidates: list[tuple[float, int, int | None, np.ndarray]] = []
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if int(contact.geom1) == gripper_geom_id:
                other_geom_id = int(contact.geom2)
            elif int(contact.geom2) == gripper_geom_id:
                other_geom_id = int(contact.geom1)
            else:
                continue
            target_robot_id = self.geom_to_robot.get(other_geom_id)
            if target_robot_id == source_robot_id:
                continue
            target_body_id = int(self.model.geom_bodyid[other_geom_id])
            if target_robot_id is None:
                if not self.config.allow_environment:
                    continue
                target_body_id = 0
            elif target_body_id not in self.body_to_link:
                continue
            candidates.append(
                (
                    float(contact.dist),
                    target_body_id,
                    target_robot_id,
                    np.asarray(contact.pos, dtype=np.float64).copy(),
                )
            )
        if not candidates:
            return None
        # A robot contact is the primary gripper use-case. Prefer it over a
        # simultaneous ground/platform contact, then choose deepest contact.
        _, body_id, robot_id, point = min(
            candidates,
            key=lambda item: (item[2] is None, item[0]),
        )
        return body_id, robot_id, point

    def _constraint_load(self, equality_id: int) -> tuple[float, float]:
        count = int(self.data.nefc)
        if count == 0:
            return 0.0, 0.0
        equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        rows = np.flatnonzero(
            (np.asarray(self.data.efc_type[:count]) == equality_type)
            & (np.asarray(self.data.efc_id[:count]) == equality_id)
        )
        if rows.size == 0:
            return 0.0, 0.0
        scalar_forces = np.asarray(self.data.efc_force[rows], dtype=np.float64)
        # A connect constraint contributes three translational rows. Their
        # Euclidean norm is reported in newtons. It transmits no moment.
        return float(np.linalg.norm(scalar_forces)), 0.0

    def _body_name(self, body_id: int) -> str:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is None:
            raise RuntimeError(f"target body {body_id} has no name")
        return name

    def _equality_name(self, equality_id: int) -> str:
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id)
            or ""
        )
