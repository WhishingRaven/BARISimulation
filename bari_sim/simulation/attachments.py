"""Contact-gated, breakable rear spike-line attachments."""

from __future__ import annotations

import os
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


def _debug_is_on() -> bool:
    value = os.environ.get("BARI_STRAIN_DEBUG", "").strip().strip("'\"").lower()
    return value in {"1", "true", "yes", "on", "y"}


@dataclass
class Attachment:
    source_robot_id: int
    target_robot_id: int | None
    target_body_id: int
    equality_id: int
    source_site_id: int
    target_site_id: int
    root_qpos: np.ndarray
    root_qvel: np.ndarray
    force_n: float = 0.0
    filtered_force_n: float = 0.0
    age_s: float = 0.0
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
    # 결합 직후 몇 밀리초는 구속이 자리를 잡는 과도구간이다.  이 동안의 힘은
    # 실제 하중이 아니므로 과부하 판정에 넣지 않는다. (BARI_STRAIN_SETTLE_S)
    _ACTIVATION_SETTLE_S = float(os.environ.get("BARI_STRAIN_SETTLE_S", "0.02"))
    # 실제 스트레인 게이지처럼 대역폭이 있다.  솔버 한 스텝짜리 스파이크가
    # 그대로 100 g로 찍히지 않도록 1차 저역통과로 읽는다. (BARI_STRAIN_TAU_S)
    _STRAIN_TAU_S = float(os.environ.get("BARI_STRAIN_TAU_S", "0.01"))
    # 결합 직전 침투를 풀 때 남겨 두는 여유 (m).  0으로 두면 수치 오차 때문에
    # 다시 살짝 파고든 상태로 붙을 수 있다.
    _PENETRATION_MARGIN_M = 2.0e-5
    # 한 번에 들어올릴 수 있는 최대량 (m).  모델의 margin 설정이 잘못돼 있으면
    # 해소량이 수 mm까지 나올 수 있는데, 그만큼 순간이동시키면 시뮬레이션이
    # 터진다.  해소는 어디까지나 보조 수단이므로 여기서 자른다.
    _RELIEF_LIMIT_M = float(os.environ.get("BARI_ATTACH_RELIEF_MAX_MM", "1.0")) / 1000.0

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
        self._last_force_breakdown: dict[int, tuple[float, float, float]] = {}
        # 제어 스텝(0.5 s) 동안의 최대값.  결합이 스텝 중간에 생겼다 끊겨도
        # 기록이 남아, 무슨 일이 있었는지 볼 수 있다.
        self._step_peak: dict[int, tuple[float, float, float]] = {}
        self._last_label_positions: list[np.ndarray | None] = [
            None for _ in range(robot_count)
        ]
        self._detached_by_other = np.zeros(robot_count, dtype=np.bool_)
        self._events: list[AttachmentEvent] = []
        self._all_equality_ids = tuple(range(model.neq))

    def begin_control_step(self) -> None:
        self._detached_by_other[:] = False
        self._events.clear()
        self._step_peak.clear()

    def apply_command(self, robot_id: int, command: GripAction) -> None:
        if command is GripAction.DETACH:
            self.detach(robot_id, caused_by_other=False, overloaded=False)
        elif command is GripAction.ATTACH:
            self.attach(robot_id)
        elif command is GripAction.STOP:
            pass

    def attach(self, robot_id: int) -> bool:
        if robot_id in self._active:
            return True
        candidate = self._find_candidate(robot_id)
        if candidate is None:
            return False
        target_body_id, target_robot_id, world_point = candidate
        # 가시를 박는 순간 로봇이 표면을 조금이라도 파고들어 있으면, 구속이 그
        # 깊이를 영구히 붙잡아 버린다.  단단한 접촉은 0.1 mm만 눌려 있어도 수십
        # N을 밀어내므로, 구속은 그 힘과 계속 맞서게 되고 곧바로 과부하가 된다.
        # 그래서 결합 전에 침투를 먼저 풀어 준다.
        if _debug_is_on():
            self._report_contacts(robot_id)
        relief_m = self._relieve_penetration(robot_id, target_robot_id)
        if relief_m > 0.0 and _debug_is_on():
            print(
                f"[strain]  결합 전 침투 해소: 로봇{robot_id} {relief_m * 1000:.3f} mm",
                flush=True,
            )
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
        self.model.site_sameframe[source_site_id] = int(
            mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        )
        # 접촉점(world_point)은 두 면 사이의 중간점이라, 가시 줄 위로 당겨 붙인
        # source 자리와 몇 mm 어긋난다.  그 어긋남이 그대로 connect 구속의 초기
        # 오차가 되어, 결합하는 순간 이를 없애려는 큰 구속력(=strain 100)이 뜬다.
        # 목표점을 '가시가 실제로 박히는 그 점'으로 맞춰 초기 오차를 0으로 만든다.
        mujoco.mj_kinematics(self.model, self.data)
        world_point = np.asarray(self.data.site_xpos[source_site_id]).copy()

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
        self._last_force_breakdown.clear()
        for robot_id, attachment in tuple(self._active.items()):
            constraint_force_n = self._constraint_force(attachment.equality_id)
            external_force_n = float(
                np.linalg.norm(self.data.xfrc_applied[attachment.target_body_id, :3])
            )
            source_body_id = self.model.body(
                body_name(attachment.source_robot_id, "rear")
            ).id
            external_force_n = max(
                external_force_n,
                float(np.linalg.norm(self.data.xfrc_applied[source_body_id, :3])),
            )
            force_n = max(constraint_force_n, external_force_n)
            error_m = self._constraint_error(attachment.equality_id)
            self._last_force_breakdown[robot_id] = (
                float(constraint_force_n),
                float(external_force_n),
                float(error_m),
            )
            peak_constraint, peak_external, peak_error = self._step_peak.get(
                robot_id, (0.0, 0.0, 0.0)
            )
            self._step_peak[robot_id] = (
                max(peak_constraint, float(constraint_force_n)),
                max(peak_external, float(external_force_n)),
                max(peak_error, float(error_m)),
            )
            timestep = float(self.model.opt.timestep)
            attachment.age_s += timestep
            # 1차 저역통과(시상수 _STRAIN_TAU_S).  솔버가 한 스텝 내놓는 스파이크를
            # 그대로 읽지 않고, 실제 게이지처럼 지속 하중만 올라오게 한다.
            alpha = timestep / max(self._STRAIN_TAU_S + timestep, timestep)
            attachment.filtered_force_n += alpha * (
                force_n - attachment.filtered_force_n
            )
            settling = attachment.age_s < self._ACTIVATION_SETTLE_S
            measured_n = 0.0 if settling else attachment.filtered_force_n
            attachment.force_n = measured_n
            strain_g = measured_n / GRAVITY_M_S2 * 1000.0
            self._last_strain_g[robot_id] = min(strain_g, self.robot.maximum_strain_g)
            self._last_label_positions[robot_id] = np.asarray(
                self.data.site_xpos[attachment.source_site_id]
            ).copy()
            if measured_n > self.robot.maximum_attachment_force_n:
                attachment.overload_duration_s += timestep
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

    def constrained_robots(self) -> frozenset[int]:
        """Robots held by an active attachment, as source or as target.

        Nothing outside the solver may move these: shifting one end of a
        ``connect`` equality by hand injects a large constraint force, which
        shows up as a saturated strain reading.
        """

        held: set[int] = set()
        for attachment in self._active.values():
            held.add(attachment.source_robot_id)
            if attachment.target_robot_id is not None:
                held.add(attachment.target_robot_id)
        return frozenset(held)

    def strain_breakdown(self) -> dict[int, tuple[float, float, float]]:
        """Per robot: peak (equality N, external N, constraint error m) this step.

        Peaks, not the final substep: an attachment that formed and broke inside
        one control step leaves nothing active at the end, yet is exactly the
        case worth seeing.
        """

        return dict(self._step_peak)

    def event_summary(self) -> str:
        """One-line description of this control step's attach/detach events."""

        if not self._events:
            return ""
        return " / ".join(
            f"{event.event}(로봇{event.source_robot_id}"
            + (
                "→바닥"
                if event.target_robot_id is None
                else f"→로봇{event.target_robot_id}"
            )
            + f", {event.strain_value:.0f} g)"
            for event in self._events
        )

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
    ) -> tuple[int, int | None, np.ndarray] | None:
        rear_geom_id = self.rear_geom_ids[robot_id]
        rear_body_id = self.model.body(body_name(robot_id, "rear")).id
        candidates: list[tuple[bool, float, int, int | None, np.ndarray]] = []
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
                )
            )
        if not candidates:
            return self._find_nearby_surface(robot_id)
        _, _, body_id, target_id, point = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        return body_id, target_id, point

    def _find_nearby_surface(
        self, robot_id: int
    ) -> tuple[int, int | None, np.ndarray] | None:
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
        return target_body_id, target_robot_id, point

    def _constraint_force(self, equality_id: int) -> float:
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
        return float(np.linalg.norm(np.asarray(self.data.efc_force[rows])))

    def _report_contacts(self, robot_id: int) -> None:
        """Print this robot's contacts with the numbers that decide the force."""

        lines: list[str] = []
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.geom_to_robot.get(int(contact.geom1))
            second = self.geom_to_robot.get(int(contact.geom2))
            if robot_id not in (first, second) or first == second:
                continue
            other = int(contact.geom2 if first == robot_id else contact.geom1)
            try:
                other_name = self.model.geom(other).name or f"geom{other}"
            except Exception:                        # pragma: no cover - 이름 없는 geom
                other_name = f"geom{other}"
            include = float(contact.includemargin)
            lines.append(
                f"{other_name}: dist {contact.dist * 1000:+.3f} mm, "
                f"힘 시작 {include * 1000:.3f} mm, "
                f"유효 침투 {(include - float(contact.dist)) * 1000:+.3f} mm"
            )
        print(
            f"[strain]  로봇{robot_id} 결합 직전 접촉 "
            + ("  |  ".join(lines) if lines else "없음"),
            flush=True,
        )

    def _relieve_penetration(self, robot_id: int, target_robot_id: int | None) -> float:
        """Lift the robot out of any surface it is currently pressed into.

        Returns the correction applied, in metres.  Only the deepest contact
        matters: everything shallower is inside the solver's normal working
        range and resolves on its own.
        """

        source_body_id = self.model.body(body_name(robot_id, "rear")).id
        root_joint_id = int(self.model.body_jntadr[source_body_id])
        address = int(self.model.jnt_qposadr[root_joint_id])
        deepest = 0.0
        correction = np.zeros(3, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.geom_to_robot.get(int(contact.geom1))
            second = self.geom_to_robot.get(int(contact.geom2))
            if robot_id not in (first, second):
                continue
            if first == second:                     # 같은 로봇의 조각끼리
                continue
            # 붙으려는 상대와의 접촉만 푼다.  다른 데서 눌린 것까지 여기서
            # 건드리면, 결합과 무관한 방향으로 로봇을 밀어내게 된다.
            other = second if robot_id == first else first
            if other != target_robot_id:
                continue
            # MuJoCo는 dist < includemargin(= margin - gap) 이면 밀기 시작한다.
            # gap 을 지정하지 않으면 includemargin = margin 이라, 1.5 mm 떨어져
            # 있어도 접촉력이 걸린다.  '파고든 깊이'는 -dist 가 아니라 이것이다.
            depth = float(contact.includemargin) - float(contact.dist)
            if depth <= deepest:
                continue
            deepest = depth
            depth = min(depth, self._RELIEF_LIMIT_M)
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            # frame[:3] 은 geom1 → geom2 방향이다.  로봇이 물러나야 하는 쪽으로.
            direction = -1.0 if robot_id == first else 1.0
            correction = direction * (depth + self._PENETRATION_MARGIN_M) * normal
        if deepest <= 0.0:
            return 0.0
        self.data.qpos[address : address + 3] += correction
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(correction))

    def _constraint_error(self, equality_id: int) -> float:
        """How far apart the two ends of the equality currently are (m).

        Zero means the latch is where it was made; a few mm here is what turns
        into newtons of constraint force.
        """

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
        return float(np.linalg.norm(np.asarray(self.data.efc_pos[rows])))

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
