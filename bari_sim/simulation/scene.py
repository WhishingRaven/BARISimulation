"""MJCF scene construction for robots and the three supported environments."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ..robot.specification import DEFAULT_ROBOT, RobotSpecification
from ..tasks.catalog import RobotGrid, TaskDefinition, TaskName

LINK_NAMES = ("rear", "middle", "front")
ROBOT_COLORS = (
    (0.12, 0.55, 0.95, 1.0),
    (0.96, 0.43, 0.18, 1.0),
    (0.18, 0.72, 0.42, 1.0),
    (0.72, 0.36, 0.88, 1.0),
    (0.95, 0.74, 0.14, 1.0),
)
SEGMENT_BRIGHTNESS = {"rear": 0.78, "middle": 1.0, "front": 1.18}


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def body_name(robot_id: int, link: str) -> str:
    return f"robot_{robot_id}_{link}_body"


def geom_name(robot_id: int, link: str) -> str:
    return f"robot_{robot_id}_{link}_geom"


def hinge_name(robot_id: int, hinge: str) -> str:
    return f"robot_{robot_id}_{hinge}_hinge"


def actuator_name(robot_id: int, hinge: str) -> str:
    return f"robot_{robot_id}_{hinge}_motor"


def spike_site_name(robot_id: int) -> str:
    return f"robot_{robot_id}_spike_site"


def gait_anchor_site_name(robot_id: int, anchor: str) -> str:
    return f"robot_{robot_id}_gait_{anchor}_anchor"


def gait_target_site_name(robot_id: int, anchor: str) -> str:
    return f"robot_{robot_id}_gait_{anchor}_target"


def gait_equality_name(robot_id: int, anchor: str) -> str:
    return f"robot_{robot_id}_gait_{anchor}_latch"


def gait_robot_equality_name(
    source_id: int, anchor: str, target_id: int, link: str
) -> str:
    return f"robot_{source_id}_gait_{anchor}_on_{target_id}_{link}"


def sensor_site_name(robot_id: int, sensor: str) -> str:
    return f"robot_{robot_id}_{sensor}_sensor"


def world_target_site_name(robot_id: int) -> str:
    return f"robot_{robot_id}_world_attachment_target"


def robot_target_site_name(source_id: int, target_id: int, link: str) -> str:
    return f"robot_{source_id}_target_on_{target_id}_{link}"


def world_equality_name(robot_id: int) -> str:
    return f"robot_{robot_id}_attach_world"


def robot_equality_name(source_id: int, target_id: int, link: str) -> str:
    return f"robot_{source_id}_attach_{target_id}_{link}"


@dataclass(frozen=True)
class SceneRequest:
    grid: RobotGrid
    environment: str
    task: TaskDefinition | None = None
    physics_timestep_s: float = 0.002

    def validate(self) -> None:
        if self.environment not in {"flat", "gap", "step"}:
            raise ValueError("environment must be flat, gap, or step")
        if self.physics_timestep_s <= 0.0:
            raise ValueError("physics timestep must be positive")
        if self.task is not None and self.environment != self.task.environment:
            raise ValueError("task and environment do not match")


@dataclass(frozen=True)
class BuiltScene:
    xml: str
    request: SceneRequest
    spawn_positions_m: tuple[tuple[float, float, float], ...]
    initial_centroid_x_m: float
    target_x_m: float | None
    gap_width_m: float | None
    step_height_m: float | None
    platform_width_m: float


class SceneBuilder:
    def __init__(
        self,
        request: SceneRequest,
        robot: RobotSpecification = DEFAULT_ROBOT,
    ):
        request.validate()
        robot.validate()
        self.request = request
        self.robot = robot
        self.robot_count = request.grid.count

    def build(self) -> BuiltScene:
        platform_width = max(1.2, self.request.grid.columns * 0.125 + 0.50)
        gap_width = self._gap_width()
        step_height = self._step_height()
        spawns = self._spawn_positions(gap_width, step_height)
        centroid_x = sum(position[0] for position in spawns) / len(spawns)
        target_x = (
            centroid_x + self.request.task.value
            if self.request.task is not None
            and self.request.task.name is TaskName.COLLISION_AVOIDANCE
            else None
        )

        root = ET.Element("mujoco", {"model": "bari_discrete_swarm"})
        ET.SubElement(
            root,
            "compiler",
            {"angle": "radian", "autolimits": "true", "balanceinertia": "true"},
        )
        ET.SubElement(
            root,
            "option",
            {
                "timestep": f"{self.request.physics_timestep_s:.10g}",
                "gravity": "0 0 -9.81",
                "integrator": "implicitfast",
                "iterations": "80",
                "cone": "elliptic",
            },
        )
        ET.SubElement(
            root,
            "size",
            {
                "njmax": str(max(20_000, self.robot_count * self.robot_count * 30)),
                "nconmax": str(max(5_000, self.robot_count * self.robot_count * 20)),
            },
        )
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-30"})
        ET.SubElement(visual, "map", {"force": "0.1", "stiffness": "80"})

        world = ET.SubElement(root, "worldbody")
        ET.SubElement(
            world,
            "light",
            {"pos": "0 -2 3", "dir": "0 0.3 -1", "directional": "true"},
        )
        self._add_environment(
            world,
            gap_width=gap_width,
            step_height=step_height,
            platform_width=platform_width,
            target_x=target_x,
        )
        for robot_id in range(self.robot_count):
            ET.SubElement(
                world,
                "site",
                {
                    "name": world_target_site_name(robot_id),
                    "pos": "0 0 0",
                    "size": "0.001",
                    "rgba": "0 0 0 0",
                },
            )
            for anchor in ("rear", "front"):
                ET.SubElement(
                    world,
                    "site",
                    {
                        "name": gait_target_site_name(robot_id, anchor),
                        "pos": "0 0 0",
                        "size": "0.001",
                        "rgba": "0 0 0 0",
                    },
                )

        bodies: dict[tuple[int, str], ET.Element] = {}
        for robot_id, position in enumerate(spawns):
            self._add_robot(world, robot_id, position, bodies)

        actuators = ET.SubElement(root, "actuator")
        for robot_id in range(self.robot_count):
            for hinge in ("rear", "front"):
                ET.SubElement(
                    actuators,
                    "motor",
                    {
                        "name": actuator_name(robot_id, hinge),
                        "joint": hinge_name(robot_id, hinge),
                        "gear": "1",
                        "ctrllimited": "true",
                        "ctrlrange": _numbers(
                            (-self.robot.joint_torque_nm, self.robot.joint_torque_nm)
                        ),
                    },
                )

        equality = ET.SubElement(root, "equality")
        equality_common = {
            "active": "false",
            "solref": "0.01 1",
            "solimp": "0.9 0.95 0.001",
        }
        for source_id in range(self.robot_count):
            for anchor in ("rear", "front"):
                ET.SubElement(
                    equality,
                    "connect",
                    {
                        "name": gait_equality_name(source_id, anchor),
                        "site1": gait_anchor_site_name(source_id, anchor),
                        "site2": gait_target_site_name(source_id, anchor),
                        **equality_common,
                        "solref": "0.001 1",
                        "solimp": "0.99 0.999 0.00001",
                    },
                )
                for target_id in range(self.robot_count):
                    if target_id == source_id:
                        continue
                    for link in LINK_NAMES:
                        ET.SubElement(
                            equality,
                            "connect",
                            {
                                "name": gait_robot_equality_name(
                                    source_id, anchor, target_id, link
                                ),
                                "site1": gait_anchor_site_name(source_id, anchor),
                                "site2": robot_target_site_name(
                                    source_id, target_id, link
                                ),
                                **equality_common,
                                "solref": "0.001 1",
                                "solimp": "0.99 0.999 0.00001",
                            },
                        )
            ET.SubElement(
                equality,
                "connect",
                {
                    "name": world_equality_name(source_id),
                    "site1": spike_site_name(source_id),
                    "site2": world_target_site_name(source_id),
                    **equality_common,
                },
            )
            for target_id in range(self.robot_count):
                if target_id == source_id:
                    continue
                for link in LINK_NAMES:
                    ET.SubElement(
                        equality,
                        "connect",
                        {
                            "name": robot_equality_name(source_id, target_id, link),
                            "site1": spike_site_name(source_id),
                            "site2": robot_target_site_name(source_id, target_id, link),
                            **equality_common,
                        },
                    )

        ET.indent(root, space="  ")
        return BuiltScene(
            xml=ET.tostring(root, encoding="unicode"),
            request=self.request,
            spawn_positions_m=tuple(spawns),
            initial_centroid_x_m=centroid_x,
            target_x_m=target_x,
            gap_width_m=gap_width if self.request.environment == "gap" else None,
            step_height_m=step_height if self.request.environment == "step" else None,
            platform_width_m=platform_width,
        )

    def _gap_width(self) -> float:
        if self.request.environment != "gap":
            return 0.10
        if self.request.task is not None:
            return self.request.task.value
        return 0.10

    def _step_height(self) -> float:
        if self.request.environment != "step":
            return 0.03
        if self.request.task is not None:
            return self.request.task.value
        return 0.03

    def _spawn_positions(
        self, gap_width: float, step_height: float
    ) -> list[tuple[float, float, float]]:
        del step_height
        if self.request.environment == "gap":
            lead_x = -gap_width / 2.0 - self.robot.length_m - 0.025
            surface_z = 0.0
        elif self.request.environment == "step":
            lead_x = -self.robot.length_m - 0.025
            surface_z = 0.0
        elif self.request.task is not None:
            lead_x = -0.50
            surface_z = 0.0
        else:
            lead_x = 0.0
            surface_z = 0.0
        row_spacing = self.robot.length_m + 0.012
        column_spacing = self.robot.width_m + 0.012
        z = surface_z + self.robot.height_m / 2.0 + 0.002
        positions: list[tuple[float, float, float]] = []
        for row in range(self.request.grid.rows):
            for column in range(self.request.grid.columns):
                centered_column = column - (self.request.grid.columns - 1) / 2.0
                positions.append(
                    (lead_x - row * row_spacing, centered_column * column_spacing, z)
                )
        return positions

    def _add_environment(
        self,
        world: ET.Element,
        *,
        gap_width: float,
        step_height: float,
        platform_width: float,
        target_x: float | None,
    ) -> None:
        common = {
            "friction": "1.1 0.005 0.0001",
            "condim": "4",
            "rgba": "0.40 0.44 0.48 1",
            "group": "0",
        }
        if self.request.environment == "gap":
            half_length = 6.0
            thickness = 0.05
            for side, center_x in (
                ("near", -gap_width / 2.0 - half_length),
                ("far", gap_width / 2.0 + half_length),
            ):
                ET.SubElement(
                    world,
                    "geom",
                    {
                        "name": f"environment_gap_{side}",
                        "type": "box",
                        "pos": _numbers((center_x, 0.0, -thickness / 2.0)),
                        "size": _numbers(
                            (half_length, platform_width / 2.0, thickness / 2.0)
                        ),
                        **common,
                    },
                )
        else:
            ET.SubElement(
                world,
                "geom",
                {
                    "name": "environment_ground",
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": "12 3 0.05",
                    **common,
                },
            )
            if self.request.environment == "flat":
                self._add_floor_grid(world, platform_width)
        if self.request.environment == "step":
            length = 6.0
            ET.SubElement(
                world,
                "geom",
                {
                    "name": "environment_step",
                    "type": "box",
                    "pos": _numbers((length / 2.0, 0.0, step_height / 2.0)),
                    "size": _numbers(
                        (length / 2.0, platform_width / 2.0, step_height / 2.0)
                    ),
                    **common,
                },
            )
        if target_x is not None:
            ET.SubElement(
                world,
                "site",
                {
                    "name": "collision_avoidance_target",
                    "type": "cylinder",
                    "pos": _numbers((target_x, 0.0, 0.002)),
                    "size": "0.25 0.002",
                    "rgba": "0.15 0.85 0.25 0.65",
                },
            )

    @staticmethod
    def _add_floor_grid(world: ET.Element, platform_width: float) -> None:
        """Add non-colliding reference lines so small translations are visible."""

        grid_half_x = 6.0
        grid_half_y = max(platform_width / 2.0, 0.8)
        spacing = 0.10
        line_height = 0.00015
        line_width = 0.00035
        grid_common = {
            "type": "box",
            "pos": "0 0 0.0002",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        }
        x_count = round(grid_half_x / spacing)
        y_count = round(grid_half_y / spacing)
        for index in range(-x_count, x_count + 1):
            x = index * spacing
            major = index % 5 == 0
            ET.SubElement(
                world,
                "geom",
                {
                    "name": f"floor_grid_x_{index:+d}",
                    "pos": _numbers((x, 0.0, 0.0002)),
                    "size": _numbers(
                        (
                            line_width if major else line_width / 2.0,
                            grid_half_y,
                            line_height,
                        )
                    ),
                    "rgba": "0.12 0.18 0.24 0.62" if major else "0.18 0.24 0.30 0.32",
                    **{
                        key: value for key, value in grid_common.items() if key != "pos"
                    },
                },
            )
        for index in range(-y_count, y_count + 1):
            y = index * spacing
            major = index % 5 == 0
            ET.SubElement(
                world,
                "geom",
                {
                    "name": f"floor_grid_y_{index:+d}",
                    "pos": _numbers((0.0, y, 0.0002)),
                    "size": _numbers(
                        (
                            grid_half_x,
                            line_width if major else line_width / 2.0,
                            line_height,
                        )
                    ),
                    "rgba": "0.12 0.18 0.24 0.62" if major else "0.18 0.24 0.30 0.32",
                    **{
                        key: value for key, value in grid_common.items() if key != "pos"
                    },
                },
            )

    def _add_robot(
        self,
        world: ET.Element,
        robot_id: int,
        position: tuple[float, float, float],
        bodies: dict[tuple[int, str], ET.Element],
    ) -> None:
        rear_length, middle_length, front_length = self.robot.segment_lengths_m
        masses = self.robot.segment_masses_kg
        color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        rear = ET.SubElement(
            world,
            "body",
            {"name": body_name(robot_id, "rear"), "pos": _numbers(position)},
        )
        bodies[(robot_id, "rear")] = rear
        ET.SubElement(rear, "freejoint", {"name": f"robot_{robot_id}_root"})
        self._add_link(
            rear,
            robot_id,
            "rear",
            rear_length,
            masses[0],
            0.0,
            self._segment_color(color, "rear"),
        )
        ET.SubElement(
            rear,
            "site",
            {
                "name": spike_site_name(robot_id),
                "pos": _numbers((-rear_length / 2.0, 0.0, -self.robot.height_m / 2.0)),
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )
        self._add_gait_anchor(rear, robot_id, "rear", -rear_length / 2.0)

        middle = ET.SubElement(
            rear,
            "body",
            {
                "name": body_name(robot_id, "middle"),
                "pos": _numbers((rear_length / 2.0, 0.0, 0.0)),
            },
        )
        bodies[(robot_id, "middle")] = middle
        self._add_hinge(middle, robot_id, "rear")
        self._add_link(
            middle,
            robot_id,
            "middle",
            middle_length,
            masses[1],
            middle_length / 2.0,
            self._segment_color(color, "middle"),
        )
        station_x = middle_length + front_length - self.robot.front_sensor_offset_m
        sensor_positions = {
            "down": (station_x, 0.0, -self.robot.height_m / 2.0),
            "left": (station_x, self.robot.width_m / 2.0, 0.0),
            "right": (station_x, -self.robot.width_m / 2.0, 0.0),
        }
        for sensor, sensor_position in sensor_positions.items():
            ET.SubElement(
                middle,
                "site",
                {
                    "name": sensor_site_name(robot_id, sensor),
                    "pos": _numbers(sensor_position),
                    "size": "0.001",
                    "rgba": "0 0 0 0",
                },
            )

        front = ET.SubElement(
            middle,
            "body",
            {
                "name": body_name(robot_id, "front"),
                "pos": _numbers((middle_length, 0.0, 0.0)),
            },
        )
        bodies[(robot_id, "front")] = front
        self._add_hinge(front, robot_id, "front")
        self._add_link(
            front,
            robot_id,
            "front",
            front_length,
            masses[2],
            front_length / 2.0,
            self._segment_color(color, "front"),
        )
        ET.SubElement(
            front,
            "site",
            {
                "name": sensor_site_name(robot_id, "front"),
                "pos": _numbers((front_length, 0.0, 0.0)),
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )
        self._add_gait_anchor(front, robot_id, "front", front_length)

        for target_id in range(self.robot_count):
            if target_id == robot_id:
                continue
            for link in LINK_NAMES:
                body = bodies[(robot_id, link)]
                link_center_x = (
                    0.0
                    if link == "rear"
                    else (
                        middle_length / 2.0 if link == "middle" else front_length / 2.0
                    )
                )
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": robot_target_site_name(target_id, robot_id, link),
                        # Attachment targets are on the top surface, so a
                        # rear spike can latch onto a robot from below.
                        "pos": _numbers(
                            (link_center_x, 0.0, self.robot.height_m / 2.0)
                        ),
                        "size": "0.001",
                        "rgba": "0 0 0 0",
                    },
                )

    def _add_gait_anchor(
        self, body: ET.Element, robot_id: int, anchor: str, x: float
    ) -> None:
        """Add a small, physical cleat at each end of the inchworm body."""

        height = self.robot.height_m
        width = self.robot.width_m
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"robot_{robot_id}_gait_{anchor}_cleat",
                "type": "box",
                "pos": _numbers((x, 0.0, -height / 2.0 - 0.0005)),
                "size": _numbers((0.0015, width / 2.0, 0.0005)),
                "mass": "0",
                "friction": "3.0 0.005 0.0001",
                "condim": "4",
                "contype": "0",
                "conaffinity": "0",
                "rgba": "0.16 0.16 0.18 1",
                "group": "1",
            },
        )
        ET.SubElement(
            body,
            "site",
            {
                "name": gait_anchor_site_name(robot_id, anchor),
                "pos": _numbers((x, 0.0, -height / 2.0 - 0.001)),
                "size": "0.001",
                "rgba": "0 0 0 0",
            },
        )

    def _add_hinge(self, body: ET.Element, robot_id: int, hinge: str) -> None:
        limit = self.robot.hinge_limit_rad
        ET.SubElement(
            body,
            "joint",
            {
                "name": hinge_name(robot_id, hinge),
                "type": "hinge",
                "axis": "0 1 0",
                "range": _numbers((-limit, limit)),
                "damping": "0.004",
                "armature": "0.000001",
            },
        )

    @staticmethod
    def _segment_color(
        color: tuple[float, float, float, float], link: str
    ) -> tuple[float, float, float, float]:
        brightness = SEGMENT_BRIGHTNESS[link]
        return (
            min(color[0] * brightness, 1.0),
            min(color[1] * brightness, 1.0),
            min(color[2] * brightness, 1.0),
            color[3],
        )

    def _add_link(
        self,
        body: ET.Element,
        robot_id: int,
        link: str,
        length: float,
        mass: float,
        center_x: float,
        color: tuple[float, float, float, float],
    ) -> None:
        width = self.robot.width_m
        height = self.robot.height_m
        inertia = (
            mass * (width**2 + height**2) / 12.0,
            mass * (length**2 + height**2) / 12.0,
            mass * (length**2 + width**2) / 12.0,
        )
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": _numbers((center_x, 0.0, 0.0)),
                "mass": f"{mass:.10g}",
                "diaginertia": _numbers(inertia),
            },
        )
        geom_attributes = {
            "name": geom_name(robot_id, link),
            "type": "box",
            "pos": _numbers((center_x, 0.0, 0.0)),
            "mass": "0",
            "friction": "2.0 0.005 0.0001",
            "condim": "4",
            "solref": "0.015 1",
            "solimp": "0.9 0.95 0.001",
            "rgba": _numbers(color),
            # MuJoCo hides geom groups 3-5 by default.  Keep the robots in
            # their own, visible group so every viewer shows them without
            # requiring a manual visibility toggle.
            "group": "1",
        }
        geom_attributes["size"] = _numbers((length / 2.0, width / 2.0, height / 2.0))
        ET.SubElement(body, "geom", geom_attributes)
