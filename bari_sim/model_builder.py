"""Generate one MuJoCo model from centralized experiment configuration.

Runtime generation avoids duplicating guessed physical values across XML files.
Control and policy code never builds or mutates morphology.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .config import EnvironmentConfig, ExperimentConfig, RobotConfig

ROBOT_COLORS = (
    (0.20, 0.55, 0.95, 1.0),
    (0.95, 0.45, 0.20, 1.0),
    (0.30, 0.75, 0.38, 1.0),
    (0.72, 0.40, 0.90, 1.0),
    (0.95, 0.75, 0.18, 1.0),
    (0.18, 0.78, 0.78, 1.0),
)
FORCE_SENSOR_LOCATIONS = ("top_rear", "top_front", "left", "right")
ANCHOR_PAD_SIDES = ("left", "right")


def robot_root_body_name(robot_id: int) -> str:
    return f"robot_{robot_id}_link_0"


def link_body_name(robot_id: int, link_id: int) -> str:
    return f"robot_{robot_id}_link_{link_id}"


def root_joint_name(robot_id: int) -> str:
    return f"robot_{robot_id}_root_free"


def pitch_joint_name(robot_id: int, joint_id: int) -> str:
    return f"robot_{robot_id}_pitch_{joint_id}"


def actuator_name(robot_id: int, joint_id: int) -> str:
    return f"robot_{robot_id}_motor_{joint_id}"


def collision_geom_name(robot_id: int, link_id: int) -> str:
    return f"robot_{robot_id}_link_{link_id}_collision"


def anchor_pad_geom_name(robot_id: int, link_id: int, side: str) -> str:
    if side not in ANCHOR_PAD_SIDES:
        raise ValueError(f"unsupported anchor-pad side {side!r}")
    return f"robot_{robot_id}_link_{link_id}_anchor_pad_{side}"


def lateral_support_width(robot: RobotConfig) -> float:
    """Paired underside pads remain fully inside the robot body width."""

    return robot.width_m


def gripper_geom_name(robot_id: int) -> str:
    return f"robot_{robot_id}_gripper_collision"


def gripper_site_name(robot_id: int) -> str:
    return f"robot_{robot_id}_gripper_site"


def range_origin_site_name(robot_id: int) -> str:
    return f"robot_{robot_id}_range_origin"


def force_sensor_name(robot_id: int, link_id: int, location: str) -> str:
    return f"robot_{robot_id}_link_{link_id}_{location}_force"


def force_sensor_key(link_id: int, location: str) -> str:
    return f"link_{link_id}.{location}"


def force_sensor_site_name(robot_id: int, link_id: int, location: str) -> str:
    return f"{force_sensor_name(robot_id, link_id, location)}_site"


def attachment_target_site_name(source_robot_id: int, target_body_name: str) -> str:
    return f"attach_target_{source_robot_id}_on_{target_body_name}"


def attachment_equality_name(source_robot_id: int, target_body_name: str) -> str:
    return f"attach_{source_robot_id}_to_{target_body_name}"


def world_attachment_site_name(source_robot_id: int) -> str:
    return f"attach_target_{source_robot_id}_on_world"


def world_attachment_equality_name(source_robot_id: int) -> str:
    return f"attach_{source_robot_id}_to_world"


def _numbers(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def box_inertia(
    mass_kg: float,
    length_m: float,
    width_m: float,
    thickness_m: float,
) -> tuple[float, float, float]:
    """Principal inertia of a uniform box about its center of mass."""

    return (
        mass_kg * (width_m**2 + thickness_m**2) / 12.0,
        mass_kg * (length_m**2 + thickness_m**2) / 12.0,
        mass_kg * (length_m**2 + width_m**2) / 12.0,
    )


@dataclass(frozen=True)
class BuiltModel:
    xml: str
    robot_count: int
    link_count: int
    environment_geom_names: tuple[str, ...]
    spawn_positions_m: tuple[tuple[float, float, float], ...]


class ModelBuilder:
    """Builds articulated robots, environments, and inactive attachment slots."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build(self) -> BuiltModel:
        robot = self.config.robot
        simulation = self.config.simulation
        root = ET.Element("mujoco", {"model": "bari_articulated_swarm"})
        ET.SubElement(
            root,
            "compiler",
            {
                "angle": "radian",
                "autolimits": "true",
                "inertiafromgeom": "false",
                "balanceinertia": "true",
            },
        )
        ET.SubElement(
            root,
            "option",
            {
                "timestep": f"{simulation.timestep_s:.10g}",
                "gravity": f"0 0 {-simulation.gravity_m_s2:.10g}",
                "integrator": simulation.integrator,
                "iterations": str(simulation.solver_iterations),
                "cone": "elliptic",
            },
        )
        ET.SubElement(root, "size", {"njmax": "20000", "nconmax": "5000"})
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})
        ET.SubElement(visual, "map", {"force": "0.15", "stiffness": "80"})

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "key_light",
                "pos": "0 -1.5 2.5",
                "dir": "0 0.4 -1",
                "directional": "true",
            },
        )
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "overview",
                "pos": "1.2 -1.8 1.2",
                "xyaxes": "0.83 0.55 0 -0.26 0.39 0.88",
            },
        )
        environment_geom_names = self._add_environment(
            worldbody, self.config.environment
        )

        for robot_id in range(simulation.robot_count):
            ET.SubElement(
                worldbody,
                "site",
                {
                    "name": world_attachment_site_name(robot_id),
                    "pos": "0 0 0",
                    "size": "0.006",
                    "rgba": "1 0.15 0.15 0",
                    "group": "4",
                },
            )

        body_elements: dict[tuple[int, int], ET.Element] = {}
        spawn_positions = self._spawn_positions()
        for robot_id, position in enumerate(spawn_positions):
            self._add_robot(worldbody, robot_id, position, robot, body_elements)

        # A target site exists on every possible contacted link. Site positions
        # are moved to the contact's body-local point immediately before the
        # corresponding equality constraint is activated.
        for target_robot_id in range(simulation.robot_count):
            for link_id in range(robot.link_count):
                body_name = link_body_name(target_robot_id, link_id)
                body = body_elements[(target_robot_id, link_id)]
                for source_robot_id in range(simulation.robot_count):
                    if source_robot_id == target_robot_id:
                        continue
                    ET.SubElement(
                        body,
                        "site",
                        {
                            "name": attachment_target_site_name(
                                source_robot_id, body_name
                            ),
                            "pos": "0 0 0",
                            "size": "0.006",
                            "rgba": "1 0.15 0.15 0",
                            "group": "4",
                        },
                    )

        contact = ET.SubElement(root, "contact")
        pair_friction = robot.contact.robot_robot_friction
        if robot.directional_friction.enabled:
            pair_friction = (0.0, pair_friction[1], pair_friction[2])
        pair_friction5 = (
            pair_friction[0],
            pair_friction[0],
            pair_friction[1],
            pair_friction[2],
            pair_friction[2],
        )
        for first_robot_id in range(simulation.robot_count):
            first_geoms = (
                [
                    collision_geom_name(first_robot_id, link_id)
                    for link_id in range(robot.link_count)
                ]
                + [
                    anchor_pad_geom_name(first_robot_id, link_id, side)
                    for link_id in range(robot.link_count)
                    for side in ANCHOR_PAD_SIDES
                ]
                + [gripper_geom_name(first_robot_id)]
            )
            for second_robot_id in range(first_robot_id + 1, simulation.robot_count):
                second_geoms = (
                    [
                        collision_geom_name(second_robot_id, link_id)
                        for link_id in range(robot.link_count)
                    ]
                    + [
                        anchor_pad_geom_name(second_robot_id, link_id, side)
                        for link_id in range(robot.link_count)
                        for side in ANCHOR_PAD_SIDES
                    ]
                    + [gripper_geom_name(second_robot_id)]
                )
                for first_geom in first_geoms:
                    for second_geom in second_geoms:
                        ET.SubElement(
                            contact,
                            "pair",
                            {
                                "geom1": first_geom,
                                "geom2": second_geom,
                                "condim": str(robot.contact.condim),
                                "friction": _numbers(pair_friction5),
                                "solref": _numbers(robot.contact.solref),
                                "solimp": _numbers(robot.contact.solimp),
                            },
                        )

        actuator = ET.SubElement(root, "actuator")
        torque = robot.joint.max_torque_nm
        for robot_id in range(simulation.robot_count):
            for joint_id in range(robot.joint_count):
                ET.SubElement(
                    actuator,
                    "motor",
                    {
                        "name": actuator_name(robot_id, joint_id),
                        "joint": pitch_joint_name(robot_id, joint_id),
                        "gear": "1",
                        "ctrllimited": "true",
                        "ctrlrange": f"{-torque:.10g} {torque:.10g}",
                        "forcelimited": "true",
                        "forcerange": f"{-torque:.10g} {torque:.10g}",
                    },
                )

        equality = ET.SubElement(root, "equality")
        attachment = robot.attachment
        equality_attrs = {
            "active": "false",
            "solref": _numbers(attachment.solref),
            "solimp": _numbers(attachment.solimp),
        }
        for source_robot_id in range(simulation.robot_count):
            ET.SubElement(
                equality,
                "connect",
                {
                    "name": world_attachment_equality_name(source_robot_id),
                    "site1": gripper_site_name(source_robot_id),
                    "site2": world_attachment_site_name(source_robot_id),
                    **equality_attrs,
                },
            )
            for target_robot_id in range(simulation.robot_count):
                if source_robot_id == target_robot_id:
                    continue
                for link_id in range(robot.link_count):
                    body_name = link_body_name(target_robot_id, link_id)
                    ET.SubElement(
                        equality,
                        "connect",
                        {
                            "name": attachment_equality_name(
                                source_robot_id, body_name
                            ),
                            "site1": gripper_site_name(source_robot_id),
                            "site2": attachment_target_site_name(
                                source_robot_id, body_name
                            ),
                            **equality_attrs,
                        },
                    )

        sensor = ET.SubElement(root, "sensor")
        for robot_id in range(simulation.robot_count):
            for link_id in range(robot.link_count):
                for location in FORCE_SENSOR_LOCATIONS:
                    ET.SubElement(
                        sensor,
                        "touch",
                        {
                            "name": force_sensor_name(robot_id, link_id, location),
                            "site": force_sensor_site_name(robot_id, link_id, location),
                        },
                    )

        ET.indent(root, space="  ")
        xml = ET.tostring(root, encoding="unicode")
        return BuiltModel(
            xml=xml,
            robot_count=simulation.robot_count,
            link_count=robot.link_count,
            environment_geom_names=tuple(environment_geom_names),
            spawn_positions_m=tuple(spawn_positions),
        )

    def _spawn_positions(self) -> list[tuple[float, float, float]]:
        robot = self.config.robot
        simulation = self.config.simulation
        environment = self.config.environment
        surface_z = environment.platform_height_m if environment.name == "gap" else 0.0
        spawn_x = simulation.spawn_x_m
        if environment.name == "gap":
            safe_rear_x = (
                -environment.gap_width_m / 2.0 - sum(robot.link_lengths_m) - 0.05
            )
            spawn_x = min(spawn_x, safe_rear_x)
        z = surface_z + robot.thickness_m / 2.0 + simulation.spawn_clearance_m
        columns = min(simulation.formation_columns, simulation.robot_count)
        column_spacing = (
            lateral_support_width(robot) + simulation.formation_column_gap_m
        )
        row_spacing = (
            sum(robot.link_lengths_m)
            + robot.gripper.length_m
            + simulation.formation_row_gap_m
        )
        positions: list[tuple[float, float, float]] = []
        for robot_id in range(simulation.robot_count):
            row = robot_id // columns
            column = robot_id % columns
            row_start = row * columns
            row_count = min(columns, simulation.robot_count - row_start)
            column_center = (row_count - 1) / 2.0
            positions.append(
                (
                    spawn_x - row * row_spacing,
                    (column - column_center) * column_spacing,
                    z,
                )
            )
        return positions

    def _add_environment(
        self, worldbody: ET.Element, config: EnvironmentConfig
    ) -> list[str]:
        surface_friction = config.surface_friction
        if self.config.robot.directional_friction.enabled:
            surface_friction = (0.0, surface_friction[1], surface_friction[2])
        friction = _numbers(surface_friction)
        common = {
            "friction": friction,
            "condim": "4",
            "rgba": "0.42 0.46 0.50 1",
        }
        names: list[str] = []
        if config.name == "flat":
            name = "environment_flat_ground"
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": name,
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": _numbers(
                        (
                            config.platform_length_m / 2.0,
                            config.platform_width_m / 2.0,
                            0.05,
                        )
                    ),
                    **common,
                },
            )
            names.append(name)
        elif config.name == "gap":
            half_length = config.platform_length_m / 2.0
            half_width = config.platform_width_m / 2.0
            half_thickness = config.platform_thickness_m / 2.0
            left_center = -config.gap_width_m / 2.0 - half_length
            right_center = config.gap_width_m / 2.0 + half_length
            for side, x in (("left", left_center), ("right", right_center)):
                name = f"environment_gap_{side}_platform"
                ET.SubElement(
                    worldbody,
                    "geom",
                    {
                        "name": name,
                        "type": "box",
                        "pos": _numbers(
                            (x, 0.0, config.platform_height_m - half_thickness)
                        ),
                        "size": _numbers((half_length, half_width, half_thickness)),
                        **common,
                    },
                )
                names.append(name)
            # Thin visual markers expose gap edges without adding collision.
            for side, x in (
                ("left", -config.gap_width_m / 2.0),
                ("right", config.gap_width_m / 2.0),
            ):
                ET.SubElement(
                    worldbody,
                    "geom",
                    {
                        "name": f"gap_edge_marker_{side}",
                        "type": "box",
                        "pos": _numbers((x, 0.0, config.platform_height_m + 0.002)),
                        "size": _numbers((0.003, half_width, 0.002)),
                        "rgba": "0.95 0.2 0.15 1",
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "2",
                    },
                )
        elif config.name == "step":
            ground_name = "environment_step_ground"
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": ground_name,
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": _numbers(
                        (
                            config.platform_length_m / 2.0,
                            config.platform_width_m / 2.0,
                            0.05,
                        )
                    ),
                    **common,
                },
            )
            names.append(ground_name)
            step_name = "environment_step_obstacle"
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": step_name,
                    "type": "box",
                    "pos": _numbers(
                        (
                            config.step_x_m + config.step_width_m / 2.0,
                            0.0,
                            config.step_height_m / 2.0,
                        )
                    ),
                    "size": _numbers(
                        (
                            config.step_width_m / 2.0,
                            config.step_depth_m / 2.0,
                            config.step_height_m / 2.0,
                        )
                    ),
                    **common,
                },
            )
            names.append(step_name)
        else:
            raise ValueError(f"unsupported environment {config.name!r}")
        return names

    def _add_robot(
        self,
        worldbody: ET.Element,
        robot_id: int,
        spawn_position: tuple[float, float, float],
        config: RobotConfig,
        body_elements: dict[tuple[int, int], ET.Element],
    ) -> None:
        color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        parent = worldbody
        for link_id in range(config.link_count):
            length = config.link_lengths_m[link_id]
            body_attrs = {"name": link_body_name(robot_id, link_id)}
            if link_id == 0:
                body_attrs["pos"] = _numbers(spawn_position)
            else:
                body_attrs["pos"] = _numbers(
                    (config.link_lengths_m[link_id - 1], 0.0, 0.0)
                )
            body = ET.SubElement(parent, "body", body_attrs)
            body_elements[(robot_id, link_id)] = body
            if link_id == 0:
                ET.SubElement(body, "freejoint", {"name": root_joint_name(robot_id)})
            else:
                low, high = config.joint.limits_rad[link_id - 1]
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": pitch_joint_name(robot_id, link_id - 1),
                        "type": "hinge",
                        "axis": "0 1 0",
                        "range": _numbers((low, high)),
                        "damping": f"{config.joint.damping_nms_rad:.10g}",
                        "stiffness": f"{config.joint.stiffness_nm_rad:.10g}",
                        "armature": f"{config.joint.armature_kg_m2:.10g}",
                    },
                )
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": f"robot_{robot_id}_joint_{link_id - 1}_marker",
                        "type": "sphere",
                        "pos": "0 0 0",
                        "size": f"{config.thickness_m * 0.58:.10g}",
                        "rgba": "0.12 0.12 0.12 1",
                        "group": "2",
                    },
                )

            offset = config.center_of_mass_offsets_m[link_id]
            com = (length / 2.0 + offset[0], offset[1], offset[2])
            inertia = (
                config.inertia_diagonal_kg_m2[link_id]
                if config.inertia_diagonal_kg_m2 is not None
                else box_inertia(
                    config.link_masses_kg[link_id],
                    length,
                    config.width_m,
                    config.thickness_m,
                )
            )
            ET.SubElement(
                body,
                "inertial",
                {
                    "pos": _numbers(com),
                    "mass": f"{config.link_masses_kg[link_id]:.10g}",
                    "diaginertia": _numbers(inertia),
                },
            )
            self._add_link_geometries(body, robot_id, link_id, length, config, color)
            self._add_force_sensor_sites(body, robot_id, link_id, length, config)

            if link_id == min(1, config.link_count - 1):
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": f"robot_{robot_id}_id_marker",
                        "type": "sphere",
                        "pos": _numbers((length / 2.0, 0.0, config.thickness_m * 0.72)),
                        "size": f"{config.thickness_m * 0.28:.10g}",
                        "rgba": _numbers(color),
                        "group": "2",
                    },
                )

            if link_id == config.link_count - 1:
                self._add_gripper(body, robot_id, length, config, color)
            parent = body

    def _add_link_geometries(
        self,
        body: ET.Element,
        robot_id: int,
        link_id: int,
        length: float,
        config: RobotConfig,
        color: tuple[float, float, float, float],
    ) -> None:
        center = (length / 2.0, 0.0, 0.0)
        visual_size = (length * 0.49, config.width_m * 0.50, config.thickness_m * 0.48)
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"robot_{robot_id}_link_{link_id}_visual",
                "type": "box",
                "pos": _numbers(center),
                "size": _numbers(visual_size),
                "rgba": _numbers(color),
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        contact_friction = config.contact.friction
        if config.directional_friction.enabled:
            contact_friction = (0.0, contact_friction[1], contact_friction[2])
        collision_attrs = {
            "name": collision_geom_name(robot_id, link_id),
            "friction": _numbers(contact_friction),
            "condim": str(config.contact.condim),
            "solref": _numbers(config.contact.solref),
            "solimp": _numbers(config.contact.solimp),
            "rgba": "0.1 0.1 0.1 0.001",
            "group": "3",
        }
        if config.contact.shape == "box":
            collision_attrs.update(
                {
                    "type": "box",
                    "pos": _numbers(center),
                    "size": _numbers(
                        (length / 2.0, config.width_m / 2.0, config.thickness_m / 2.0)
                    ),
                }
            )
        else:
            radius = min(config.width_m, config.thickness_m) / 2.0
            collision_attrs.update(
                {
                    "type": "capsule",
                    "fromto": _numbers((radius, 0.0, 0.0, length - radius, 0.0, 0.0)),
                    "size": f"{radius:.10g}",
                }
            )
        ET.SubElement(body, "geom", collision_attrs)
        if config.directional_friction.enabled:
            # Two thin, visible underside pads stay inside the body envelope so
            # neighboring robots cannot collide with protruding pad geometry.
            # DirectionalFrictionModel applies asymmetric tangential reaction
            # only at contacts involving these geoms.
            pad_height = min(0.001, config.thickness_m * 0.16)
            pad_half_width = config.width_m * 0.18
            pad_center_y = config.width_m / 2.0 - pad_half_width
            for side, sign in (("left", 1.0), ("right", -1.0)):
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": anchor_pad_geom_name(robot_id, link_id, side),
                        "type": "box",
                        "pos": _numbers(
                            (
                                length / 2.0,
                                sign * pad_center_y,
                                -config.thickness_m / 2.0 - pad_height,
                            )
                        ),
                        "size": _numbers((length * 0.35, pad_half_width, pad_height)),
                        "friction": "0 0 0",
                        "condim": str(config.contact.condim),
                        "solref": _numbers(config.contact.solref),
                        "solimp": _numbers(config.contact.solimp),
                        "rgba": "0.08 0.75 0.22 1",
                        "group": "1",
                    },
                )

    def _add_force_sensor_sites(
        self,
        body: ET.Element,
        robot_id: int,
        link_id: int,
        length: float,
        config: RobotConfig,
    ) -> None:
        patch_depth = 0.0025
        sites = {
            "top_rear": (
                (length * 0.25, 0.0, config.thickness_m / 2.0),
                (length * 0.255, config.width_m * 0.52, patch_depth),
                (0.15, 0.95, 1.0, 0.85),
            ),
            "top_front": (
                (length * 0.75, 0.0, config.thickness_m / 2.0),
                (length * 0.255, config.width_m * 0.52, patch_depth),
                (0.15, 0.95, 1.0, 0.85),
            ),
            "left": (
                (length * 0.5, config.width_m / 2.0, 0.0),
                (length * 0.505, patch_depth, config.thickness_m * 0.52),
                (1.0, 0.75, 0.15, 0.85),
            ),
            "right": (
                (length * 0.5, -config.width_m / 2.0, 0.0),
                (length * 0.505, patch_depth, config.thickness_m * 0.52),
                (1.0, 0.35, 0.70, 0.85),
            ),
        }
        for location, (position, size, color) in sites.items():
            ET.SubElement(
                body,
                "site",
                {
                    "name": force_sensor_site_name(robot_id, link_id, location),
                    "type": "box",
                    "pos": _numbers(position),
                    "size": _numbers(size),
                    "rgba": _numbers(color),
                    "group": "2",
                },
            )

    def _add_gripper(
        self,
        body: ET.Element,
        robot_id: int,
        front_link_length: float,
        config: RobotConfig,
        color: tuple[float, float, float, float],
    ) -> None:
        grip = config.gripper
        contact_friction = config.contact.friction
        if config.directional_friction.enabled:
            contact_friction = (0.0, contact_friction[1], contact_friction[2])
        center_x = front_link_length + grip.length_m / 2.0
        half_size = (grip.length_m / 2.0, grip.width_m / 2.0, grip.thickness_m / 2.0)
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"robot_{robot_id}_gripper_visual",
                "type": "box",
                "pos": _numbers((center_x, 0.0, 0.0)),
                "size": _numbers(tuple(value * 0.96 for value in half_size)),
                "rgba": _numbers((0.12, 0.12, 0.12, color[3])),
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": gripper_geom_name(robot_id),
                "type": "box",
                "pos": _numbers((center_x, 0.0, 0.0)),
                "size": _numbers(half_size),
                "friction": _numbers(contact_friction),
                "condim": str(config.contact.condim),
                "solref": _numbers(config.contact.solref),
                "solimp": _numbers(config.contact.solimp),
                "rgba": "0.1 0.1 0.1 0.001",
                "group": "3",
            },
        )
        tip_x = front_link_length + grip.length_m
        ET.SubElement(
            body,
            "site",
            {
                "name": gripper_site_name(robot_id),
                "type": "sphere",
                "pos": _numbers((tip_x, 0.0, 0.0)),
                "size": "0.008",
                "rgba": "1 0.2 0.15 1",
                "group": "2",
            },
        )
        ET.SubElement(
            body,
            "site",
            {
                "name": range_origin_site_name(robot_id),
                "type": "sphere",
                "pos": _numbers((tip_x + 0.004, 0.0, 0.0)),
                "size": "0.004",
                "rgba": "0.15 1 0.25 1",
                "group": "2",
            },
        )
