"""Central TOML configuration with validation and documented SI units."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import pi
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "config" / "robot.toml"
DEFAULT_ENVIRONMENT_CONFIG = PROJECT_ROOT / "config" / "environment.toml"
DEFAULT_SIMULATION_CONFIG = PROJECT_ROOT / "config" / "simulation.toml"


def _tuple_floats(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return tuple(float(item) for item in value)


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    parsed = _tuple_floats(value, name)
    if len(parsed) != 3:
        raise ValueError(f"{name} must contain 3 values")
    return parsed[0], parsed[1], parsed[2]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


@dataclass(frozen=True)
class JointConfig:
    limits_rad: tuple[tuple[float, float], ...]
    max_torque_nm: float
    max_speed_rad_s: float
    damping_nms_rad: float
    stiffness_nm_rad: float
    armature_kg_m2: float
    position_kp_nm_rad: float
    velocity_kd_nms_rad: float


@dataclass(frozen=True)
class ContactConfig:
    shape: str
    friction: tuple[float, float, float]
    robot_robot_friction: tuple[float, float, float]
    condim: int
    solref: tuple[float, float]
    solimp: tuple[float, float, float]


@dataclass(frozen=True)
class GripperConfig:
    length_m: float
    width_m: float
    thickness_m: float
    location: str


@dataclass(frozen=True)
class AttachmentConfig:
    break_force_n: float
    break_moment_nm: float
    failure_model: str
    allow_environment: bool
    solref: tuple[float, float]
    solimp: tuple[float, float, float]


@dataclass(frozen=True)
class RangeSensorConfig:
    name: str
    direction_local: tuple[float, float, float]
    max_range_m: float
    field_of_view_deg: float
    noise_std_m: float
    update_rate_hz: float


@dataclass(frozen=True)
class DirectionalFrictionConfig:
    enabled: bool
    forward_sliding_coefficient: float
    reverse_sliding_coefficient: float
    lateral_coefficient: float
    transition_speed_m_s: float


@dataclass(frozen=True)
class RobotConfig:
    link_count: int
    link_lengths_m: tuple[float, ...]
    width_m: float
    thickness_m: float
    link_masses_kg: tuple[float, ...]
    center_of_mass_offsets_m: tuple[tuple[float, float, float], ...]
    inertia_diagonal_kg_m2: tuple[tuple[float, float, float], ...] | None
    joint: JointConfig
    contact: ContactConfig
    gripper: GripperConfig
    attachment: AttachmentConfig
    range_sensors: tuple[RangeSensorConfig, ...]
    directional_friction: DirectionalFrictionConfig

    @property
    def joint_count(self) -> int:
        return self.link_count - 1

    @property
    def total_mass_kg(self) -> float:
        return sum(self.link_masses_kg)

    def validate(self) -> None:
        if self.link_count < 2:
            raise ValueError("link_count must be at least 2")
        for name, values, expected in (
            ("link_lengths_m", self.link_lengths_m, self.link_count),
            ("link_masses_kg", self.link_masses_kg, self.link_count),
            (
                "center_of_mass_offsets_m",
                self.center_of_mass_offsets_m,
                self.link_count,
            ),
            ("joint.limits_rad", self.joint.limits_rad, self.joint_count),
        ):
            if len(values) != expected:
                raise ValueError(f"{name} requires {expected} entries")
        if (
            self.inertia_diagonal_kg_m2 is not None
            and len(self.inertia_diagonal_kg_m2) != self.link_count
        ):
            raise ValueError("inertia_diagonal_kg_m2 requires one vector per link")
        if any(length <= 0 for length in self.link_lengths_m):
            raise ValueError("link lengths must be positive")
        if any(mass <= 0 for mass in self.link_masses_kg):
            raise ValueError("link masses must be positive")
        if self.width_m <= 0 or self.thickness_m <= 0:
            raise ValueError("robot width and thickness must be positive")
        if self.contact.shape not in {"box", "capsule"}:
            raise ValueError("contact.shape must be 'box' or 'capsule'")
        if self.gripper.location != "front":
            raise ValueError(
                "initial implementation supports gripper.location='front' only"
            )
        if self.joint.max_torque_nm <= 0 or self.joint.max_speed_rad_s <= 0:
            raise ValueError("joint torque and speed limits must be positive")
        if self.joint.damping_nms_rad < 0 or self.joint.armature_kg_m2 < 0:
            raise ValueError("joint damping and armature must be non-negative")
        directional = self.directional_friction
        if (
            min(
                directional.forward_sliding_coefficient,
                directional.reverse_sliding_coefficient,
                directional.lateral_coefficient,
            )
            < 0
        ):
            raise ValueError("directional friction coefficients must be non-negative")
        if self.attachment.break_force_n <= 0:
            raise ValueError("attachment break force must be positive")
        if self.attachment.failure_model not in {"force", "combined"}:
            raise ValueError("attachment failure_model must be force or combined")
        names = [sensor.name for sensor in self.range_sensors]
        if len(names) != len(set(names)):
            raise ValueError("range sensor names must be unique")


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    surface_friction: tuple[float, float, float]
    platform_length_m: float = 1.2
    platform_width_m: float = 1.0
    platform_thickness_m: float = 0.10
    platform_height_m: float = 0.10
    gap_width_m: float = 0.30
    step_height_m: float = 0.22
    step_width_m: float = 0.35
    step_depth_m: float = 1.0
    step_x_m: float = 0.60


@dataclass(frozen=True)
class SimulationConfig:
    timestep_s: float
    gravity_m_s2: float
    integrator: str
    solver_iterations: int
    robot_count: int
    formation_columns: int
    formation_row_gap_m: float
    formation_column_gap_m: float
    random_seed: int
    environment: str
    spawn_x_m: float
    spawn_clearance_m: float
    log_interval_s: float
    realtime_scale: float

    def validate(self) -> None:
        if self.timestep_s <= 0:
            raise ValueError("timestep_s must be positive")
        if self.robot_count < 1:
            raise ValueError("robot_count must be at least 1")
        if self.formation_columns < 1:
            raise ValueError("formation_columns must be at least 1")
        if self.formation_row_gap_m < 0 or self.formation_column_gap_m < 0:
            raise ValueError("formation gaps must be non-negative")
        if self.integrator not in {"Euler", "RK4", "implicit", "implicitfast"}:
            raise ValueError("unsupported MuJoCo integrator")
        if self.log_interval_s <= 0 or self.realtime_scale <= 0:
            raise ValueError("log interval and realtime scale must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    robot: RobotConfig
    environment: EnvironmentConfig
    simulation: SimulationConfig

    def with_overrides(
        self,
        *,
        robot_count: int | None = None,
        robot_rows: int | None = None,
        environment: str | None = None,
        random_seed: int | None = None,
    ) -> ExperimentConfig:
        if robot_count is not None and robot_rows is not None:
            raise ValueError("robot_count and robot_rows cannot both be overridden")
        if robot_rows is not None and robot_rows < 1:
            raise ValueError("robot_rows must be at least 1")
        resolved_robot_count = (
            robot_rows * self.simulation.formation_columns
            if robot_rows is not None
            else self.simulation.robot_count
            if robot_count is None
            else robot_count
        )
        simulation = replace(
            self.simulation,
            robot_count=resolved_robot_count,
            environment=self.simulation.environment
            if environment is None
            else environment,
            random_seed=self.simulation.random_seed
            if random_seed is None
            else random_seed,
        )
        env = self.environment
        if environment is not None and environment != env.name:
            env = load_environment_config(DEFAULT_ENVIRONMENT_CONFIG, environment)
        simulation.validate()
        return replace(self, simulation=simulation, environment=env)


def load_robot_config(path: Path = DEFAULT_ROBOT_CONFIG) -> RobotConfig:
    raw = _load_toml(path)["robot"]
    link_count = int(raw["link_count"])
    com_offsets = tuple(
        _vec3(value, "center_of_mass_offsets_m")
        for value in raw["center_of_mass_offsets_m"]
    )
    inertia_raw = raw.get("inertia_diagonal_kg_m2", [])
    inertia = (
        tuple(_vec3(value, "inertia_diagonal_kg_m2") for value in inertia_raw)
        if inertia_raw
        else None
    )
    joint_raw = raw["joint"]
    limits = tuple(
        (float(pair[0]) * pi / 180.0, float(pair[1]) * pi / 180.0)
        for pair in joint_raw["limits_deg"]
    )
    contact_raw = raw["contact"]
    gripper_raw = raw["gripper"]
    attachment_raw = raw["attachment"]
    friction_raw = raw["directional_friction"]
    sensors = tuple(
        RangeSensorConfig(
            name=str(item["name"]),
            direction_local=_vec3(item["direction_local"], "direction_local"),
            max_range_m=float(item["max_range_m"]),
            field_of_view_deg=float(item["field_of_view_deg"]),
            noise_std_m=float(item["noise_std_m"]),
            update_rate_hz=float(item["update_rate_hz"]),
        )
        for item in raw["range_sensors"]
    )
    config = RobotConfig(
        link_count=link_count,
        link_lengths_m=_tuple_floats(raw["link_lengths_m"], "link_lengths_m"),
        width_m=float(raw["width_m"]),
        thickness_m=float(raw["thickness_m"]),
        link_masses_kg=_tuple_floats(raw["link_masses_kg"], "link_masses_kg"),
        center_of_mass_offsets_m=com_offsets,
        inertia_diagonal_kg_m2=inertia,
        joint=JointConfig(
            limits_rad=limits,
            max_torque_nm=float(joint_raw["max_torque_nm"]),
            max_speed_rad_s=float(joint_raw["max_speed_rad_s"]),
            damping_nms_rad=float(joint_raw["damping_nms_rad"]),
            stiffness_nm_rad=float(joint_raw["stiffness_nm_rad"]),
            armature_kg_m2=float(joint_raw["armature_kg_m2"]),
            position_kp_nm_rad=float(joint_raw["position_kp_nm_rad"]),
            velocity_kd_nms_rad=float(joint_raw["velocity_kd_nms_rad"]),
        ),
        contact=ContactConfig(
            shape=str(contact_raw["shape"]),
            friction=_vec3(contact_raw["friction"], "contact.friction"),
            robot_robot_friction=_vec3(
                contact_raw["robot_robot_friction"],
                "contact.robot_robot_friction",
            ),
            condim=int(contact_raw["condim"]),
            solref=tuple(_tuple_floats(contact_raw["solref"], "contact.solref")),
            solimp=tuple(_tuple_floats(contact_raw["solimp"], "contact.solimp")),
        ),
        gripper=GripperConfig(
            length_m=float(gripper_raw["length_m"]),
            width_m=float(gripper_raw["width_m"]),
            thickness_m=float(gripper_raw["thickness_m"]),
            location=str(gripper_raw["location"]),
        ),
        attachment=AttachmentConfig(
            break_force_n=float(attachment_raw["break_force_n"]),
            break_moment_nm=float(attachment_raw["break_moment_nm"]),
            failure_model=str(attachment_raw["failure_model"]),
            allow_environment=bool(attachment_raw["allow_environment"]),
            solref=tuple(_tuple_floats(attachment_raw["solref"], "attachment.solref")),
            solimp=tuple(_tuple_floats(attachment_raw["solimp"], "attachment.solimp")),
        ),
        range_sensors=sensors,
        directional_friction=DirectionalFrictionConfig(
            enabled=bool(friction_raw["enabled"]),
            forward_sliding_coefficient=float(
                friction_raw["forward_sliding_coefficient"]
            ),
            reverse_sliding_coefficient=float(
                friction_raw["reverse_sliding_coefficient"]
            ),
            lateral_coefficient=float(friction_raw["lateral_coefficient"]),
            transition_speed_m_s=float(friction_raw["transition_speed_m_s"]),
        ),
    )
    config.validate()
    return config


def load_environment_config(
    path: Path = DEFAULT_ENVIRONMENT_CONFIG,
    name: str = "flat",
) -> EnvironmentConfig:
    all_raw = _load_toml(path)
    if name not in all_raw:
        raise ValueError(f"unknown environment {name!r}; choose flat, gap, or step")
    raw: Mapping[str, Any] = all_raw[name]
    config = EnvironmentConfig(
        name=name,
        surface_friction=_vec3(raw["surface_friction"], f"{name}.surface_friction"),
        platform_length_m=float(raw.get("platform_length_m", 1.2)),
        platform_width_m=float(raw.get("platform_width_m", 1.0)),
        platform_thickness_m=float(raw.get("platform_thickness_m", 0.10)),
        platform_height_m=float(raw.get("platform_height_m", 0.10)),
        gap_width_m=float(raw.get("gap_width_m", 0.30)),
        step_height_m=float(raw.get("step_height_m", 0.22)),
        step_width_m=float(raw.get("step_width_m", 0.35)),
        step_depth_m=float(raw.get("step_depth_m", 1.0)),
        step_x_m=float(raw.get("step_x_m", 0.60)),
    )
    if config.gap_width_m <= 0 or config.step_height_m <= 0:
        raise ValueError("gap width and step height must be positive")
    return config


def load_simulation_config(path: Path = DEFAULT_SIMULATION_CONFIG) -> SimulationConfig:
    raw = _load_toml(path)["simulation"]
    formation_columns = int(raw.get("robot_columns", 1))
    robot_rows = int(raw.get("robot_rows", raw.get("robot_count", 1)))
    config = SimulationConfig(
        timestep_s=float(raw["timestep_s"]),
        gravity_m_s2=float(raw["gravity_m_s2"]),
        integrator=str(raw["integrator"]),
        solver_iterations=int(raw["solver_iterations"]),
        robot_count=robot_rows * formation_columns,
        formation_columns=formation_columns,
        formation_row_gap_m=float(raw.get("formation_row_gap_m", 0.0)),
        formation_column_gap_m=float(raw.get("formation_column_gap_m", 0.0)),
        random_seed=int(raw["random_seed"]),
        environment=str(raw["environment"]),
        spawn_x_m=float(raw["spawn_x_m"]),
        spawn_clearance_m=float(raw["spawn_clearance_m"]),
        log_interval_s=float(raw["log_interval_s"]),
        realtime_scale=float(raw["realtime_scale"]),
    )
    config.validate()
    return config


def load_experiment_config(
    robot_path: Path = DEFAULT_ROBOT_CONFIG,
    environment_path: Path = DEFAULT_ENVIRONMENT_CONFIG,
    simulation_path: Path = DEFAULT_SIMULATION_CONFIG,
) -> ExperimentConfig:
    simulation = load_simulation_config(simulation_path)
    return ExperimentConfig(
        robot=load_robot_config(robot_path),
        environment=load_environment_config(environment_path, simulation.environment),
        simulation=simulation,
    )
