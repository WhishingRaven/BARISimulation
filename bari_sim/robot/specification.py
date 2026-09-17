"""Immutable physical and timing specification for every BARI robot."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class RobotSpecification:
    """The user-owned robot specification, expressed in SI units.

    The three body segments are ordered rear -> middle -> front.  Their lengths
    put the rear hinge 6 cm from the rear edge and leave a 2 cm front flap.
    """

    width_m: float = 0.10
    length_m: float = 0.15
    height_m: float = 0.005
    mass_kg: float = 0.05
    rear_length_m: float = 0.06
    middle_length_m: float = 0.07
    front_length_m: float = 0.02
    front_sensor_offset_m: float = 0.04
    maximum_strain_g: float = 100.0
    control_interval_s: float = 0.5

    # Implementation parameters for the pre-tuned discrete actions.
    curl_angle_rad: float = radians(52.0)
    front_lift_angle_rad: float = radians(-52.0)
    hinge_limit_rad: float = radians(65.0)
    joint_torque_nm: float = 0.045
    joint_speed_rad_s: float = radians(180.0)
    joint_kp_nm_rad: float = 0.55
    joint_kd_nms_rad: float = 0.012
    # The yaw motor is applied while the rear cleat is physically latched.
    # This is a torque limit, not an imposed orientation increment.
    turn_torque_nm: float = 0.020
    turn_speed_rad_s: float = radians(10.0)
    turn_rate_kp_nms_rad: float = 0.12
    turn_angle_rad: float = radians(5.0)
    sensor_range_m: float = 1.0
    communication_range_m: float = 0.50
    attachment_contact_tolerance_m: float = 0.012

    @property
    def segment_lengths_m(self) -> tuple[float, float, float]:
        return self.rear_length_m, self.middle_length_m, self.front_length_m

    @property
    def segment_masses_kg(self) -> tuple[float, float, float]:
        return tuple(
            self.mass_kg * length / self.length_m for length in self.segment_lengths_m
        )  # type: ignore[return-value]

    @property
    def maximum_attachment_force_n(self) -> float:
        return self.maximum_strain_g / 1000.0 * GRAVITY_M_S2

    def validate(self) -> None:
        if abs(sum(self.segment_lengths_m) - self.length_m) > 1e-12:
            raise ValueError("robot segment lengths must total exactly 0.15 m")
        if (self.width_m, self.length_m, self.height_m, self.mass_kg) != (
            0.10,
            0.15,
            0.005,
            0.05,
        ):
            raise ValueError("the fixed 10 x 15 x 0.5 cm, 50 g specification changed")
        if self.front_length_m != 0.02 or self.rear_length_m != 0.06:
            raise ValueError("front and rear articulated lengths are fixed")
        if self.front_sensor_offset_m != 0.04:
            raise ValueError(
                "the lateral/down sensor station must be 4 cm from the front"
            )
        if self.control_interval_s != 0.5:
            raise ValueError("the policy interval is fixed at 0.5 seconds")


DEFAULT_ROBOT = RobotSpecification()
DEFAULT_ROBOT.validate()
