from __future__ import annotations

import mujoco
import pytest

from bari_sim.robot import DEFAULT_ROBOT, RobotObservation
from bari_sim.simulation import SceneBuilder, SceneRequest
from bari_sim.tasks import RobotGrid


def test_fixed_robot_specification() -> None:
    robot = DEFAULT_ROBOT
    assert (robot.width_m, robot.length_m, robot.height_m, robot.mass_kg) == (
        0.10,
        0.15,
        0.01,
        0.05,
    )
    assert robot.segment_lengths_m == (0.06, 0.07, 0.02)
    assert sum(robot.segment_masses_kg) == pytest.approx(0.05)
    assert robot.front_sensor_offset_m == 0.04
    assert robot.maximum_strain_g == 50.0
    assert robot.maximum_attachment_force_n == pytest.approx(0.4905)
    assert robot.control_interval_s == 0.5


def test_sensors_and_spike_are_sites_not_physical_geometries() -> None:
    scene = SceneBuilder(SceneRequest(RobotGrid(1, 1), "flat")).build()
    model = mujoco.MjModel.from_xml_string(scene.xml)
    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(model.ngeom)
    }
    site_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index)
        for index in range(model.nsite)
    }
    assert not any("sensor" in name or "spike" in name for name in geom_names)
    assert "robot_0_spike_site" in site_names
    assert {
        "robot_0_front_sensor",
        "robot_0_down_sensor",
        "robot_0_left_sensor",
        "robot_0_right_sensor",
    } <= site_names
    rear_body = model.body("robot_0_rear_body").id
    middle_body = model.body("robot_0_middle_body").id
    front_body = model.body("robot_0_front_body").id
    assert sum(model.body_mass[[rear_body, middle_body, front_body]]) == pytest.approx(
        0.05
    )
    assert tuple(model.body_pos[middle_body]) == pytest.approx((0.03, 0.0, 0.0))
    assert tuple(model.body_pos[front_body]) == pytest.approx((0.07, 0.0, 0.0))
    assert tuple(model.site("robot_0_spike_site").pos) == pytest.approx(
        (-0.03, 0.0, -0.005)
    )
    assert tuple(model.site("robot_0_front_sensor").pos) == pytest.approx(
        (0.02, 0.0, 0.0)
    )
    assert tuple(model.site("robot_0_down_sensor").pos) == pytest.approx(
        (0.05, 0.0, -0.005)
    )
    assert tuple(model.site("robot_0_left_sensor").pos) == pytest.approx(
        (0.05, 0.05, 0.0)
    )
    assert tuple(model.site("robot_0_right_sensor").pos) == pytest.approx(
        (0.05, -0.05, 0.0)
    )


def test_observation_contains_only_declared_policy_fields() -> None:
    expected = {
        "nearby_robot_ids",
        "strain_value",
        "distance1",
        "distance2",
        "distance3",
        "distance4",
        "is_curled",
        "is_front_lifted",
        "is_possible_to_attach",
        "is_attaching",
        "is_detached",
    }
    assert set(RobotObservation.__dataclass_fields__) == expected
