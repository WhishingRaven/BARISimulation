from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from math import isclose
from xml.etree import ElementTree as ET

from bari_sim.config import load_experiment_config
from bari_sim.model_builder import ModelBuilder, box_inertia, lateral_support_width


def test_default_robot_has_three_links_and_weight_scale_attachment() -> None:
    config = load_experiment_config()

    assert config.robot.link_count == 3
    assert config.robot.joint_count == 2
    assert config.robot.joint.max_speed_rad_s == 40.0
    assert config.robot.width_m == 0.110
    assert config.robot.thickness_m == 0.00625
    assert config.simulation.robot_count == 15
    assert config.simulation.formation_columns == 5
    assert isclose(config.robot.total_mass_kg, 0.240)
    assert isclose(
        config.robot.attachment.break_force_n,
        config.robot.total_mass_kg * config.simulation.gravity_m_s2,
        rel_tol=0.01,
    )


def test_robot_row_override_builds_dense_five_column_formation() -> None:
    config = load_experiment_config().with_overrides(robot_rows=3, environment="flat")
    built = ModelBuilder(config).build()
    positions = built.spawn_positions_m
    assert len(positions) == 15
    assert len({position[0] for position in positions[:5]}) == 1
    assert len({position[0] for position in positions[5:10]}) == 1
    assert positions[5][0] < positions[0][0]
    for first, second in pairwise(positions[:5]):
        assert isclose(second[1] - first[1], lateral_support_width(config.robot))
    straight_length = sum(config.robot.link_lengths_m) + config.robot.gripper.length_m
    assert isclose(positions[0][0] - positions[5][0], straight_length)


def test_each_robot_has_twelve_surface_force_sensors() -> None:
    config = load_experiment_config().with_overrides(robot_count=1)
    root = ET.fromstring(ModelBuilder(config).build().xml)
    sensors = root.findall("./sensor/touch")
    assert len(sensors) == 12
    names = {sensor.attrib["name"] for sensor in sensors}
    expected = {
        f"robot_0_link_{link_id}_{location}_force"
        for link_id in range(3)
        for location in ("top_rear", "top_front", "left", "right")
    }
    assert names == expected
    for sensor in sensors:
        assert root.find(f".//site[@name='{sensor.attrib['site']}']") is not None


def test_generated_model_contains_separate_visual_and_collision_geometry() -> None:
    config = load_experiment_config().with_overrides(robot_count=3, environment="gap")
    built = ModelBuilder(config).build()
    root = ET.fromstring(built.xml)

    collision = root.find(".//geom[@name='robot_0_link_0_collision']")
    visual = root.find(".//geom[@name='robot_0_link_0_visual']")
    assert collision is not None
    assert visual is not None
    assert collision.attrib["group"] == "3"
    assert visual.attrib["group"] == "1"
    assert visual.attrib["contype"] == "0"

    motors = root.findall("./actuator/motor")
    assert len(motors) == 3 * 2
    equalities = root.findall("./equality/connect")
    assert len(equalities) == 3 * (1 + 2 * 3)
    assert all(item.attrib["active"] == "false" for item in equalities)
    robot_pairs = root.findall("./contact/pair")
    # Three link shells, six side outriggers, and one gripper per robot.
    assert len(robot_pairs) == 3 * 10 * 10
    assert all(item.attrib["friction"].startswith("0 0") for item in robot_pairs)


def test_box_inertia_uses_full_dimensions() -> None:
    inertia = box_inertia(2.0, 3.0, 4.0, 5.0)
    assert inertia == (
        2.0 * (4.0**2 + 5.0**2) / 12.0,
        2.0 * (3.0**2 + 5.0**2) / 12.0,
        2.0 * (3.0**2 + 4.0**2) / 12.0,
    )


def test_anchor_pads_use_explicit_asymmetric_friction_model() -> None:
    config = load_experiment_config().with_overrides(robot_count=2, environment="flat")
    root = ET.fromstring(ModelBuilder(config).build().xml)

    robot_geom = root.find(".//geom[@name='robot_0_link_0_collision']")
    ground_geom = root.find(".//geom[@name='environment_flat_ground']")
    robot_pair = root.find("./contact/pair")
    assert robot_geom is not None and robot_geom.attrib["friction"].startswith("0 ")
    assert ground_geom is not None and ground_geom.attrib["friction"].startswith("0 ")
    assert robot_pair is not None and robot_pair.attrib["friction"].startswith("0 0 ")
    pads = [
        geom
        for geom in root.findall(".//geom")
        if "_anchor_pad_" in geom.attrib["name"]
    ]
    assert len(pads) == 2 * 2 * config.robot.link_count
    assert all(pad.attrib["friction"] == "0 0 0" for pad in pads)
    for pad in pads:
        center_y = abs(float(pad.attrib["pos"].split()[1]))
        half_width = float(pad.attrib["size"].split()[1])
        assert center_y + half_width <= config.robot.width_m / 2.0 + 1e-12


def test_model_generation_supports_configured_link_count_without_policy_changes() -> (
    None
):
    config = load_experiment_config().with_overrides(robot_count=1, environment="flat")
    robot = config.robot
    four_link_robot = replace(
        robot,
        link_count=4,
        link_lengths_m=robot.link_lengths_m + (0.16,),
        link_masses_kg=robot.link_masses_kg + (0.065,),
        center_of_mass_offsets_m=robot.center_of_mass_offsets_m + ((0.0, 0.0, 0.0),),
        joint=replace(robot.joint, limits_rad=robot.joint.limits_rad + ((-1.0, 1.0),)),
    )
    four_link_robot.validate()
    root = ET.fromstring(
        ModelBuilder(replace(config, robot=four_link_robot)).build().xml
    )

    assert root.find(".//body[@name='robot_0_link_3']") is not None
    assert len(root.findall("./actuator/motor")) == 3
    assert len(root.findall("./sensor/touch")) == 16
