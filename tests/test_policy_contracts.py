from __future__ import annotations

from math import isclose, pi

from bari_sim.controllers import (
    BridgeBaselineController,
    ExplorationController,
    TravelingWaveGait,
)
from bari_sim.types import (
    AttachmentObservation,
    ForceReading,
    LocalObservation,
    RangeReading,
    RobotAction,
)


def observation(*, edge: bool = False, attached: bool = False) -> LocalObservation:
    return LocalObservation(
        robot_id=0,
        simulation_time_s=0.25,
        joint_positions_rad=(0.0, 0.0),
        joint_velocities_rad_s=(0.0, 0.0),
        force_sensors={
            f"link_{link_id}.{location}": ForceReading()
            for link_id in range(3)
            for location in ("top_rear", "top_front", "left", "right")
        },
        ranges={
            "forward": RangeReading(0.60, False),
            "forward_left": RangeReading(0.60, False),
            "forward_right": RangeReading(0.60, False),
            "forward_down": RangeReading(0.60 if edge else 0.08, not edge),
        },
        attachment=AttachmentObservation(active=attached, utilization=0.2),
        gripper_contact=edge,
    )


def test_local_observation_has_no_privileged_pose_or_map() -> None:
    keys = observation().as_dict().keys()
    forbidden = {
        "position",
        "position_m",
        "orientation",
        "orientation_wxyz",
        "global_map",
        "map",
    }
    assert keys.isdisjoint(forbidden)


def test_action_rejects_conflicting_control_modes() -> None:
    try:
        RobotAction(joint_targets_rad=(0.0, 0.0), joint_torques_nm=(0.0, 0.0))
    except ValueError as error:
        assert "simultaneously" in str(error)
    else:
        raise AssertionError("conflicting action was accepted")


def test_traveling_wave_has_requested_phase_offset() -> None:
    gait = TravelingWaveGait(
        amplitude_rad=0.5, frequency_hz=1.0, phase_offset_rad=pi / 2.0
    )
    targets = gait.targets(0.0, 2)
    assert isclose(targets[0], 0.0, abs_tol=1e-12)
    assert isclose(targets[1], 0.5, abs_tol=1e-12)


def test_positive_wave_direction_propagates_toward_robot_rear() -> None:
    gait = TravelingWaveGait(
        amplitude_rad=0.5, frequency_hz=1.0, phase_offset_rad=pi / 2.0
    )
    forward = gait.targets(0.25, 2)
    gait.direction = -1.0
    reverse = gait.targets(0.25, 2)
    assert forward[0] < 0.0
    assert reverse[0] > 0.0


def test_stability_wave_keeps_two_outer_links_in_stance() -> None:
    gait = TravelingWaveGait()
    assert gait.stance_links(0.0, 3) == (0, 2)
    assert gait.stance_links(17.0, 3) == (0, 2)


def test_decentralized_baselines_accept_only_local_observation() -> None:
    exploration = ExplorationController().act(observation())
    bridge = BridgeBaselineController().act(observation(edge=True))
    assert exploration.joint_targets_rad is not None
    assert bridge.attach
