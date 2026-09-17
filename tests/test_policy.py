from __future__ import annotations

from bari_sim.policies import LinearPolicy, PolicyMetadata
from bari_sim.robot import GripAction, LiftAction, MotionAction, RobotObservation


def test_policy_round_trip_uses_only_local_observation(tmp_path) -> None:
    metadata = PolicyMetadata("gap", 2, "2*5", 7)
    policy = LinearPolicy.idle(metadata)
    path = tmp_path / "policy.json"
    policy.save(path)
    loaded = LinearPolicy.load(path)
    observation = RobotObservation(
        nearby_robot_ids=(1, 2),
        strain_value=0.0,
        distance1=1.0,
        distance2=0.01,
        distance3=1.0,
        distance4=1.0,
        is_curled=False,
        is_front_lifted=False,
        is_possible_to_attach=True,
        is_attaching=False,
        is_detached=False,
    )
    action = loaded.act(observation)
    assert action.motion is MotionAction.STOP
    assert action.lift is LiftAction.STOP
    assert action.grip is GripAction.STOP
    assert loaded.metadata == metadata
