"""Policy rollout with an optional live MuJoCo viewer."""

from __future__ import annotations

from ..policies import LinearPolicy
from ..simulation import Simulation
from ..tasks import TaskResult


def run_inference(
    simulation: Simulation,
    policy: LinearPolicy,
    *,
    duration_s: float,
    viewer_enabled: bool,
) -> TaskResult:
    if not viewer_enabled:
        result = simulation.run(
            lambda _robot_id, observation: policy.act(observation), duration_s
        )
        assert result is not None
        return result

    import mujoco.viewer

    observations = simulation.observations()
    end_time = simulation.time_s + duration_s
    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -28
        viewer.cam.distance = max(1.4, 0.35 * simulation.request.grid.rows + 1.2)
        while viewer.is_running() and simulation.time_s < end_time:
            actions = {
                robot_id: policy.act(observations[robot_id])
                for robot_id in range(simulation.robot_count)
            }
            result = simulation.step(
                actions,
                frame_callback=lambda _simulation: viewer.sync() or viewer.is_running(),
                realtime=True,
            )
            observations = result.observations
            if simulation.evaluator is not None and simulation.evaluator.complete:
                break
    assert simulation.evaluator is not None
    return simulation.evaluator.result()
