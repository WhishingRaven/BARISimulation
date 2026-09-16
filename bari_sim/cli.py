"""Command-line entry points and viewer loops."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from math import pi
from pathlib import Path

from .config import PROJECT_ROOT, load_experiment_config
from .controllers import (
    BridgeBaselineController,
    ExplorationController,
    InchwormController,
    ManualTeleop,
    TravelingWaveGait,
)
from .simulator import SwarmSimulator
from .types import LocalObservation, RobotAction
from .visualization import DebugVisualizer


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--environment", choices=("flat", "gap", "step"), default=None)
    parser.add_argument(
        "--robots",
        type=int,
        default=None,
        metavar="ROWS",
        help="formation rows; each row contains five tightly packed robots",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-log", action="store_true", help="disable CSV logging")
    return parser


def _make_simulator(
    args, *, environment: str | None = None, robot_count: int | None = None
) -> SwarmSimulator:
    config = load_experiment_config().with_overrides(
        robot_count=robot_count,
        robot_rows=None if robot_count is not None else args.robots,
        environment=environment if environment is not None else args.environment,
        random_seed=args.seed,
    )
    return SwarmSimulator(config, logging_enabled=not args.no_log)


def manual_main(
    argv: Sequence[str] | None = None, *, forced_environment: str | None = None
) -> None:
    parser = _common_parser("Manually control multiple articulated robots")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional wall-clock duration, useful for viewer smoke tests",
    )
    args = parser.parse_args(argv)
    simulator = _make_simulator(args, environment=forced_environment)
    teleop = ManualTeleop(simulator.robot_count, simulator.config.robot.joint_count)
    observations = simulator.get_observations()
    visualizer = DebugVisualizer()
    try:
        import mujoco.viewer

        try:
            viewer_context = mujoco.viewer.launch_passive(
                simulator.model,
                simulator.data,
                key_callback=teleop.key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
        except RuntimeError as error:
            if sys.platform == "darwin" and "mjpython" in str(error):
                raise SystemExit(
                    "macOS passive viewer requires mjpython. Run: "
                    f"mjpython {Path(sys.argv[0]).as_posix()} {' '.join(sys.argv[1:])}"
                ) from error
            raise
        with viewer_context as viewer:
            viewer_start = time.monotonic()
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -28
            viewer.cam.lookat[:] = (0.0, 0.0, 0.10)
            while viewer.is_running():
                if (
                    args.duration is not None
                    and time.monotonic() - viewer_start >= args.duration
                ):
                    break
                frame_start = time.monotonic()
                actions = teleop.update(simulator, observations)
                if teleop.state.paused:
                    for robot_id, action in actions.items():
                        simulator.robot(robot_id).apply_action(action)
                    mujoco.mj_forward(simulator.model, simulator.data)
                    observations = simulator.get_observations()
                else:
                    observations = simulator.step(actions).observations
                selected = teleop.state.selected_robot_id
                visualizer.update(
                    viewer,
                    simulator,
                    selected,
                    show_sensor_rays=teleop.state.debug_visible,
                )
                visualizer.set_text(viewer, teleop, observations[selected], simulator)
                viewer.sync()
                target_wall_step = simulator.timestep_s / teleop.state.realtime_scale
                remaining = target_wall_step - (time.monotonic() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        simulator.close()


def gait_test_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser(
        "Measure physics-produced displacement from an experimental gait"
    )
    parser.set_defaults(environment="flat")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--settle", type=float, default=0.6)
    parser.add_argument("--gait", choices=("wave", "inchworm"), default="wave")
    parser.add_argument("--amplitude-deg", type=float, default=40.1)
    parser.add_argument("--frequency-hz", type=float, default=1.5)
    parser.add_argument("--phase-deg", type=float, default=75.0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args(argv)
    if args.environment == "flat":
        environment = "flat"
    else:
        environment = args.environment
    simulator = _make_simulator(args, environment=environment, robot_count=1)
    if args.gait == "inchworm":
        controller = InchwormController(
            bend_amplitude_rad=args.amplitude_deg * pi / 180.0
        )
    else:
        controller = TravelingWaveGait(
            amplitude_rad=args.amplitude_deg * pi / 180.0,
            frequency_hz=args.frequency_hz,
            phase_offset_rad=args.phase_deg * pi / 180.0,
        )
    try:
        _settle(simulator, args.settle)
        initial = simulator.robot(0).get_global_state().position_m
        if args.viewer:
            _run_controller_viewer(simulator, {0: controller}, args.duration)
        else:
            _run_controllers_headless(simulator, {0: controller}, args.duration)
        final = simulator.robot(0).get_global_state().position_m
        print(
            json.dumps(
                {
                    "gait": args.gait,
                    "duration_s": args.duration,
                    "initial_position_m": initial,
                    "final_position_m": final,
                    "displacement_m": tuple(
                        final[index] - initial[index] for index in range(3)
                    ),
                    "note": "displacement is unassisted MuJoCo physics; zero/negative results are retained",
                },
                indent=2,
            )
        )
    finally:
        simulator.close()


def attachment_test_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser("Ramp load on a gripper attachment until it fails")
    parser.add_argument("--settle", type=float, default=0.8)
    parser.add_argument("--ramp-duration", type=float, default=3.0)
    parser.add_argument("--maximum-force", type=float, default=None)
    args = parser.parse_args(argv)
    simulator = _make_simulator(args, environment="flat", robot_count=1)
    try:
        _settle(simulator, args.settle)
        contact_observation = simulator.get_observations()[0]
        for _ in range(max(1, int(1.5 / simulator.timestep_s))):
            if contact_observation.gripper_contact:
                break
            contact_observation = simulator.step(
                {
                    0: RobotAction(
                        joint_targets_rad=(0.50,) * simulator.config.robot.joint_count
                    )
                }
            ).observations[0]
        if contact_observation.gripper_contact:
            for _ in range(max(1, int(0.8 / simulator.timestep_s))):
                contact_observation = simulator.step(
                    {
                        0: RobotAction(
                            joint_targets_rad=(0.50,)
                            * simulator.config.robot.joint_count
                        )
                    }
                ).observations[0]
        attached = simulator.robot(0).attach()
        maximum = (
            args.maximum_force or 4.0 * simulator.config.robot.attachment.break_force_n
        )
        failure_event = None
        commanded_force = 0.0
        if attached:
            steps = max(1, int(args.ramp_duration / simulator.timestep_s))
            for index in range(steps):
                commanded_force = maximum * (index + 1) / steps
                result = simulator.step(
                    external_forces_n={0: (0.0, 0.0, commanded_force)}
                )
                failure_event = next(
                    (
                        event
                        for event in result.attachment_events
                        if event.event == "attachment_failure"
                    ),
                    None,
                )
                if failure_event is not None:
                    break
        print(
            json.dumps(
                {
                    "attached_from_contact": attached,
                    "configured_break_force_n": simulator.config.robot.attachment.break_force_n,
                    "commanded_force_at_end_n": commanded_force,
                    "measured_failure_load_n": None
                    if failure_event is None
                    else failure_event.force_n,
                    "failed": failure_event is not None,
                },
                indent=2,
            )
        )
    finally:
        simulator.close()


def baseline_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser("Run decentralized local-rule controllers")
    parser.add_argument(
        "--controller", choices=("exploration", "bridge"), default="exploration"
    )
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args(argv)
    simulator = _make_simulator(args)
    controller_type = (
        BridgeBaselineController
        if args.controller == "bridge"
        else ExplorationController
    )
    controllers = {
        robot_id: controller_type() for robot_id in range(simulator.robot_count)
    }
    try:
        if args.viewer:
            observations = _run_controller_viewer(simulator, controllers, args.duration)
        else:
            observations = _run_controllers_headless(
                simulator, controllers, args.duration
            )
        print(
            json.dumps(
                {
                    "controller": args.controller,
                    "duration_s": args.duration,
                    "environment": simulator.config.environment.name,
                    "crossed_robot_ids": sorted(
                        simulator.environment.crossed_robot_ids
                    ),
                    "attachments_active": [
                        robot_id
                        for robot_id, observation in observations.items()
                        if observation.attachment.active
                    ],
                },
                indent=2,
            )
        )
    finally:
        simulator.close()


def export_model_main(argv: Sequence[str] | None = None) -> None:
    parser = _common_parser("Export generated MJCF for inspection")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "generated_scene.xml"
    )
    args = parser.parse_args(argv)
    simulator = _make_simulator(args)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(simulator.built_model.xml, encoding="utf-8")
        print(args.output.resolve())
    finally:
        simulator.close()


def _settle(simulator: SwarmSimulator, duration_s: float) -> None:
    for _ in range(max(0, int(duration_s / simulator.timestep_s))):
        simulator.step()


def _run_controllers_headless(
    simulator: SwarmSimulator,
    controllers: Mapping[int, object],
    duration_s: float,
) -> Mapping[int, LocalObservation]:
    observations = simulator.get_observations()
    steps = max(0, int(duration_s / simulator.timestep_s))
    for _ in range(steps):
        actions = {
            robot_id: controller.act(observations[robot_id])
            for robot_id, controller in controllers.items()
        }
        observations = simulator.step(actions).observations
    return observations


def _run_controller_viewer(
    simulator: SwarmSimulator,
    controllers: Mapping[int, object],
    duration_s: float,
) -> Mapping[int, LocalObservation]:
    import mujoco.viewer

    observations = simulator.get_observations()
    visualizer = DebugVisualizer()
    start_simulation_time = float(simulator.data.time)
    try:
        context = mujoco.viewer.launch_passive(simulator.model, simulator.data)
    except RuntimeError as error:
        if sys.platform == "darwin" and "mjpython" in str(error):
            raise SystemExit("macOS viewer runs require mjpython") from error
        raise
    with context as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -28
        while (
            viewer.is_running()
            and float(simulator.data.time) - start_simulation_time < duration_s
        ):
            frame_start = time.monotonic()
            actions = {
                robot_id: controller.act(observations[robot_id])
                for robot_id, controller in controllers.items()
            }
            observations = simulator.step(actions).observations
            visualizer.update(viewer, simulator, 0)
            if hasattr(viewer, "set_texts"):
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Controller\nEnvironment\nTime",
                        f"decentralized local rules\n{simulator.config.environment.name}\n{simulator.data.time:.2f} s",
                    )
                )
            viewer.sync()
            remaining = simulator.timestep_s - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    return observations


def run_gap_main(argv: Sequence[str] | None = None) -> None:
    manual_main(argv, forced_environment="gap")


def run_step_main(argv: Sequence[str] | None = None) -> None:
    manual_main(argv, forced_environment="step")
