"""The single ``barisimulation`` command and its four workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .policies import LinearPolicy
from .simulation import SceneRequest, Simulation
from .tasks import RobotGrid, TaskName, parse_robot_grid, task_definition
from .train import SUPPORTED_ALGORITHMS, TrainingSettings, train_policy
from .workflows import (
    evaluate_policy,
    run_inference,
    run_manual_viewer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CHOICES = tuple(task.value for task in TaskName)
ALGORITHMS = SUPPORTED_ALGORITHMS


def _grid_argument(value: str) -> RobotGrid:
    try:
        return parse_robot_grid(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def _elite_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 0.5:
        raise argparse.ArgumentTypeError("value must be in (0, 0.5]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barisimulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "BARI articulated swarm simulation\n\n"
            "One policy step is 0.5 s. Policies select a discrete "
            "(motion, lift, grip) tuple for every robot."
        ),
        epilog=(
            "Task matrix:\n"
            "  robots: 2*5, 4*5, 6*5\n"
            "  collision-avoidance: target distance 2, 4, 6, 8, 10 m\n"
            "  gap: width 0.10, 0.15, 0.20, 0.25, 0.30 m\n"
            "  step: height 0.03, 0.05, 0.08, 0.10, 0.15 m\n\n"
            "Run 'barisimulation help COMMAND' for command-specific details."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    help_parser = subparsers.add_parser(
        "help", help="show detailed general or command-specific help"
    )
    help_parser.add_argument(
        "topic",
        nargs="?",
        choices=("manual", "train", "infer", "evaluate"),
        help="command whose detailed help should be shown",
    )

    manual = subparsers.add_parser(
        "manual",
        help="control robots interactively with latched keyboard actions",
        description=(
            "Open the MuJoCo viewer for human control. A key press changes one "
            "component of the selected robot's action and remains latched on "
            "later 0.5 s steps until another key changes it."
        ),
        epilog=(
            "Keys: W curl, S flatten, A/D turn whole body, C stop, R lift front, "
            "F lower front, Space attach, X detach, 1-9/0 or B/N select, "
            "P pause, Z reset."
        ),
    )
    _add_grid(manual, default="1*1")
    manual.add_argument(
        "--environment", choices=("flat", "gap", "step"), default="flat"
    )
    manual.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        default=1,
        help="gap width or step height level; ignored for flat (default: 1)",
    )

    train = subparsers.add_parser(
        "train",
        help="train a shared decentralized discrete policy",
        description=(
            "Run an explicitly requested cross-entropy search over a compact "
            "shared linear policy. No training starts from other commands."
        ),
    )
    _add_grid(train)
    _add_task(train)
    _add_difficulty(train, required=True)
    train.add_argument("--algorithm", choices=ALGORITHMS, default="cem")
    train.add_argument("--output", type=Path, help="output model JSON path")
    train.add_argument("--generations", type=_positive_int, default=5)
    train.add_argument("--population", type=_positive_int, default=8)
    train.add_argument("--elite-fraction", type=_elite_fraction, default=0.25)
    train.add_argument("--initial-std", type=_positive_float, default=0.75)
    train.add_argument("--min-std", type=_positive_float, default=0.05)
    train.add_argument(
        "--episodes-per-candidate", type=_positive_int, default=1
    )
    train.add_argument("--duration", type=_positive_float, default=60.0)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--render", action="store_true", help="show each training rollout in MuJoCo")

    infer = subparsers.add_parser(
        "infer",
        help="run one trained policy rollout",
        description=(
            "Load a policy JSON and run one rollout. The model's stored "
            "difficulty is used unless --difficulty overrides it."
        ),
    )
    _add_grid(infer)
    _add_task(infer)
    _add_difficulty(infer, required=False)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--duration", type=_positive_float, default=120.0)
    infer.add_argument(
        "--render", "--viewer", dest="render", action="store_true",
        help="show the rollout in MuJoCo",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate the task metrics without changing the policy",
        description=(
            "Evaluate a saved policy. Specify the model produced by train with "
            "--model."
        ),
    )
    _add_grid(evaluate)
    _add_task(evaluate)
    _add_difficulty(evaluate, required=True)
    evaluate.add_argument("--model", type=Path, required=True, help="policy JSON path")
    evaluate.add_argument("--duration", type=_positive_float, default=120.0)
    evaluate.add_argument("--episodes", type=_positive_int, default=1)
    evaluate.add_argument("--render", action="store_true", help="show each evaluation rollout in MuJoCo")

    parser._bari_subparsers = {
        "manual": manual,
        "train": train,
        "infer": infer,
        "evaluate": evaluate,
    }
    return parser


def _add_grid(parser: argparse.ArgumentParser, default: str | None = None) -> None:
    parser.add_argument(
        "--robots",
        type=_grid_argument,
        required=default is None,
        default=None if default is None else parse_robot_grid(default),
        metavar="M*N",
        help=(
            "robot formation as rows*columns; quote it in zsh, e.g. '2*5'"
            + (f" (default: {default})" if default is not None else "")
        ),
    )


def _add_task(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=TASK_CHOICES, required=True)


def _add_difficulty(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        required=required,
        help="difficulty level 1 through 5",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "help":
        if args.topic is None:
            parser.print_help()
        else:
            parser._bari_subparsers[args.topic].print_help()
        return 0
    if args.command == "manual":
        return _manual(args)
    if args.command == "train":
        return _train(args)
    if args.command == "infer":
        return _infer(args, parser)
    if args.command == "evaluate":
        return _evaluate(args, parser)
    parser.error(f"unknown command {args.command}")
    return 2


def _manual(args: argparse.Namespace) -> int:
    _ensure_mjpython()
    task = (
        task_definition(args.environment, args.difficulty)
        if args.environment in {"gap", "step"}
        else None
    )
    simulation = Simulation(
        SceneRequest(grid=args.robots, environment=args.environment, task=task)
    )
    run_manual_viewer(simulation)
    return 0


def _train(args: argparse.Namespace) -> int:
    task = task_definition(args.task, args.difficulty)
    if args.render:
        _ensure_mjpython()
    output = args.output or _default_model_path(args.algorithm)
    print(
        f"[train] algorithm={args.algorithm} task={task.name.value} "
        f"difficulty={task.difficulty} robots={args.robots} output={output}",
        file=sys.stderr,
    )
    summary = train_policy(
        args.robots,
        task,
        output,
        TrainingSettings(
            generations=args.generations,
            population_size=args.population,
            elite_fraction=args.elite_fraction,
            initial_std=args.initial_std,
            min_std=args.min_std,
            evaluation_episodes=args.episodes_per_candidate,
            duration_s=args.duration,
            seed=args.seed,
        ),
        algorithm=args.algorithm,
        render=args.render,
        progress=lambda message: print(f"[train] {message}", file=sys.stderr),
    )
    print(
        json.dumps(summary.as_dict(), indent=2)
    )
    return 0


def _infer(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    policy = _load_policy(args.model, args.task, parser)
    difficulty = args.difficulty or policy.metadata.difficulty
    task = task_definition(args.task, difficulty)
    if args.render:
        _ensure_mjpython()
    simulation = Simulation(
        SceneRequest(grid=args.robots, environment=task.environment, task=task)
    )
    result = run_inference(
        simulation,
        policy,
        duration_s=args.duration,
        viewer_enabled=args.render,
    )
    print(
        f"[infer] model={args.model} task={task.name.value} "
        f"difficulty={task.difficulty} score={float(result.metrics['score']):.3f} "
        f"success={result.success}",
        file=sys.stderr,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def _evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    policy = _load_policy(args.model, args.task, parser)
    task = task_definition(args.task, args.difficulty)
    if args.render:
        _ensure_mjpython()
    print(
        f"[evaluate] model={args.model} task={task.name.value} "
        f"difficulty={task.difficulty} episodes={args.episodes}",
        file=sys.stderr,
    )
    summary = evaluate_policy(
        policy,
        args.robots,
        task,
        duration_s=args.duration,
        episodes=args.episodes,
        render=args.render,
        progress=lambda message: print(f"[evaluate] {message}", file=sys.stderr),
    )
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


def _default_model_path(algorithm: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return PROJECT_ROOT / "models" / algorithm / f"{timestamp}.json"


def _load_policy(
    path: Path, expected_task: str, parser: argparse.ArgumentParser
) -> LinearPolicy:
    if not path.is_file():
        parser.error(f"model file does not exist: {path}")
    try:
        policy = LinearPolicy.load(path)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(f"invalid model file {path}: {error}")
    if policy.metadata.task != expected_task:
        parser.error(f"model task is {policy.metadata.task!r}, not {expected_task!r}")
    return policy


def _ensure_mjpython() -> None:
    """Relaunch viewer commands through mjpython when macOS requires it."""

    if sys.platform != "darwin" or os.environ.get("BARISIMULATION_MJPYTHON") == "1":
        return
    launcher = shutil.which("mjpython")
    if launcher is None:
        raise SystemExit(
            "MuJoCo's macOS viewer requires mjpython in the active conda environment"
        )
    environment = os.environ.copy()
    environment["BARISIMULATION_MJPYTHON"] = "1"
    os.execvpe(launcher, [launcher, sys.argv[0], *sys.argv[1:]], environment)


if __name__ == "__main__":
    raise SystemExit(main())
