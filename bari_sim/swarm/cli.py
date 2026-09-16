"""Command-line entry point for planar swarm experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .controllers import LinearLocalController, make_controller_factory
from .core import run_episode
from .experiments import (
    SCENARIOS,
    benchmark,
    scaling_benchmark,
    summarize,
    train_aggregation_cem,
    write_episode_csv,
    write_summary_csv,
    write_training_history,
)
from .tasks import TASKS, make_task

DEFAULT_POLICY_PATH = Path("models/swarm/aggregation_linear.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeatable decentralized planar swarm experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one episode")
    run_parser.add_argument("--task", choices=sorted(TASKS), required=True)
    run_parser.add_argument(
        "--controller",
        choices=("rules", "random", "stationary", "learned"),
        default="rules",
    )
    run_parser.add_argument("--robots", type=int)
    run_parser.add_argument("--duration", type=float)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="compare controllers over matched seeds"
    )
    benchmark_parser.add_argument(
        "--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS)
    )
    benchmark_parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("rules", "random", "stationary", "learned"),
        default=("rules", "random", "stationary"),
    )
    benchmark_parser.add_argument("--seeds", type=int, default=10)
    benchmark_parser.add_argument("--seed-start", type=int, default=1000)
    benchmark_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    benchmark_parser.add_argument(
        "--output", type=Path, default=Path("logs/swarm_benchmark.csv")
    )

    sweep_parser = subparsers.add_parser(
        "sweep", help="compare one controller across robot counts"
    )
    sweep_parser.add_argument("--task", choices=sorted(TASKS), required=True)
    sweep_parser.add_argument(
        "--controller",
        choices=("rules", "random", "stationary", "learned"),
        default="rules",
    )
    sweep_parser.add_argument("--robot-counts", nargs="+", type=int, required=True)
    sweep_parser.add_argument("--duration", type=float)
    sweep_parser.add_argument("--seeds", type=int, default=10)
    sweep_parser.add_argument("--seed-start", type=int, default=1000)
    sweep_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    sweep_parser.add_argument(
        "--output", type=Path, default=Path("logs/swarm_scaling.csv")
    )

    train_parser = subparsers.add_parser(
        "train", help="train the lightweight aggregation policy"
    )
    train_parser.add_argument("--generations", type=int, default=12)
    train_parser.add_argument("--population", type=int, default=24)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    train_parser.add_argument(
        "--history", type=Path, default=Path("logs/swarm_training.csv")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "sweep":
        return _sweep(args)
    if args.command == "train":
        return _train(args)
    raise AssertionError("unreachable command")


def _load_parameters(path: Path):
    return LinearLocalController.load(path).parameters


def _run(args: argparse.Namespace) -> int:
    scenario = SCENARIOS[args.task]
    parameters = _load_parameters(args.policy) if args.controller == "learned" else None
    factory = make_controller_factory(
        args.task, args.controller, learned_parameters=parameters
    )
    result = run_episode(
        make_task(args.task),
        factory,
        controller_name=args.controller,
        robot_count=args.robots or scenario.robot_count,
        seed=args.seed,
        duration_s=args.duration or scenario.duration_s,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    parameters = (
        _load_parameters(args.policy) if "learned" in args.controllers else None
    )
    results = benchmark(
        tasks=args.tasks,
        controllers=args.controllers,
        seeds=range(args.seed_start, args.seed_start + args.seeds),
        learned_parameters=parameters,
    )
    rows = summarize(results)
    write_episode_csv(args.output, results)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    write_summary_csv(summary_path, rows)
    print(json.dumps(rows, indent=2))
    print(f"episodes: {args.output}")
    print(f"summary: {summary_path}")
    return 0


def _train(args: argparse.Namespace) -> int:
    result = train_aggregation_cem(
        seed=args.seed,
        generations=args.generations,
        population_size=args.population,
    )
    controller = LinearLocalController(result.parameters)
    controller.save(
        args.policy,
        metadata={
            "task": "aggregation",
            "training_seed": args.seed,
            "generations": args.generations,
            "population_size": args.population,
            "best_training_score": result.best_score,
        },
    )
    write_training_history(args.history, result.history)
    print(f"best training score: {result.best_score:.6f}")
    print(f"policy: {args.policy}")
    print(f"history: {args.history}")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    parameters = _load_parameters(args.policy) if args.controller == "learned" else None
    results = scaling_benchmark(
        task_name=args.task,
        controller_name=args.controller,
        robot_counts=args.robot_counts,
        seeds=range(args.seed_start, args.seed_start + args.seeds),
        duration_s=args.duration,
        learned_parameters=parameters,
    )
    rows = summarize(results, by_robot_count=True)
    write_episode_csv(args.output, results)
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    write_summary_csv(summary_path, rows)
    print(json.dumps(rows, indent=2))
    print(f"episodes: {args.output}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
