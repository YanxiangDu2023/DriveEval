from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from driveeval.clustering import FailureClusterer
from driveeval.demo import make_demo_scenarios
from driveeval.io import load_scenarios, write_jsonl
from driveeval.metrics import EvaluationConfig
from driveeval.pipeline import evaluate_scenarios
from driveeval.planners import PLANNERS, PlannerContext, get_planner
from driveeval.simulation import RolloutConfig, aggregate_rollouts, run_rollouts
from driveeval.visualization import save_scenario_html


def _evaluation_config(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        observation_steps=args.observation_steps,
        horizon_steps=args.horizon_steps,
        miss_threshold_m=args.miss_threshold_m,
    )


def command_make_demo_data(args: argparse.Namespace) -> None:
    scenarios = make_demo_scenarios(args.scenarios, args.seed)
    write_jsonl(scenarios, args.output)
    print(f"wrote {len(scenarios)} scenarios to {args.output}")


def command_evaluate(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.input)
    planner = get_planner(args.planner)
    frame = evaluate_scenarios(scenarios, planner, _evaluation_config(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} metric rows to {output}")


def command_cluster(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.metrics)
    clusterer = FailureClusterer(n_clusters=args.clusters, backend=args.backend)
    clustered = clusterer.fit_predict(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(output, index=False)
    summary_path = output.with_name(output.stem + "_summary.csv")
    clusterer.summarize(clustered).to_csv(summary_path, index=False)
    print(f"wrote {len(clustered)} clustered rows to {output}")
    print(f"wrote cluster summary to {summary_path}")


def command_visualize(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.input)
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if args.scenario_id not in by_id:
        raise SystemExit(f"scenario_id not found: {args.scenario_id}")
    scenario = by_id[args.scenario_id]
    planner = get_planner(args.planner)
    config = _evaluation_config(args)
    planned = planner.plan(
        scenario,
        context=PlannerContext(
            observation_steps=config.observation_steps,
            horizon_steps=config.horizon_steps,
        ),
    )
    save_scenario_html(scenario, args.output, predicted_xy=planned)
    print(f"wrote interactive visualization to {args.output}")


def command_rollout(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.input)
    if args.planner == "all":
        planners = list(PLANNERS.values())
    else:
        planners = [get_planner(args.planner)]
    config = RolloutConfig(observation_steps=args.observation_steps, horizon_steps=args.horizon_steps)
    frame = run_rollouts(scenarios, planners, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    aggregate_path = output.with_name(output.stem + "_summary.csv")
    aggregate_rollouts(frame).to_csv(aggregate_path, index=False)
    print(f"wrote {len(frame)} rollout rows to {output}")
    print(f"wrote rollout summary to {aggregate_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driveeval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo-data", help="write deterministic scenario records")
    demo.add_argument("--output", required=True)
    demo.add_argument("--scenarios", type=int, default=72)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=command_make_demo_data)

    evaluate = subparsers.add_parser("evaluate", help="run open-loop planner metrics")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--planner", choices=sorted(PLANNERS), default="constant_velocity")
    evaluate.add_argument("--observation-steps", type=int, default=10)
    evaluate.add_argument("--horizon-steps", type=int, default=30)
    evaluate.add_argument("--miss-threshold-m", type=float, default=2.0)
    evaluate.set_defaults(func=command_evaluate)

    cluster = subparsers.add_parser("cluster", help="cluster long-tail failure rows")
    cluster.add_argument("--metrics", required=True)
    cluster.add_argument("--output", required=True)
    cluster.add_argument("--clusters", type=int, default=23)
    cluster.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    cluster.set_defaults(func=command_cluster)

    visualize = subparsers.add_parser("visualize", help="write a Plotly scenario visualization")
    visualize.add_argument("--input", required=True)
    visualize.add_argument("--scenario-id", required=True)
    visualize.add_argument("--output", required=True)
    visualize.add_argument("--planner", choices=sorted(PLANNERS), default="lane_keeping")
    visualize.add_argument("--observation-steps", type=int, default=10)
    visualize.add_argument("--horizon-steps", type=int, default=30)
    visualize.add_argument("--miss-threshold-m", type=float, default=2.0)
    visualize.set_defaults(func=command_visualize)

    rollout = subparsers.add_parser("rollout", help="run closed-loop planner rollouts")
    rollout.add_argument("--input", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--planner", choices=["all", *sorted(PLANNERS)], default="all")
    rollout.add_argument("--observation-steps", type=int, default=10)
    rollout.add_argument("--horizon-steps", type=int, default=30)
    rollout.set_defaults(func=command_rollout)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
