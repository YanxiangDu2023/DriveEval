from pathlib import Path

import pandas as pd

from driveeval.clustering import FailureClusterer, find_similar_failures
from driveeval.demo import make_demo_scenarios
from driveeval.io import load_scenarios, write_jsonl
from driveeval.metrics import EvaluationConfig
from driveeval.pipeline import evaluate_scenarios
from driveeval.planners import get_planner
from driveeval.simulation import aggregate_rollouts, run_rollouts
from driveeval.slicing import SCENARIO_SLICES, ScenarioSlicer
from driveeval.waymo import validate_waymo_motion_export


def test_demo_io_round_trip(tmp_path: Path) -> None:
    scenarios = make_demo_scenarios(5)
    path = tmp_path / "demo.jsonl"

    write_jsonl(scenarios, path)
    loaded = load_scenarios(path)

    assert [item.scenario_id for item in loaded] == [item.scenario_id for item in scenarios]
    assert loaded[0].ego.xy.shape[1] == 2


def test_pipeline_adds_all_slice_columns() -> None:
    scenarios = make_demo_scenarios(12)
    frame = evaluate_scenarios(
        scenarios,
        get_planner("lane_keeping"),
        EvaluationConfig(observation_steps=10, horizon_steps=20),
        ScenarioSlicer(),
    )

    assert len(SCENARIO_SLICES) == 18
    assert len(frame) == 12
    assert all(f"slice_{item.name}" in frame.columns for item in SCENARIO_SLICES)
    assert {"ade", "fde", "miss", "time_to_collision", "jerk_mean", "curvature_mean"}.issubset(frame.columns)


def test_failure_clustering_and_retrieval_are_repeatable() -> None:
    scenarios = make_demo_scenarios(30)
    frame = evaluate_scenarios(scenarios, get_planner("constant_velocity"), EvaluationConfig(horizon_steps=20))

    clustered_a = FailureClusterer(n_clusters=6).fit_predict(frame)
    clustered_b = FailureClusterer(n_clusters=6).fit_predict(frame)

    assert clustered_a["failure_cluster"].tolist() == clustered_b["failure_cluster"].tolist()
    assert clustered_a["failure_cluster"].nunique() <= 6
    similar = find_similar_failures(frame, frame["scenario_id"].iloc[0], k=3)
    assert len(similar) == 3
    assert frame["scenario_id"].iloc[0] not in set(similar["scenario_id"])


def test_rollout_metrics_have_expected_columns() -> None:
    scenarios = make_demo_scenarios(4)
    frame = run_rollouts(scenarios, [get_planner("constant_velocity"), get_planner("idm")])
    summary = aggregate_rollouts(frame)

    assert len(frame) == 8
    assert set(frame["planner"]) == {"constant_velocity", "idm"}
    assert {"collision_rate", "route_completion", "intervention_rate", "offroad_events", "motion_smoothness"}.issubset(frame.columns)
    assert isinstance(summary, pd.DataFrame)


def test_waymo_export_validation_requires_core_columns() -> None:
    frame = pd.DataFrame(
        {
            "scenario_id": ["s0"],
            "track_id": ["ego"],
            "timestep": [0],
            "x": [0.0],
            "y": [0.0],
        }
    )

    validate_waymo_motion_export(frame)
