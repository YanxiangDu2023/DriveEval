from __future__ import annotations

import math
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np

from driveeval.demo import make_demo_scenarios
from driveeval.metrics import EvaluationConfig
from driveeval.pipeline import evaluate_scenarios
from driveeval.planners import PLANNERS, PlannerContext, get_planner
from driveeval.schema import Scenario


PLANNER_LABELS = {
    "constant_velocity": "Constant velocity",
    "lane_keeping": "Lane keeping",
    "idm": "IDM style",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


class EvaluationRunStore:
    """Small in-memory store for local dashboard evaluation runs."""

    def __init__(self, maximum_runs: int = 10) -> None:
        self.maximum_runs = maximum_runs
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def create(
        self,
        *,
        scenario_count: int,
        seed: int,
        planners: list[str],
        observation_steps: int,
        horizon_steps: int,
        miss_threshold_m: float,
    ) -> dict[str, Any]:
        scenarios = make_demo_scenarios(scenario_count, seed)
        if observation_steps + horizon_steps > len(scenarios[0].timestamps):
            raise ValueError(
                "observation_steps + horizon_steps must not exceed the 50-frame demo scenario"
            )

        unknown = sorted(set(planners).difference(PLANNERS))
        if unknown:
            raise ValueError(f"unknown planners: {', '.join(unknown)}")

        config = EvaluationConfig(
            observation_steps=observation_steps,
            horizon_steps=horizon_steps,
            miss_threshold_m=miss_threshold_m,
        )
        context = PlannerContext(
            observation_steps=observation_steps,
            horizon_steps=horizon_steps,
        )

        metrics_by_planner: dict[str, list[dict[str, Any]]] = {}
        predictions: dict[str, dict[str, list[list[float]]]] = {
            scenario.scenario_id: {} for scenario in scenarios
        }
        summaries = []

        for planner_name in planners:
            planner = get_planner(planner_name)
            frame = evaluate_scenarios(scenarios, planner, config)
            records = _json_value(frame.to_dict(orient="records"))
            metrics_by_planner[planner_name] = records

            for scenario in scenarios:
                predictions[scenario.scenario_id][planner_name] = _json_value(
                    planner.plan(scenario, context)
                )

            summaries.append(
                {
                    "planner": planner_name,
                    "label": PLANNER_LABELS[planner_name],
                    "ade": _mean(records, "ade"),
                    "fde": _mean(records, "fde"),
                    "miss_rate": _mean(records, "miss"),
                    "collision_rate": _mean(records, "collision"),
                    "offroad_rate": _mean(records, "offroad_rate"),
                    "lane_deviation": _mean(records, "lane_deviation_mean"),
                    "jerk": _mean(records, "jerk_mean"),
                }
            )

        scenario_rows = []
        for scenario in scenarios:
            planner_metrics = {
                planner: next(
                    record
                    for record in metrics_by_planner[planner]
                    if record["scenario_id"] == scenario.scenario_id
                )
                for planner in planners
            }
            scenario_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "conditions": _json_value(scenario.conditions),
                    "actor_count": len(scenario.actors_except_ego()),
                    "metrics": planner_metrics,
                }
            )

        run_id = uuid4().hex[:12]
        run = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config": {
                "scenario_count": scenario_count,
                "seed": seed,
                "planners": planners,
                "observation_steps": observation_steps,
                "horizon_steps": horizon_steps,
                "miss_threshold_m": miss_threshold_m,
            },
            "summaries": summaries,
            "scenarios": scenario_rows,
            "_scenario_objects": {scenario.scenario_id: scenario for scenario in scenarios},
            "_predictions": predictions,
        }
        self._runs[run_id] = run
        while len(self._runs) > self.maximum_runs:
            self._runs.popitem(last=False)
        return self.public_run(run)

    def public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in run.items() if not key.startswith("_")}

    def get(self, run_id: str) -> dict[str, Any]:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"evaluation run not found: {run_id}") from exc

    def scenario(self, run_id: str, scenario_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        try:
            scenario: Scenario = run["_scenario_objects"][scenario_id]
        except KeyError as exc:
            raise KeyError(f"scenario not found: {scenario_id}") from exc

        return {
            "scenario_id": scenario.scenario_id,
            "timestamps": _json_value(scenario.timestamps),
            "observation_steps": run["config"]["observation_steps"],
            "horizon_steps": run["config"]["horizon_steps"],
            "conditions": _json_value(scenario.conditions),
            "tracks": [
                {
                    "track_id": track.track_id,
                    "object_type": track.object_type,
                    "xy": _json_value(track.xy),
                }
                for track in scenario.tracks.values()
            ],
            "map_polylines": [
                {
                    "polyline_id": polyline.polyline_id,
                    "kind": polyline.kind,
                    "xy": _json_value(polyline.xy),
                }
                for polyline in scenario.map_polylines
            ],
            "route": _json_value(scenario.route) if scenario.route is not None else None,
            "predictions": run["_predictions"][scenario_id],
            "metrics": next(
                row["metrics"] for row in run["scenarios"] if row["scenario_id"] == scenario_id
            ),
        }


RUN_STORE = EvaluationRunStore()

