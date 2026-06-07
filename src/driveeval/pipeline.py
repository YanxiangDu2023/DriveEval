from __future__ import annotations

import pandas as pd

from driveeval.metrics import EvaluationConfig, evaluate_prediction
from driveeval.planners import Planner, PlannerContext
from driveeval.schema import Scenario
from driveeval.slicing import ScenarioSlicer


def evaluate_scenarios(
    scenarios: list[Scenario],
    planner: Planner,
    config: EvaluationConfig | None = None,
    slicer: ScenarioSlicer | None = None,
) -> pd.DataFrame:
    config = config or EvaluationConfig()
    slicer = slicer or ScenarioSlicer()
    context = PlannerContext(
        observation_steps=config.observation_steps,
        horizon_steps=config.horizon_steps,
    )
    rows = []
    for scenario in scenarios:
        if scenario.ego.length <= config.observation_steps:
            continue
        planned = planner.plan(scenario, context)
        row = evaluate_prediction(scenario, planned, config)
        row["planner"] = planner.name
        row.update(slicer.as_record(scenario))
        rows.append(row)
    return pd.DataFrame(rows)

