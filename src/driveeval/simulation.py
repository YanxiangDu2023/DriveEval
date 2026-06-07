from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from driveeval.metrics import EvaluationConfig, curvature, jerk, nearest_polyline_distance, time_to_collision
from driveeval.planners import Planner, PlannerContext
from driveeval.schema import Scenario


@dataclass(frozen=True)
class RolloutConfig:
    observation_steps: int = 10
    horizon_steps: int = 30
    lane_half_width_m: float = 2.1
    collision_radius_m: float = 2.5
    completion_tolerance_m: float = 5.0


class KinematicSimulator:
    def rollout(self, scenario: Scenario, planner: Planner, config: RolloutConfig) -> dict[str, float | str]:
        context = PlannerContext(
            observation_steps=config.observation_steps,
            horizon_steps=config.horizon_steps,
        )
        planned = planner.plan(scenario, context)
        timestamps = scenario.timestamps[config.observation_steps : config.observation_steps + len(planned)]
        actor_futures = [
            actor.future(config.observation_steps, len(planned))
            for actor in scenario.actors_except_ego()
            if actor.length > config.observation_steps
        ]

        lane_distances = nearest_polyline_distance(planned, scenario.lane_centerlines)
        offroad = np.zeros(len(planned), dtype=bool) if np.all(np.isnan(lane_distances)) else lane_distances > config.lane_half_width_m
        ttc = time_to_collision(planned, actor_futures, timestamps, config.collision_radius_m)

        if scenario.route is not None and len(scenario.route):
            route_end = scenario.route[-1]
        else:
            route_end = scenario.ego.xy[min(config.observation_steps + len(planned) - 1, scenario.ego.length - 1)]
        distance_to_goal = float(np.linalg.norm(planned[-1] - route_end))
        nominal_distance = float(np.linalg.norm(scenario.ego.xy[config.observation_steps - 1] - route_end))
        route_completion = 1.0 - min(1.0, distance_to_goal / max(nominal_distance, config.completion_tolerance_m))

        jerk_metrics = jerk(planned, timestamps)
        curvature_metrics = curvature(planned)
        smoothness = 1.0 / (1.0 + jerk_metrics["jerk_mean"] + curvature_metrics["curvature_mean"])

        return {
            "scenario_id": scenario.scenario_id,
            "planner": planner.name,
            "collision_rate": float(np.isfinite(ttc)),
            "route_completion": float(route_completion),
            "intervention_rate": float(np.isfinite(ttc) or np.any(offroad)),
            "offroad_events": float(np.sum(np.diff(np.r_[False, offroad]).clip(min=0))),
            "motion_smoothness": float(smoothness),
        }


class CarlaRolloutAdapter:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout_s: float = 10.0) -> None:
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError("CARLA Python API is required for CarlaRolloutAdapter") from exc
        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)

    def world(self):
        return self.client.get_world()

    def rollout(self, scenario: Scenario, planner: Planner, config: RolloutConfig) -> dict[str, float | str]:
        raise NotImplementedError(
            "Connect this adapter to a CARLA map, actor spawning policy, and control bridge for the target simulator setup."
        )


def run_rollouts(
    scenarios: list[Scenario],
    planners: list[Planner],
    config: RolloutConfig | None = None,
) -> pd.DataFrame:
    config = config or RolloutConfig()
    simulator = KinematicSimulator()
    rows = [
        simulator.rollout(scenario, planner, config)
        for scenario in scenarios
        for planner in planners
        if scenario.ego.length > config.observation_steps
    ]
    return pd.DataFrame(rows)


def aggregate_rollouts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metrics = ["collision_rate", "route_completion", "intervention_rate", "offroad_events", "motion_smoothness"]
    return frame.groupby("planner", sort=True)[metrics].mean().reset_index()


def evaluation_to_rollout_config(config: EvaluationConfig) -> RolloutConfig:
    return RolloutConfig(
        observation_steps=config.observation_steps,
        horizon_steps=config.horizon_steps,
        lane_half_width_m=config.lane_half_width_m,
        collision_radius_m=config.collision_radius_m,
    )

