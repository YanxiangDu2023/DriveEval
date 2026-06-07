from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from driveeval.metrics import nearest_polyline_distance
from driveeval.schema import Scenario


def _step_dt(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.2
    diffs = np.diff(timestamps[: min(len(timestamps), 8)])
    return float(np.median(diffs))


def _last_velocity(xy: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return np.array([0.0, 0.0])
    dt = max(float(timestamps[-1] - timestamps[-2]), 1e-3)
    return (xy[-1] - xy[-2]) / dt


@dataclass(frozen=True)
class PlannerContext:
    observation_steps: int = 10
    horizon_steps: int = 30


class Planner(ABC):
    name: str

    @abstractmethod
    def plan(self, scenario: Scenario, context: PlannerContext) -> np.ndarray:
        raise NotImplementedError


class ConstantVelocityPlanner(Planner):
    name = "constant_velocity"

    def plan(self, scenario: Scenario, context: PlannerContext) -> np.ndarray:
        observed_xy = scenario.ego.xy[: context.observation_steps]
        observed_t = scenario.timestamps[: context.observation_steps]
        velocity = _last_velocity(observed_xy, observed_t)
        dt = _step_dt(scenario.timestamps)
        steps = np.arange(1, context.horizon_steps + 1, dtype=float)[:, None]
        return observed_xy[-1] + velocity * dt * steps


class LaneKeepingPlanner(Planner):
    name = "lane_keeping"

    def plan(self, scenario: Scenario, context: PlannerContext) -> np.ndarray:
        observed_xy = scenario.ego.xy[: context.observation_steps]
        observed_t = scenario.timestamps[: context.observation_steps]
        lane_centerlines = scenario.lane_centerlines
        if not lane_centerlines:
            return ConstantVelocityPlanner().plan(scenario, context)

        distances = [nearest_polyline_distance(observed_xy[-1:], [line])[0] for line in lane_centerlines]
        lane = lane_centerlines[int(np.nanargmin(distances))]
        lane_steps = np.linalg.norm(np.diff(lane, axis=0), axis=1)
        lane_s = np.r_[0.0, np.cumsum(lane_steps)]
        nearest = int(np.argmin(np.linalg.norm(lane - observed_xy[-1], axis=1)))
        speed = float(np.linalg.norm(_last_velocity(observed_xy, observed_t)))
        dt = _step_dt(scenario.timestamps)
        query_s = lane_s[nearest] + np.arange(1, context.horizon_steps + 1) * speed * dt
        x = np.interp(query_s, lane_s, lane[:, 0])
        y = np.interp(query_s, lane_s, lane[:, 1])
        return np.column_stack([x, y])


class IDMPlanner(Planner):
    name = "idm"

    def __init__(self, desired_speed_mps: float = 16.0, min_gap_m: float = 5.0) -> None:
        self.desired_speed_mps = desired_speed_mps
        self.min_gap_m = min_gap_m

    def plan(self, scenario: Scenario, context: PlannerContext) -> np.ndarray:
        observed_xy = scenario.ego.xy[: context.observation_steps]
        observed_t = scenario.timestamps[: context.observation_steps]
        dt = _step_dt(scenario.timestamps)
        velocity = _last_velocity(observed_xy, observed_t)
        speed = float(np.linalg.norm(velocity))
        heading = velocity / speed if speed > 1e-6 else np.array([1.0, 0.0])
        position = observed_xy[-1].astype(float).copy()
        planned = []

        for _ in range(context.horizon_steps):
            lead_gap = np.inf
            for actor in scenario.actors_except_ego():
                if actor.length <= context.observation_steps:
                    continue
                rel = actor.xy[context.observation_steps - 1] - position
                along = float(np.dot(rel, heading))
                lateral = float(np.linalg.norm(rel - along * heading))
                if along > 0 and lateral < 3.0:
                    lead_gap = min(lead_gap, along)

            gap_factor = 1.0
            if lead_gap < np.inf:
                gap_factor = max(0.0, min(1.0, (lead_gap - self.min_gap_m) / max(lead_gap, 1.0)))
            target_speed = self.desired_speed_mps * gap_factor
            speed += np.clip(target_speed - speed, -2.5 * dt, 1.5 * dt)
            position = position + heading * speed * dt
            planned.append(position.copy())

        raw_plan = np.asarray(planned)
        if scenario.lane_centerlines:
            lane_plan = LaneKeepingPlanner().plan(scenario, context)
            blend = np.linspace(0.2, 0.8, context.horizon_steps)[:, None]
            return raw_plan * (1.0 - blend) + lane_plan * blend
        return raw_plan


PLANNERS: dict[str, Planner] = {
    ConstantVelocityPlanner.name: ConstantVelocityPlanner(),
    LaneKeepingPlanner.name: LaneKeepingPlanner(),
    IDMPlanner.name: IDMPlanner(),
}


def get_planner(name: str) -> Planner:
    try:
        return PLANNERS[name]
    except KeyError as exc:
        options = ", ".join(sorted(PLANNERS))
        raise ValueError(f"unknown planner '{name}', expected one of: {options}") from exc

