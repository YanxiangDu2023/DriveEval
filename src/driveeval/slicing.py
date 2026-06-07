from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from driveeval.schema import Scenario


SlicePredicate = Callable[[Scenario], bool]


def _condition(name: str, value: object) -> SlicePredicate:
    return lambda scenario: scenario.conditions.get(name) == value


def _speed(scenario: Scenario) -> float:
    xy = scenario.ego.xy
    timestamps = scenario.timestamps
    horizon = min(len(xy), len(timestamps))
    if horizon < 2:
        return 0.0
    distance = np.linalg.norm(np.diff(xy[:horizon], axis=0), axis=1).sum()
    duration = timestamps[horizon - 1] - timestamps[0]
    return float(distance / duration) if duration > 0 else 0.0


def _actor_count(scenario: Scenario) -> int:
    return max(0, len(scenario.tracks) - 1)


@dataclass(frozen=True)
class ScenarioSlice:
    name: str
    predicate: SlicePredicate


SCENARIO_SLICES: tuple[ScenarioSlice, ...] = (
    ScenarioSlice("urban", _condition("road_type", "urban")),
    ScenarioSlice("highway", _condition("road_type", "highway")),
    ScenarioSlice("intersection", _condition("junction", "intersection")),
    ScenarioSlice("merge", _condition("interaction", "merge")),
    ScenarioSlice("lane_change", _condition("interaction", "lane_change")),
    ScenarioSlice("unprotected_left", _condition("interaction", "unprotected_left")),
    ScenarioSlice("pedestrian_crossing", _condition("interaction", "pedestrian_crossing")),
    ScenarioSlice("cyclist_nearby", _condition("nearby_actor", "cyclist")),
    ScenarioSlice("cut_in", _condition("interaction", "cut_in")),
    ScenarioSlice("dense_traffic", lambda scenario: _actor_count(scenario) >= 6 or scenario.conditions.get("traffic") == "dense"),
    ScenarioSlice("sparse_traffic", lambda scenario: _actor_count(scenario) <= 2 or scenario.conditions.get("traffic") == "sparse"),
    ScenarioSlice("stop_and_go", _condition("traffic_flow", "stop_and_go")),
    ScenarioSlice("high_speed", lambda scenario: _speed(scenario) >= 18.0),
    ScenarioSlice("low_speed", lambda scenario: _speed(scenario) <= 5.0),
    ScenarioSlice("night", _condition("time_of_day", "night")),
    ScenarioSlice("rain", _condition("weather", "rain")),
    ScenarioSlice("off_route", _condition("route_state", "off_route")),
    ScenarioSlice("occluded_actor", lambda scenario: bool(scenario.conditions.get("occluded_actor", False))),
)


class ScenarioSlicer:
    def __init__(self, slices: tuple[ScenarioSlice, ...] = SCENARIO_SLICES) -> None:
        self.slices = slices

    def labels_for(self, scenario: Scenario) -> list[str]:
        return [item.name for item in self.slices if item.predicate(scenario)]

    def as_record(self, scenario: Scenario) -> dict[str, float]:
        labels = set(self.labels_for(scenario))
        return {f"slice_{item.name}": float(item.name in labels) for item in self.slices}

