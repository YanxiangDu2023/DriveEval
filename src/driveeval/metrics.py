from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from driveeval.schema import Scenario


def _align(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    horizon = min(len(predicted), len(target))
    if horizon == 0:
        raise ValueError("predicted and target trajectories must not be empty")
    return predicted[:horizon], target[:horizon]


def displacement_errors(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted, target = _align(predicted, target)
    distances = np.linalg.norm(predicted - target, axis=1)
    return {
        "ade": float(np.mean(distances)),
        "fde": float(distances[-1]),
        "max_de": float(np.max(distances)),
    }


def miss_rate(fde: float, threshold_m: float = 2.0) -> float:
    return float(fde > threshold_m)


def nearest_polyline_distance(points: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if not polylines:
        return np.full(len(points), np.nan)
    all_points = np.concatenate([np.asarray(polyline, dtype=float) for polyline in polylines if len(polyline)], axis=0)
    if len(all_points) == 0:
        return np.full(len(points), np.nan)
    delta = points[:, None, :] - all_points[None, :, :]
    return np.min(np.linalg.norm(delta, axis=2), axis=1)


def lane_deviation(points: np.ndarray, lane_centerlines: list[np.ndarray]) -> dict[str, float]:
    distances = nearest_polyline_distance(points, lane_centerlines)
    if np.all(np.isnan(distances)):
        return {"lane_deviation_mean": np.nan, "lane_deviation_max": np.nan}
    return {
        "lane_deviation_mean": float(np.nanmean(distances)),
        "lane_deviation_max": float(np.nanmax(distances)),
    }


def offroad_events(points: np.ndarray, lane_centerlines: list[np.ndarray], lane_half_width_m: float = 2.1) -> dict[str, float]:
    distances = nearest_polyline_distance(points, lane_centerlines)
    if np.all(np.isnan(distances)):
        return {"offroad_events": 0.0, "offroad_rate": 0.0}
    offroad = distances > lane_half_width_m
    transitions = np.diff(np.r_[False, offroad.astype(bool)]).clip(min=0)
    return {
        "offroad_events": float(np.sum(transitions)),
        "offroad_rate": float(np.mean(offroad)),
    }


def time_to_collision(
    ego_xy: np.ndarray,
    actor_trajectories: list[np.ndarray],
    timestamps: np.ndarray,
    collision_radius_m: float = 2.5,
) -> float:
    ego_xy = np.asarray(ego_xy, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    best = np.inf
    for actor_xy in actor_trajectories:
        actor_xy = np.asarray(actor_xy, dtype=float)
        horizon = min(len(ego_xy), len(actor_xy), len(timestamps))
        if horizon == 0:
            continue
        distances = np.linalg.norm(ego_xy[:horizon] - actor_xy[:horizon], axis=1)
        hits = np.flatnonzero(distances <= collision_radius_m)
        if len(hits):
            best = min(best, float(timestamps[hits[0]] - timestamps[0]))
    return best


def jerk(points: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    points = np.asarray(points, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    horizon = min(len(points), len(timestamps))
    points = points[:horizon]
    timestamps = timestamps[:horizon]
    if horizon < 4:
        return {"jerk_mean": 0.0, "jerk_max": 0.0}
    velocity = np.gradient(points, timestamps, axis=0)
    acceleration = np.gradient(velocity, timestamps, axis=0)
    jerk_values = np.gradient(acceleration, timestamps, axis=0)
    magnitudes = np.linalg.norm(jerk_values, axis=1)
    return {
        "jerk_mean": float(np.mean(magnitudes)),
        "jerk_max": float(np.max(magnitudes)),
    }


def curvature(points: np.ndarray) -> dict[str, float]:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return {"curvature_mean": 0.0, "curvature_max": 0.0}
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)
    return {
        "curvature_mean": float(np.mean(values)),
        "curvature_max": float(np.max(values)),
    }


@dataclass(frozen=True)
class EvaluationConfig:
    observation_steps: int = 10
    horizon_steps: int = 30
    miss_threshold_m: float = 2.0
    lane_half_width_m: float = 2.1
    collision_radius_m: float = 2.5


def evaluate_prediction(
    scenario: Scenario,
    predicted_xy: np.ndarray,
    config: EvaluationConfig | None = None,
) -> dict[str, float | str]:
    config = config or EvaluationConfig()
    target = scenario.ego.future(config.observation_steps, config.horizon_steps)
    predicted_xy, target = _align(predicted_xy, target)
    timestamps = scenario.timestamps[config.observation_steps : config.observation_steps + len(target)]

    result: dict[str, float | str] = {
        "scenario_id": scenario.scenario_id,
    }
    result.update(displacement_errors(predicted_xy, target))
    result["miss"] = miss_rate(float(result["fde"]), config.miss_threshold_m)

    actor_futures = [
        actor.future(config.observation_steps, len(target))
        for actor in scenario.actors_except_ego()
        if actor.length > config.observation_steps
    ]
    ttc = time_to_collision(predicted_xy, actor_futures, timestamps, config.collision_radius_m)
    result["time_to_collision"] = float(ttc)
    result["collision"] = float(np.isfinite(ttc))
    result.update(lane_deviation(predicted_xy, scenario.lane_centerlines))
    result.update(offroad_events(predicted_xy, scenario.lane_centerlines, config.lane_half_width_m))
    result.update(jerk(predicted_xy, timestamps))
    result.update(curvature(predicted_xy))
    return result

