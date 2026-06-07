from __future__ import annotations

import math

import numpy as np

from driveeval.schema import MapPolyline, Scenario, Trajectory


INTERACTIONS = [
    "merge",
    "lane_change",
    "unprotected_left",
    "pedestrian_crossing",
    "cut_in",
    "straight",
]


def _lane(y: float, lane_id: str) -> MapPolyline:
    x = np.linspace(-20.0, 110.0, 90)
    curve = 1.5 * np.sin(x / 34.0)
    return MapPolyline(polyline_id=lane_id, xy=np.column_stack([x, y + curve]), kind="lane_center")


def make_demo_scenarios(count: int = 72, seed: int = 7) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    timestamps = np.round(np.arange(0.0, 10.0, 0.2), 3)
    scenarios: list[Scenario] = []
    for index in range(count):
        road_type = "highway" if index % 5 == 0 else "urban"
        interaction = INTERACTIONS[index % len(INTERACTIONS)]
        time_of_day = "night" if index % 8 in {0, 1} else "day"
        weather = "rain" if index % 9 == 0 else "clear"
        traffic = "dense" if index % 4 == 0 else "sparse"
        nearby_actor = "cyclist" if index % 11 == 0 else "vehicle"
        traffic_flow = "stop_and_go" if index % 7 == 0 else "free_flow"
        route_state = "off_route" if index % 13 == 0 else "on_route"

        speed = 22.0 if road_type == "highway" else 8.0 + (index % 6)
        if traffic_flow == "stop_and_go":
            speed = 4.0
        t = timestamps
        x = speed * t + 0.4 * np.sin(t * 0.8 + index)
        y_base = 0.0 if index % 3 else 3.5
        y = y_base + 0.25 * np.sin(t * 0.9)
        if interaction == "lane_change":
            y += 3.5 / (1.0 + np.exp(-(t - 5.0) * 2.2))
        if interaction == "unprotected_left":
            x = 28.0 * np.sin(t / t[-1] * math.pi / 2)
            y = 28.0 * (1.0 - np.cos(t / t[-1] * math.pi / 2))
        ego_xy = np.column_stack([x, y])

        actor_count = 7 if traffic == "dense" else 2
        tracks = {
            "ego": Trajectory(track_id="ego", xy=ego_xy, object_type="vehicle"),
        }
        for actor_idx in range(actor_count):
            offset_x = 12.0 + actor_idx * 7.0 + rng.normal(0.0, 0.5)
            offset_y = (actor_idx % 3 - 1) * 3.5
            actor_speed = max(1.0, speed - 1.5 - actor_idx * 0.25)
            actor_xy = np.column_stack([actor_speed * t + offset_x, offset_y + 0.2 * np.sin(t + actor_idx)])
            if interaction == "cut_in" and actor_idx == 0:
                actor_xy[:, 1] = 6.0 - 6.0 / (1.0 + np.exp(-(t - 4.0) * 2.5))
            if interaction == "pedestrian_crossing" and actor_idx == 0:
                actor_xy = np.column_stack([np.full_like(t, 32.0), -8.0 + 2.0 * t])
            tracks[f"actor_{actor_idx}"] = Trajectory(
                track_id=f"actor_{actor_idx}",
                xy=actor_xy,
                object_type="pedestrian" if interaction == "pedestrian_crossing" and actor_idx == 0 else "vehicle",
            )

        if nearby_actor == "cyclist":
            cyclist_xy = np.column_stack([speed * t * 0.7 + 8.0, np.full_like(t, -3.5)])
            tracks["cyclist_0"] = Trajectory(track_id="cyclist_0", xy=cyclist_xy, object_type="cyclist")

        lanes = [_lane(-3.5, "lane-left"), _lane(0.0, "lane-center"), _lane(3.5, "lane-right")]
        route = lanes[1].xy.copy()
        scenario = Scenario(
            scenario_id=f"demo-{index:04d}",
            timestamps=timestamps,
            tracks=tracks,
            ego_track_id="ego",
            map_polylines=lanes,
            route=route,
            conditions={
                "road_type": road_type,
                "junction": "intersection" if index % 6 == 0 else "none",
                "interaction": interaction,
                "nearby_actor": nearby_actor,
                "traffic": traffic,
                "traffic_flow": traffic_flow,
                "time_of_day": time_of_day,
                "weather": weather,
                "route_state": route_state,
                "occluded_actor": index % 10 == 0,
            },
            metadata={"source": "demo", "index": index},
        )
        scenarios.append(scenario)
    return scenarios

