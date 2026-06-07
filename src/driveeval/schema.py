from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _xy_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    return array


@dataclass(frozen=True)
class Trajectory:
    track_id: str
    xy: np.ndarray
    yaw: np.ndarray | None = None
    valid: np.ndarray | None = None
    object_type: str = "vehicle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "xy", _xy_array(self.xy, name="trajectory xy"))
        if self.yaw is not None:
            yaw = np.asarray(self.yaw, dtype=float)
            if yaw.shape != (len(self.xy),):
                raise ValueError("yaw must have shape (N,)")
            object.__setattr__(self, "yaw", yaw)
        if self.valid is None:
            object.__setattr__(self, "valid", np.ones(len(self.xy), dtype=bool))
        else:
            valid = np.asarray(self.valid, dtype=bool)
            if valid.shape != (len(self.xy),):
                raise ValueError("valid must have shape (N,)")
            object.__setattr__(self, "valid", valid)

    @property
    def length(self) -> int:
        return len(self.xy)

    def observed(self, steps: int) -> "Trajectory":
        return Trajectory(
            track_id=self.track_id,
            xy=self.xy[:steps],
            yaw=None if self.yaw is None else self.yaw[:steps],
            valid=None if self.valid is None else self.valid[:steps],
            object_type=self.object_type,
        )

    def future(self, start: int, horizon: int | None = None) -> np.ndarray:
        stop = None if horizon is None else start + horizon
        return self.xy[start:stop]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        return cls(
            track_id=str(data["track_id"]),
            xy=data["xy"],
            yaw=data.get("yaw"),
            valid=data.get("valid"),
            object_type=data.get("object_type", "vehicle"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "track_id": self.track_id,
            "xy": self.xy.tolist(),
            "object_type": self.object_type,
        }
        if self.yaw is not None:
            payload["yaw"] = self.yaw.tolist()
        if self.valid is not None:
            payload["valid"] = self.valid.astype(bool).tolist()
        return payload


@dataclass(frozen=True)
class MapPolyline:
    polyline_id: str
    xy: np.ndarray
    kind: str = "lane_center"
    speed_limit_mps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "xy", _xy_array(self.xy, name="map polyline xy"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapPolyline":
        return cls(
            polyline_id=str(data.get("polyline_id", data.get("id", "polyline"))),
            xy=data["xy"],
            kind=data.get("kind", "lane_center"),
            speed_limit_mps=data.get("speed_limit_mps"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "polyline_id": self.polyline_id,
            "kind": self.kind,
            "xy": self.xy.tolist(),
        }
        if self.speed_limit_mps is not None:
            payload["speed_limit_mps"] = self.speed_limit_mps
        return payload


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    timestamps: np.ndarray
    tracks: dict[str, Trajectory]
    ego_track_id: str = "ego"
    map_polylines: list[MapPolyline] = field(default_factory=list)
    route: np.ndarray | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must have shape (N,)")
        object.__setattr__(self, "timestamps", timestamps)
        if self.ego_track_id not in self.tracks:
            raise KeyError(f"ego track '{self.ego_track_id}' is not present")
        if self.route is not None:
            object.__setattr__(self, "route", _xy_array(self.route, name="route"))

    @property
    def ego(self) -> Trajectory:
        return self.tracks[self.ego_track_id]

    @property
    def lane_centerlines(self) -> list[np.ndarray]:
        return [p.xy for p in self.map_polylines if p.kind == "lane_center"]

    def actors_except_ego(self) -> list[Trajectory]:
        return [track for key, track in self.tracks.items() if key != self.ego_track_id]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        tracks = {track["track_id"]: Trajectory.from_dict(track) for track in data["tracks"]}
        return cls(
            scenario_id=str(data["scenario_id"]),
            timestamps=np.asarray(data["timestamps"], dtype=float),
            tracks=tracks,
            ego_track_id=data.get("ego_track_id", "ego"),
            map_polylines=[MapPolyline.from_dict(item) for item in data.get("map_polylines", [])],
            route=data.get("route"),
            conditions=dict(data.get("conditions", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "timestamps": self.timestamps.tolist(),
            "ego_track_id": self.ego_track_id,
            "conditions": self.conditions,
            "metadata": self.metadata,
            "tracks": [track.to_dict() for track in self.tracks.values()],
            "map_polylines": [polyline.to_dict() for polyline in self.map_polylines],
        }
        if self.route is not None:
            payload["route"] = self.route.tolist()
        return payload

