from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from driveeval.schema import MapPolyline, Scenario, Trajectory


def load_scenarios(path: str | Path) -> list[Scenario]:
    source = Path(path)
    if source.is_dir():
        scenarios: list[Scenario] = []
        for child in sorted(source.glob("*.jsonl")):
            scenarios.extend(load_jsonl(child))
        for child in sorted(source.glob("*.json")):
            scenarios.extend(load_json(child))
        return scenarios

    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(source)
    if suffix == ".json":
        return load_json(source)
    if suffix in {".csv", ".parquet"}:
        return load_tabular(source)
    raise ValueError(f"unsupported scenario input: {source}")


def load_jsonl(path: str | Path) -> list[Scenario]:
    scenarios = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                scenarios.append(Scenario.from_dict(json.loads(line)))
    return scenarios


def load_json(path: str | Path) -> list[Scenario]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "scenarios" in data:
            data = data["scenarios"]
        else:
            data = [data]
    return [Scenario.from_dict(item) for item in data]


def write_jsonl(scenarios: Iterable[Scenario], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario.to_dict(), separators=(",", ":")) + "\n")


def load_tabular(path: str | Path) -> list[Scenario]:
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    required = {"scenario_id", "track_id", "timestep", "x", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"tabular input missing columns: {sorted(missing)}")

    scenarios: list[Scenario] = []
    for scenario_id, group in frame.groupby("scenario_id", sort=True):
        timestamps = (
            group[["timestep"]]
            .drop_duplicates()
            .sort_values("timestep")["timestep"]
            .astype(float)
            .to_numpy()
        )
        tracks: dict[str, Trajectory] = {}
        for track_id, track_rows in group.groupby("track_id", sort=True):
            track_rows = track_rows.sort_values("timestep")
            tracks[str(track_id)] = Trajectory(
                track_id=str(track_id),
                xy=track_rows[["x", "y"]].to_numpy(dtype=float),
                object_type=str(track_rows["object_type"].iloc[0]) if "object_type" in track_rows else "vehicle",
            )

        map_polylines: list[MapPolyline] = []
        if {"polyline_id", "polyline_x", "polyline_y"}.issubset(frame.columns):
            for polyline_id, rows in group.dropna(subset=["polyline_id"]).groupby("polyline_id", sort=True):
                rows = rows.drop_duplicates(["polyline_x", "polyline_y"])
                map_polylines.append(
                    MapPolyline(
                        polyline_id=str(polyline_id),
                        xy=rows[["polyline_x", "polyline_y"]].to_numpy(dtype=float),
                        kind=str(rows["polyline_kind"].iloc[0]) if "polyline_kind" in rows else "lane_center",
                    )
                )

        condition_cols = [col for col in group.columns if col.startswith("condition_")]
        conditions = {
            col.removeprefix("condition_"): group[col].iloc[0]
            for col in condition_cols
            if pd.notna(group[col].iloc[0])
        }
        ego_track_id = str(group["ego_track_id"].iloc[0]) if "ego_track_id" in group else "ego"
        scenarios.append(
            Scenario(
                scenario_id=str(scenario_id),
                timestamps=timestamps,
                tracks=tracks,
                ego_track_id=ego_track_id,
                map_polylines=map_polylines,
                conditions=conditions,
            )
        )
    return scenarios

