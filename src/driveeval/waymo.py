from __future__ import annotations

from pathlib import Path

import pandas as pd

from driveeval.io import load_tabular
from driveeval.schema import Scenario


WAYMO_EXPORT_COLUMNS = {
    "scenario_id",
    "track_id",
    "timestep",
    "x",
    "y",
}


def validate_waymo_motion_export(frame: pd.DataFrame) -> None:
    missing = WAYMO_EXPORT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Waymo Motion export missing columns: {sorted(missing)}")


def load_waymo_motion_export(path: str | Path) -> list[Scenario]:
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    validate_waymo_motion_export(frame)
    return load_tabular(source)

