from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from driveeval.schema import Scenario


def scenario_figure(
    scenario: Scenario,
    predicted_xy: np.ndarray | None = None,
    similar_scenarios: list[Scenario] | None = None,
) -> go.Figure:
    figure = go.Figure()

    for polyline in scenario.map_polylines:
        figure.add_trace(
            go.Scatter(
                x=polyline.xy[:, 0],
                y=polyline.xy[:, 1],
                mode="lines",
                name=f"{polyline.kind}:{polyline.polyline_id}",
                line={"color": "#8aa0b6", "width": 1},
                hoverinfo="skip",
            )
        )

    for track in scenario.tracks.values():
        color = "#1f77b4" if track.track_id == scenario.ego_track_id else "#59656f"
        width = 4 if track.track_id == scenario.ego_track_id else 2
        figure.add_trace(
            go.Scatter(
                x=track.xy[:, 0],
                y=track.xy[:, 1],
                mode="lines+markers",
                name=f"track:{track.track_id}",
                line={"color": color, "width": width},
                marker={"size": 4},
            )
        )

    if predicted_xy is not None:
        predicted_xy = np.asarray(predicted_xy, dtype=float)
        figure.add_trace(
            go.Scatter(
                x=predicted_xy[:, 0],
                y=predicted_xy[:, 1],
                mode="lines+markers",
                name="planned",
                line={"color": "#d62728", "width": 4, "dash": "dash"},
                marker={"size": 5},
            )
        )

    for similar in similar_scenarios or []:
        offset = scenario.ego.xy[0] - similar.ego.xy[0]
        xy = similar.ego.xy + offset
        figure.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines",
                name=f"similar:{similar.scenario_id}",
                line={"color": "#2ca02c", "width": 2, "dash": "dot"},
                opacity=0.55,
            )
        )

    figure.update_layout(
        title=f"Scenario {scenario.scenario_id}",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 40, "r": 20, "t": 60, "b": 80},
    )
    return figure


def save_scenario_html(
    scenario: Scenario,
    output_path: str | Path,
    predicted_xy: np.ndarray | None = None,
    similar_scenarios: list[Scenario] | None = None,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = scenario_figure(scenario, predicted_xy=predicted_xy, similar_scenarios=similar_scenarios)
    figure.write_html(target, include_plotlyjs="cdn")


def render_static_png(
    scenario: Scenario,
    output_path: str | Path,
    predicted_xy: np.ndarray | None = None,
    pixels: int = 900,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for PNG rendering") from exc

    all_xy = [track.xy for track in scenario.tracks.values()]
    all_xy.extend(polyline.xy for polyline in scenario.map_polylines)
    if predicted_xy is not None:
        all_xy.append(np.asarray(predicted_xy, dtype=float))
    stacked = np.concatenate(all_xy, axis=0)
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    pad = 0.1 * span
    minimum -= pad
    maximum += pad
    span = maximum - minimum

    def project(xy: np.ndarray) -> np.ndarray:
        xy = (xy - minimum) / span
        xy[:, 1] = 1.0 - xy[:, 1]
        return np.round(xy * (pixels - 1)).astype(int)

    canvas = np.full((pixels, pixels, 3), 255, dtype=np.uint8)
    for polyline in scenario.map_polylines:
        pts = project(polyline.xy.copy())
        cv2.polylines(canvas, [pts], False, (190, 180, 150), 1, cv2.LINE_AA)
    for track in scenario.tracks.values():
        pts = project(track.xy.copy())
        color = (180, 90, 40) if track.track_id == scenario.ego_track_id else (120, 120, 120)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    if predicted_xy is not None:
        pts = project(np.asarray(predicted_xy, dtype=float).copy())
        cv2.polylines(canvas, [pts], False, (40, 40, 210), 3, cv2.LINE_AA)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), canvas)

