# DriveEval

DriveEval is an autonomous-driving evaluation and planner-simulation workbench for deterministic open-loop metrics, scenario slicing, long-tail failure analysis, and closed-loop rollout comparison.

The project is organized around three workflows:

- Open-loop evaluation over scenario records with displacement error, miss rate, time to collision, lane deviation, off-road events, jerk, and curvature.
- Scenario slicing across 18 driving conditions with deterministic failure clustering and nearest-neighbor retrieval.
- Closed-loop planner rollouts with constant-velocity, lane-keeping, and IDM-style baselines, plus an optional CARLA adapter.

## Capabilities

- Reads compact JSONL scenario files and tabular exports from motion-dataset preprocessing jobs.
- Includes Waymo Motion Dataset export validation for CSV and Parquet preprocessing outputs.
- Evaluates planner trajectories with NumPy and Pandas outputs that are stable across runs.
- Supports Plotly vector visualizations with map polylines, observed trajectories, planned trajectories, and similar-failure overlays.
- Provides optional OpenCV rendering for static frames and optional PyTorch acceleration for clustering.
- Includes a lightweight kinematic simulator for repeatable tests and CARLA-facing hooks for simulator-backed rollout jobs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional integrations:

```bash
pip install -e ".[torch,vision,sim]"
```

## Quick Start

Create deterministic demo scenarios:

```bash
driveeval make-demo-data --output data/demo_scenarios.jsonl --scenarios 72
```

Evaluate a planner:

```bash
driveeval evaluate \
  --input data/demo_scenarios.jsonl \
  --planner lane_keeping \
  --output reports/open_loop_metrics.csv
```

Cluster long-tail failures:

```bash
driveeval cluster \
  --metrics reports/open_loop_metrics.csv \
  --clusters 23 \
  --output reports/failure_clusters.csv
```

Create an interactive scene:

```bash
driveeval visualize \
  --input data/demo_scenarios.jsonl \
  --scenario-id demo-0007 \
  --planner idm \
  --output reports/demo-0007.html
```

Run closed-loop rollouts:

```bash
driveeval rollout \
  --input data/demo_scenarios.jsonl \
  --planner all \
  --output reports/rollout_metrics.csv
```

## Interactive Dashboard

DriveEval includes a local React dashboard backed by a FastAPI service. The dashboard can generate
deterministic demo scenarios, run all three planners, compare aggregate metrics, inspect individual
scenes, and play predicted trajectories against the recorded future.

Install the Python and frontend dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd frontend && npm install && npm run build && cd ..
```

Start the combined API and dashboard server:

```bash
source .venv/bin/activate
driveeval-api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is available at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

For frontend development with hot reload, run the API as above and start Vite in a second terminal:

```bash
cd frontend
npm run dev
```

Then open [http://127.0.0.1:5174](http://127.0.0.1:5174).

## Scenario Format

DriveEval uses newline-delimited JSON records:

```json
{
  "scenario_id": "demo-0001",
  "timestamps": [0.0, 0.2, 0.4],
  "ego_track_id": "ego",
  "conditions": {"road_type": "urban", "interaction": "merge"},
  "tracks": [
    {"track_id": "ego", "xy": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]}
  ],
  "map_polylines": [
    {"polyline_id": "lane-0", "kind": "lane_center", "xy": [[0.0, 0.0], [50.0, 0.0]]}
  ],
  "route": [[0.0, 0.0], [50.0, 0.0]]
}
```

Waymo Motion Dataset preprocessing jobs can export this schema directly after selecting tracks, map polylines, and scenario tags. Large corpora can be sharded into multiple JSONL files and evaluated independently before concatenating the metric tables.

## Development

```bash
pytest
python -m driveeval.cli --help
```
