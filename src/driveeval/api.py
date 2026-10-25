from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from driveeval.planners import PLANNERS
from driveeval.web_service import PLANNER_LABELS, RUN_STORE


class RunRequest(BaseModel):
    scenario_count: int = Field(default=24, ge=1, le=200)
    seed: int = 7
    planners: list[str] = Field(
        default_factory=lambda: ["constant_velocity", "lane_keeping", "idm"]
    )
    observation_steps: int = Field(default=10, ge=2, le=40)
    horizon_steps: int = Field(default=30, ge=1, le=40)
    miss_threshold_m: float = Field(default=2.0, gt=0, le=20)


app = FastAPI(
    title="DriveEval API",
    version="0.2.0",
    description="Interactive API for running and inspecting DriveEval experiments.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/planners")
def planners() -> list[dict[str, str]]:
    return [
        {"id": name, "label": PLANNER_LABELS[name]}
        for name in PLANNERS
    ]


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict:
    try:
        return RUN_STORE.create(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return RUN_STORE.public_run(RUN_STORE.get(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/scenarios/{scenario_id}")
def get_scenario(run_id: str, scenario_id: str) -> dict:
    try:
        return RUN_STORE.scenario(run_id, scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


def run() -> None:
    uvicorn.run("driveeval.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()

