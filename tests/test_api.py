from fastapi.testclient import TestClient

from driveeval.api import app


client = TestClient(app)


def test_health_and_planners() -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    planner_ids = {item["id"] for item in client.get("/api/planners").json()}
    assert planner_ids == {"constant_velocity", "lane_keeping", "idm"}


def test_create_run_and_get_scenario() -> None:
    response = client.post(
        "/api/runs",
        json={
            "scenario_count": 3,
            "seed": 9,
            "planners": ["constant_velocity", "lane_keeping"],
            "observation_steps": 10,
            "horizon_steps": 12,
            "miss_threshold_m": 2.0,
        },
    )
    assert response.status_code == 200
    run = response.json()
    assert len(run["scenarios"]) == 3
    assert len(run["summaries"]) == 2

    scenario_id = run["scenarios"][0]["scenario_id"]
    detail = client.get(f"/api/runs/{run['run_id']}/scenarios/{scenario_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert set(payload["predictions"]) == {"constant_velocity", "lane_keeping"}
    assert len(payload["predictions"]["constant_velocity"]) == 12


def test_rejects_horizon_beyond_demo_length() -> None:
    response = client.post(
        "/api/runs",
        json={"scenario_count": 2, "observation_steps": 30, "horizon_steps": 30},
    )
    assert response.status_code == 422

