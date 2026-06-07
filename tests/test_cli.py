from pathlib import Path

from driveeval.cli import main


def test_cli_end_to_end(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios.jsonl"
    metrics = tmp_path / "metrics.csv"
    clusters = tmp_path / "clusters.csv"
    html = tmp_path / "scene.html"
    rollout = tmp_path / "rollout.csv"

    main(["make-demo-data", "--output", str(scenarios), "--scenarios", "8"])
    main(["evaluate", "--input", str(scenarios), "--planner", "lane_keeping", "--output", str(metrics)])
    main(["cluster", "--metrics", str(metrics), "--clusters", "4", "--output", str(clusters)])
    main(
        [
            "visualize",
            "--input",
            str(scenarios),
            "--scenario-id",
            "demo-0001",
            "--planner",
            "idm",
            "--output",
            str(html),
        ]
    )
    main(["rollout", "--input", str(scenarios), "--planner", "all", "--output", str(rollout)])

    assert scenarios.exists()
    assert metrics.exists()
    assert clusters.exists()
    assert clusters.with_name("clusters_summary.csv").exists()
    assert html.exists()
    assert rollout.exists()
    assert rollout.with_name("rollout_summary.csv").exists()

