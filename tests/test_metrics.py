import math

import numpy as np

from driveeval.metrics import curvature, displacement_errors, miss_rate, time_to_collision


def test_displacement_errors_are_deterministic() -> None:
    target = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    predicted = np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 0.0]])

    errors = displacement_errors(predicted, target)

    assert errors["ade"] == 1.0
    assert errors["fde"] == 2.0
    assert errors["max_de"] == 2.0
    assert miss_rate(errors["fde"], threshold_m=1.5) == 1.0
    assert miss_rate(errors["fde"], threshold_m=2.0) == 0.0


def test_time_to_collision_returns_first_hit() -> None:
    timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    ego = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    actor = np.array([[10.0, 0.0], [8.0, 0.0], [5.0, 0.0], [4.0, 0.0]])

    assert time_to_collision(ego, [actor], timestamps, collision_radius_m=1.1) == 1.0


def test_curvature_for_straight_line_is_zero() -> None:
    points = np.column_stack([np.arange(10.0), np.zeros(10)])

    values = curvature(points)

    assert math.isclose(values["curvature_mean"], 0.0, abs_tol=1e-9)
    assert math.isclose(values["curvature_max"], 0.0, abs_tol=1e-9)

