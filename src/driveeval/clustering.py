from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "ade",
    "fde",
    "max_de",
    "miss",
    "time_to_collision",
    "collision",
    "lane_deviation_mean",
    "lane_deviation_max",
    "offroad_events",
    "offroad_rate",
    "jerk_mean",
    "jerk_max",
    "curvature_mean",
    "curvature_max",
)


def failure_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in ("miss", "collision", "offroad_events"):
        if column in frame:
            mask = mask | (frame[column].fillna(0) > 0)
    if "fde" in frame:
        mask = mask | (frame["fde"].fillna(0) >= frame["fde"].fillna(0).quantile(0.9))
    return mask


def build_feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    columns.extend(column for column in frame.columns if column.startswith("slice_"))
    if not columns:
        raise ValueError("no numeric metric columns found")
    values = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return (values - mean) / std, columns


def _initial_centroids(values: np.ndarray, k: int) -> np.ndarray:
    first = int(np.argmax(np.linalg.norm(values, axis=1)))
    centers = [values[first]]
    while len(centers) < k:
        distances = np.min(
            np.linalg.norm(values[:, None, :] - np.asarray(centers)[None, :, :], axis=2),
            axis=1,
        )
        centers.append(values[int(np.argmax(distances))])
    return np.asarray(centers, dtype=float)


def _kmeans_numpy(values: np.ndarray, k: int, max_iter: int) -> np.ndarray:
    centers = _initial_centroids(values, k)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            members = values[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)
    return labels


def _kmeans_torch(values: np.ndarray, k: int, max_iter: int) -> np.ndarray:
    try:
        import torch
    except ImportError:
        return _kmeans_numpy(values, k, max_iter)

    device = torch.device("cpu")
    data = torch.as_tensor(values, dtype=torch.float32, device=device)
    centers = torch.as_tensor(_initial_centroids(values, k), dtype=torch.float32, device=device)
    labels = torch.zeros(len(values), dtype=torch.long, device=device)
    for _ in range(max_iter):
        distances = torch.cdist(data, centers)
        next_labels = torch.argmin(distances, dim=1)
        if torch.equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            members = data[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(dim=0)
    return labels.cpu().numpy()


@dataclass
class FailureClusterer:
    n_clusters: int = 23
    max_iter: int = 100
    backend: str = "numpy"

    def fit_predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        failures = frame.loc[failure_mask(frame)].copy()
        if failures.empty:
            failures = frame.copy()
        values, columns = build_feature_matrix(failures)
        k = min(self.n_clusters, len(failures))
        labels = (
            _kmeans_torch(values, k, self.max_iter)
            if self.backend == "torch"
            else _kmeans_numpy(values, k, self.max_iter)
        )
        failures["failure_cluster"] = labels
        failures.attrs["feature_columns"] = columns
        return failures

    def summarize(self, clustered: pd.DataFrame) -> pd.DataFrame:
        if "failure_cluster" not in clustered:
            raise ValueError("clustered frame must contain failure_cluster")
        aggregations = {
            "scenario_id": "count",
            "ade": "mean",
            "fde": "mean",
            "miss": "mean",
            "collision": "mean",
            "offroad_events": "mean",
            "jerk_mean": "mean",
        }
        available = {key: value for key, value in aggregations.items() if key in clustered.columns}
        summary = clustered.groupby("failure_cluster", sort=True).agg(available)
        summary = summary.rename(columns={"scenario_id": "count"}).reset_index()
        return summary.sort_values(["count", "fde"], ascending=[False, False])


def find_similar_failures(frame: pd.DataFrame, scenario_id: str, k: int = 5) -> pd.DataFrame:
    if scenario_id not in set(frame["scenario_id"]):
        raise KeyError(f"scenario_id not found: {scenario_id}")
    values, columns = build_feature_matrix(frame)
    query_index = int(np.flatnonzero(frame["scenario_id"].to_numpy() == scenario_id)[0])
    distances = np.linalg.norm(values - values[query_index], axis=1)
    order = np.argsort(distances)
    result = frame.iloc[order[: k + 1]].copy()
    result["similarity_distance"] = distances[order[: k + 1]]
    result.attrs["feature_columns"] = columns
    return result[result["scenario_id"] != scenario_id].head(k)

