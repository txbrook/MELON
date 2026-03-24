from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def read_coords(melon_dir: str, coords_csv: str | None = None) -> pd.DataFrame:
    if coords_csv:
        if not os.path.exists(coords_csv):
            raise FileNotFoundError(f"Coordinate file not found: {coords_csv}")
        return pd.read_csv(coords_csv, index_col=0)

    path = os.path.join(melon_dir, "coords.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)

    raise FileNotFoundError(f"No coords.csv found in {melon_dir}")


def spatial_smooth_labels(
    labels,
    coords,
    k: int = 8,
    threshold: float = 0.6,
    rounds: int = 1,
):
    labels = np.asarray(labels, dtype=int).copy()
    coords = np.asarray(coords, dtype=float)
    n_obs = int(labels.shape[0])
    if n_obs <= 1:
        return labels, []

    k = max(1, min(int(k), n_obs - 1))
    rounds = max(1, int(rounds))
    threshold = float(threshold)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    scale = float(np.median(distances[:, -1]))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    weights = np.exp(-0.5 * (distances / scale) ** 2)
    current = labels.copy()
    changes_per_round = []

    for _ in range(rounds):
        updated = current.copy()
        changed = 0
        for i in range(n_obs):
            neigh_idx = indices[i]
            neigh_labels = current[neigh_idx]
            neigh_weights = weights[i]

            score_by_label = {}
            for lab in np.unique(neigh_labels):
                score_by_label[int(lab)] = float(neigh_weights[neigh_labels == lab].sum())

            current_label = int(current[i])
            score_by_label[current_label] = score_by_label.get(current_label, 0.0) + 1.0

            best_label, best_score = max(score_by_label.items(), key=lambda item: (item[1], -item[0]))
            total_score = float(sum(score_by_label.values()))
            support = best_score / total_score if total_score > 0 else 0.0
            if best_label != current_label and support >= threshold:
                updated[i] = best_label
                changed += 1

        changes_per_round.append(changed)
        current = updated
        if changed == 0:
            break

    return current, changes_per_round
