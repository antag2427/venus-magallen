from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def run_anomaly_detection(
    sample_csv: str | Path,
    output_csv: str | Path,
    k: int = 4,
    random_state: int = 42,
    contamination: float = 0.01,
    n_estimators: int = 300,
    fit_subset: int = 50_000,
) -> dict[str, float | int | str]:
    df = pd.read_csv(sample_csv)
    features = [c for c in ALL_FEATURES if c in df.columns]
    X = df[features].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=500)
    kmeans.fit(Xs)
    centroid_distance = kmeans.transform(Xs).min(axis=1)

    rng = np.random.default_rng(random_state)
    subset_n = min(fit_subset, len(df))
    subset_idx = rng.choice(len(df), size=subset_n, replace=False)
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(Xs[subset_idx])
    isolation_raw = -iso.score_samples(Xs)

    centroid_score = _percentile_rank(centroid_distance)
    isolation_score = _percentile_rank(isolation_raw)
    combined = 0.5 * centroid_score + 0.5 * isolation_score

    labels = kmeans.predict(Xs)
    out = df[[c for c in ("row", "col") if c in df.columns]].copy()
    out["cluster"] = labels
    out["centroid_distance"] = centroid_distance
    out["centroid_anomaly_percentile"] = centroid_score
    out["isolation_anomaly_score"] = isolation_raw
    out["isolation_anomaly_percentile"] = isolation_score
    out["combined_anomaly_score"] = combined
    out = out.sort_values("combined_anomaly_score", ascending=False).reset_index(drop=True)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    return {
        "rows": int(len(out)),
        "fit_subset": int(subset_n),
        "top_1pct_count": int(max(1, round(len(out) * 0.01))),
        "mean_combined_score": float(combined.mean()),
        "max_combined_score": float(combined.max()),
        "output": str(output),
    }
