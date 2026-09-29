from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES


def fit_distance_uncertainty(
    sample_csv: str | Path,
    output_csv: str | Path,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 20,
    temperature: float = 1.0,
) -> dict[str, float | int | str]:
    df = pd.read_csv(sample_csv)
    features = [c for c in ALL_FEATURES if c in df.columns]
    X = df[features].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=500)
    labels = model.fit_predict(Xs)
    distances = model.transform(Xs)
    order = np.argsort(distances, axis=1)
    nearest = distances[np.arange(len(df)), order[:, 0]]
    second = distances[np.arange(len(df)), order[:, 1]]
    margin = second - nearest
    logits = -distances / max(float(temperature), 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(k)
    confidence = 1.0 - entropy

    out = df[[c for c in ("row", "col") if c in df.columns]].copy()
    out["cluster"] = labels
    out["nearest_centroid_distance"] = nearest
    out["second_centroid_distance"] = second
    out["distance_margin"] = margin
    out["confidence"] = confidence
    out["entropy"] = entropy
    for c in range(k):
        out[f"p_cluster_{c}"] = probs[:, c]

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    return {
        "rows": int(len(out)),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "mean_entropy": float(entropy.mean()),
        "ambiguous_fraction_entropy_gt_0.5": float((entropy > 0.5).mean()),
        "output": str(output),
    }
