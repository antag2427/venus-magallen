from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES


def _contingency(reference: np.ndarray, candidate: np.ndarray, k: int) -> np.ndarray:
    table = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(reference, candidate):
        if 0 <= a < k and 0 <= b < k:
            table[a, b] += 1
    return table


def align_labels(reference: np.ndarray, candidate: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int]]:
    table = _contingency(reference, candidate, k)
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    aligned = np.array([mapping.get(int(x), int(x)) for x in candidate], dtype=np.int16)
    return aligned, mapping


def normalized_entropy(labels_matrix: np.ndarray, k: int) -> np.ndarray:
    counts = np.zeros((labels_matrix.shape[1], k), dtype=np.float64)
    for model_idx in range(labels_matrix.shape[1]):
        for c in range(k):
            counts[:, c] += (labels_matrix[:, model_idx] == c)
    probs = counts / labels_matrix.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return ent / np.log(k)


def run_consensus(
    sample_csv: str | Path,
    output_dir: str | Path,
    k: int = 4,
    subset_size: int = 20_000,
    random_state: int = 42,
    n_init: int = 20,
) -> dict[str, str | int | float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(sample_csv)
    feature_columns = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Advanced sample is missing features: {missing}")
    X = df[feature_columns].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rng = np.random.default_rng(random_state)
    subset_n = min(subset_size, len(df))
    subset_idx = np.sort(rng.choice(len(df), size=subset_n, replace=False))
    Xsub = Xs[subset_idx]

    models = {
        "kmeans": KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=500),
        "minibatch_kmeans": MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            batch_size=2048,
            max_iter=500,
        ),
        "gmm": GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=3,
            max_iter=300,
        ),
        "birch": Birch(n_clusters=k, threshold=0.5),
    }

    raw_labels: dict[str, np.ndarray] = {}
    for name, model in models.items():
        raw_labels[name] = model.fit_predict(Xsub)

    reference_name = "kmeans"
    reference = raw_labels[reference_name]
    aligned = {reference_name: reference.copy()}
    mappings: dict[str, dict[int, int]] = {reference_name: {i: i for i in range(k)}}
    for name, labels in raw_labels.items():
        if name == reference_name:
            continue
        aligned[name], mapping = align_labels(reference, labels, k)
        mappings[name] = mapping

    label_matrix = np.column_stack([aligned[name] for name in models])
    consensus = np.apply_along_axis(lambda row: np.bincount(row, minlength=k).argmax(), 1, label_matrix)
    agreement = (label_matrix == consensus[:, None]).mean(axis=1)
    entropy = normalized_entropy(label_matrix, k)

    pairwise = []
    for a, b in combinations(models, 2):
        pairwise.append({"model_a": a, "model_b": b, "ARI": adjusted_rand_score(raw_labels[a], raw_labels[b])})
    pairwise_df = pd.DataFrame(pairwise)
    pairwise_df.to_csv(output / "v2_consensus_pairwise_ari.csv", index=False)

    labels_df = pd.DataFrame({"sample_index": subset_idx})
    for name in models:
        labels_df[f"label_{name}"] = raw_labels[name]
        labels_df[f"aligned_{name}"] = aligned[name]
    labels_df["consensus_label"] = consensus
    labels_df["consensus_agreement"] = agreement
    labels_df["consensus_entropy"] = entropy
    labels_df.to_csv(output / "v2_consensus_labels.csv", index=False)

    scaler_df = pd.DataFrame({"feature": feature_columns, "mean": scaler.mean_, "scale": scaler.scale_})
    scaler_df.to_csv(output / "v2_advanced_scaler.csv", index=False)

    metrics = {
        "sample_rows": len(df),
        "consensus_rows": subset_n,
        "models": list(models),
        "mean_consensus_agreement": float(agreement.mean()),
        "median_consensus_agreement": float(np.median(agreement)),
        "mean_consensus_entropy": float(entropy.mean()),
        "min_pairwise_ARI": float(pairwise_df["ARI"].min()),
        "mean_pairwise_ARI": float(pairwise_df["ARI"].mean()),
        "max_pairwise_ARI": float(pairwise_df["ARI"].max()),
        "random_state": random_state,
    }
    pd.DataFrame([metrics]).to_csv(output / "v2_consensus_summary.csv", index=False)
    (output / "v2_label_mappings.json").write_text(
        __import__("json").dumps(mappings, indent=2),
        encoding="utf-8",
    )
    return {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in metrics.items()}
