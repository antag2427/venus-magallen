from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES
from .raster_features import aligned_rasters, derive_local_features, iter_blocks, _read_aligned_window

CLUSTER_NODATA = 255
FLOAT_NODATA = -9999.0


def _percentile_reference(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    positions = np.searchsorted(ordered, query, side="right")
    return positions.astype(np.float64) / max(len(ordered), 1)


def build_full_v2_maps(
    sample_csv: str | Path,
    output_dir: str | Path,
    k: int = 4,
    random_state: int = 42,
    block_size: int = 512,
    local_radius: int = 2,
    isolation_contamination: float = 0.01,
    isolation_estimators: int = 300,
    isolation_fit_subset: int = 50_000,
) -> dict[str, str | int | float]:
    """Create full-resolution V2 regime, confidence, and anomaly GeoTIFFs.

    The model is trained on the V2 enriched sample and then applied block-wise so the
    full Venus raster does not need to be held in RAM.
    """
    sample = pd.read_csv(sample_csv)
    features = [c for c in ALL_FEATURES if c in sample.columns]
    if features != ALL_FEATURES:
        missing = [c for c in ALL_FEATURES if c not in sample.columns]
        raise ValueError(f"Advanced sample is missing features: {missing}")
    X = sample[features].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=500)
    model.fit(Xs)

    rng = np.random.default_rng(random_state)
    iso_n = min(isolation_fit_subset, len(sample))
    iso_idx = rng.choice(len(sample), size=iso_n, replace=False)
    isolation = IsolationForest(
        n_estimators=isolation_estimators,
        contamination=isolation_contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    isolation.fit(Xs[iso_idx])

    sample_centroid_distance = model.transform(Xs).min(axis=1)
    sample_isolation_raw = -isolation.score_samples(Xs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model.cluster_centers_, columns=features).assign(cluster=np.arange(k)).to_csv(
        out / "v2_kmeans_centroids_standardized.csv", index=False
    )
    pd.DataFrame(scaler.mean_, index=features, columns=["mean"]).assign(scale=scaler.scale_).to_csv(
        out / "v2_scaler_parameters.csv"
    )

    cluster_path = out / "venus_v2_clusters_k4.tif"
    confidence_path = out / "venus_v2_confidence.tif"
    anomaly_path = out / "venus_v2_anomaly.tif"

    ctx, radar, sources = aligned_rasters()
    with ctx:
        profile = radar.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=CLUSTER_NODATA, compress="lzw", predictor=2)
        float_profile = radar.profile.copy()
        float_profile.update(count=1, dtype="float32", nodata=FLOAT_NODATA, compress="lzw", predictor=3)
        with rasterio.open(cluster_path, "w", **profile) as dst_cluster, \
             rasterio.open(confidence_path, "w", **float_profile) as dst_conf, \
             rasterio.open(anomaly_path, "w", **float_profile) as dst_anomaly:
            px_x, px_y = abs(radar.res[0]), abs(radar.res[1])
            halo = int(local_radius)
            for row, col, base_window in iter_blocks(radar.width, radar.height, block_size):
                read_window = rasterio.windows.Window(
                    max(0, col - halo),
                    max(0, row - halo),
                    min(radar.width - max(0, col - halo), int(base_window.width) + 2 * halo),
                    min(radar.height - max(0, row - halo), int(base_window.height) + 2 * halo),
                )
                arrays, masks = _read_aligned_window(sources, read_window)
                valid = np.logical_and.reduce([masks[name] for name in features[:5]])
                derived = derive_local_features(arrays, masks, local_radius, px_x, px_y)
                local_features = {**arrays, **derived}

                r0 = row - int(read_window.row_off)
                c0 = col - int(read_window.col_off)
                h = int(base_window.height)
                w = int(base_window.width)
                center_valid = valid[r0:r0 + h, c0:c0 + w]

                cluster_block = np.full((h, w), CLUSTER_NODATA, dtype=np.uint8)
                confidence_block = np.full((h, w), FLOAT_NODATA, dtype=np.float32)
                anomaly_block = np.full((h, w), FLOAT_NODATA, dtype=np.float32)

                if np.any(center_valid):
                    stack = np.column_stack([
                        local_features[name][r0:r0 + h, c0:c0 + w][center_valid]
                        for name in ALL_FEATURES
                    ])
                    block_scaled = scaler.transform(stack)
                    labels = model.predict(block_scaled)
                    distances = model.transform(block_scaled)
                    nearest = distances.min(axis=1)
                    order = np.argsort(distances, axis=1)
                    second = distances[np.arange(len(nearest)), order[:, 1]]
                    logits = -distances
                    logits -= logits.max(axis=1, keepdims=True)
                    probs = np.exp(logits)
                    probs /= probs.sum(axis=1, keepdims=True)
                    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(k)
                    confidence = 1.0 - entropy

                    iso_raw = -isolation.score_samples(block_scaled)
                    centroid_pct = _percentile_reference(sample_centroid_distance, nearest)
                    isolation_pct = _percentile_reference(sample_isolation_raw, iso_raw)
                    combined = 0.5 * centroid_pct + 0.5 * isolation_pct

                    flat_idx = np.flatnonzero(center_valid)
                    cluster_block.ravel()[flat_idx] = labels.astype(np.uint8)
                    confidence_block.ravel()[flat_idx] = confidence.astype(np.float32)
                    anomaly_block.ravel()[flat_idx] = combined.astype(np.float32)

                dst_cluster.write(cluster_block, 1, window=base_window)
                dst_conf.write(confidence_block, 1, window=base_window)
                dst_anomaly.write(anomaly_block, 1, window=base_window)

    return {
        "rows": int(len(sample)),
        "k": int(k),
        "cluster_raster": str(cluster_path),
        "confidence_raster": str(confidence_path),
        "anomaly_raster": str(anomaly_path),
    }
