import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# Make src importable
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from paths import (  # noqa: E402
    K4_CENTROIDS_CSV,
    K4_SAMPLE_CSV,
    K4_SCALER_CSV,
    PROJECT_ROOT,
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "venus_training_sample.csv"

OUTPUT_DATA = K4_SAMPLE_CSV
OUTPUT_CENTROIDS = K4_CENTROIDS_CSV
OUTPUT_SCALER = K4_SCALER_CSV

FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity",
]

K = 4
RANDOM_SEED = 42
N_INIT = 10
MAX_ITER = 300


# ============================================================
# Load training sample
# ============================================================

print("=" * 60)
print("K=4 K-MEANS TRAINING")
print("=" * 60)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nTraining sample:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Training sample not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nSamples loaded:", len(df))
print("Columns:", list(df.columns))


missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required feature columns: {missing_features}"
    )


# ============================================================
# Feature matrix
# ============================================================

X = df[FEATURES].to_numpy(dtype=np.float64)

print("\nFeature matrix shape:", X.shape)


if not np.isfinite(X).all():
    raise ValueError(
        "Feature matrix contains NaN or infinite values."
    )


# ============================================================
# Standardization
# ============================================================

print("\nStarting standardization...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardization complete.")


# ============================================================
# K-Means
# ============================================================

print("\n" + "=" * 60)
print("STARTING K=4 K-MEANS")
print("=" * 60)

print("K:", K)
print("Samples:", X_scaled.shape[0])
print("Features:", X_scaled.shape[1])
print("n_init:", N_INIT)
print("max_iter:", MAX_ITER)
print("random_state:", RANDOM_SEED)

print("\nK-Means is running...")
print("Please wait...")


kmeans = KMeans(
    n_clusters=K,
    random_state=RANDOM_SEED,
    n_init=N_INIT,
    max_iter=MAX_ITER,
)

clusters = kmeans.fit_predict(X_scaled)


print("\nK=4 K-Means finished!")

print("Iterations:", kmeans.n_iter_)
print("Final inertia:", kmeans.inertia_)


# ============================================================
# Attach cluster labels
# ============================================================

df["cluster"] = clusters


# ============================================================
# Cluster sizes
# ============================================================

print("\n" + "=" * 60)
print("K=4 CLUSTER SIZES")
print("=" * 60)

cluster_sizes = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_sizes)

print("\nCluster percentages:")

cluster_percentages = (
    df["cluster"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for cluster, percentage in cluster_percentages.items():
    print(
        f"Cluster {cluster}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# Centroids in original feature units
# ============================================================

centroids_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centroid_df = pd.DataFrame(
    centroids_original,
    columns=FEATURES,
)

centroid_df.index.name = "cluster"


print("\n" + "=" * 60)
print("K=4 CLUSTER CENTROIDS")
print("=" * 60)

print(centroid_df)


# ============================================================
# Save outputs
# ============================================================

K4_SAMPLE_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_DATA,
    index=False,
)

centroid_df.to_csv(
    OUTPUT_CENTROIDS,
)

scaler_parameters = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    }
)

scaler_parameters.to_csv(
    OUTPUT_SCALER,
    index=False,
)


# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 60)
print("K=4 TRAINING COMPLETE")
print("=" * 60)

print("\nFiles created:")

print(OUTPUT_DATA)
print(OUTPUT_CENTROIDS)
print(OUTPUT_SCALER)
