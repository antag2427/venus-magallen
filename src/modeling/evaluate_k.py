import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)


# ============================================================
# Make src importable
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from paths import CLUSTERING_OUTPUT_DIR, FIGURES_DIR, PROJECT_ROOT  # noqa: E402


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "venus_training_sample.csv"
)

OUTPUT_FILE = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_k_evaluation_corrected.csv"
)

FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity",
]

K_VALUES = range(2, 11)

RANDOM_SEED = 42
N_INIT = 10
MAX_ITER = 300


# ============================================================
# Validation
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Training sample not found:\n{INPUT_FILE}"
    )


# ============================================================
# Load training sample
# ============================================================

print("=" * 60)
print("LOADING CORRECTED TRAINING SAMPLE")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Input file:", INPUT_FILE)
print("Number of samples:", len(df))


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


X = df[FEATURES].to_numpy(dtype=np.float64)

if not np.isfinite(X).all():
    raise ValueError(
        "Feature matrix contains NaN or infinite values."
    )

print("Feature matrix shape:", X.shape)


# ============================================================
# Standardization
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardized shape:", X_scaled.shape)


# ============================================================
# Evaluate K values
# ============================================================

inertias = []
silhouette_scores = []
davies_bouldin_scores = []

print("\n" + "=" * 60)
print("EVALUATING DIFFERENT K VALUES")
print("=" * 60)


for k in K_VALUES:

    print(f"\nRunning K = {k}...")

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels,
    )

    db_score = davies_bouldin_score(
        X_scaled,
        labels,
    )

    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(db_score)

    print(f"Inertia: {inertia:.6f}")
    print(f"Silhouette: {silhouette:.6f}")
    print(f"Davies-Bouldin: {db_score:.6f}")


# ============================================================
# Results table
# ============================================================

results = pd.DataFrame(
    {
        "K": list(K_VALUES),
        "Inertia": inertias,
        "Silhouette": silhouette_scores,
        "Davies_Bouldin": davies_bouldin_scores,
    }
)


print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# Best K according to internal metrics
# ============================================================

best_silhouette_index = int(
    np.argmax(silhouette_scores)
)

best_k_silhouette = list(K_VALUES)[
    best_silhouette_index
]

best_db_index = int(
    np.argmin(davies_bouldin_scores)
)

best_k_db = list(K_VALUES)[
    best_db_index
]

print(
    "\nBest K by Silhouette:",
    best_k_silhouette,
)

print(
    "Best K by Davies-Bouldin:",
    best_k_db,
)


# ============================================================
# Create output directories
# ============================================================

CLUSTERING_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Save numerical results
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Save diagnostic figures
# ============================================================

plots = [
    (
        inertias,
        "Inertia",
        "K-Means Inertia vs Number of Clusters",
        "kmeans_inertia.png",
    ),
    (
        silhouette_scores,
        "Silhouette Score",
        "Silhouette Score vs Number of Clusters",
        "kmeans_silhouette.png",
    ),
    (
        davies_bouldin_scores,
        "Davies-Bouldin Index",
        "Davies-Bouldin Index vs Number of Clusters",
        "kmeans_davies_bouldin.png",
    ),
]


for values, ylabel, title, filename in plots:

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        list(K_VALUES),
        values,
        marker="o",
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(ylabel)

    plt.title(title)

    plt.xticks(
        list(K_VALUES)
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 60)
print("K SELECTION EVALUATION COMPLETE")
print("=" * 60)

print("\nSaved results:")
print(OUTPUT_FILE)

print("\nSaved figures:")
print(FIGURES_DIR / "kmeans_inertia.png")
print(FIGURES_DIR / "kmeans_silhouette.png")
print(FIGURES_DIR / "kmeans_davies_bouldin.png")
