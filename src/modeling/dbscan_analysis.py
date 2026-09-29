import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "venus_clustered_k4.csv"
SCALER_CSV = "scaler_parameters_k4.csv"

FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]

# Keep this relatively small.
# Larger values make DBSCAN more restrictive.
MIN_SAMPLES = 10

# We will test only four values.
EPS_VALUES = [
    0.90,
    1.00,
    1.10,
    1.20
]

OUTPUT_CSV = "dbscan_parameter_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

df = pd.read_csv(INPUT_CSV)

missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required features: {missing}"
    )

X = df[FEATURES].values.astype(np.float64)

print(f"Samples: {len(X):,}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler parameters...")

scaler_df = pd.read_csv(SCALER_CSV)

required = {"feature", "mean", "scale"}

if not required.issubset(scaler_df.columns):
    raise ValueError(
        "Scaler CSV must contain: "
        "feature, mean, scale"
    )

scaler_df = scaler_df.set_index("feature")

means = np.array(
    [
        scaler_df.loc[f, "mean"]
        for f in FEATURES
    ],
    dtype=np.float64
)

scales = np.array(
    [
        scaler_df.loc[f, "scale"]
        for f in FEATURES
    ],
    dtype=np.float64
)

if np.any(scales == 0):
    raise ValueError(
        "One or more feature scales are zero."
    )

X_scaled = (X - means) / scales

print("Standardization complete.")


# ============================================================
# K-DISTANCE ANALYSIS
# ============================================================

print("\nCalculating nearest-neighbor distances...")

neighbors = NearestNeighbors(
    n_neighbors=MIN_SAMPLES,
    algorithm="auto",
    n_jobs=-1
)

neighbors.fit(X_scaled)

distances, _ = neighbors.kneighbors(
    X_scaled
)

k_distances = distances[:, -1]

print("\nNearest-neighbor distance statistics:")
print("---------------------------------------")

percentiles = [
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    95,
    99
]

for p in percentiles:
    value = np.percentile(
        k_distances,
        p
    )

    print(
        f"{p:>3}th percentile: "
        f"{value:.4f}"
    )

print(
    f"Maximum:          "
    f"{np.max(k_distances):.4f}"
)


# ============================================================
# DBSCAN TESTS
# ============================================================

print("\n")
print("=" * 90)
print("DBSCAN PARAMETER TEST")
print("=" * 90)

print(
    f"{'eps':>8}"
    f"{'clusters':>12}"
    f"{'noise %':>12}"
    f"{'largest %':>14}"
    f"{'silhouette':>16}"
    f"{'davies-bouldin':>18}"
)

print("-" * 90)


results = []


for eps in EPS_VALUES:

    print(
        f"\nRunning DBSCAN "
        f"eps={eps:.2f}..."
    )

    dbscan = DBSCAN(
        eps=eps,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1
    )

    labels = dbscan.fit_predict(
        X_scaled
    )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    noise_mask = labels == -1

    noise_count = np.sum(
        noise_mask
    )

    noise_percentage = (
        noise_count
        / len(labels)
        * 100
    )

    # --------------------------------------------------------
    # Actual clusters
    # --------------------------------------------------------

    valid_mask = ~noise_mask

    valid_labels = labels[
        valid_mask
    ]

    unique_clusters = np.unique(
        valid_labels
    )

    n_clusters = len(
        unique_clusters
    )

    # --------------------------------------------------------
    # Largest cluster
    # --------------------------------------------------------

    largest_cluster_percentage = 0.0

    if n_clusters > 0:

        counts = np.bincount(
            valid_labels
        )

        counts = counts[
            counts > 0
        ]

        largest_cluster_percentage = (
            np.max(counts)
            / len(labels)
            * 100
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    silhouette = np.nan
    davies_bouldin = np.nan

    if n_clusters >= 2:

        silhouette = silhouette_score(
            X_scaled[valid_mask],
            valid_labels
        )

        davies_bouldin = (
            davies_bouldin_score(
                X_scaled[valid_mask],
                valid_labels
            )
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "eps": eps,
        "min_samples": MIN_SAMPLES,
        "clusters": n_clusters,
        "noise_count": int(noise_count),
        "noise_percentage": noise_percentage,
        "largest_cluster_percentage":
            largest_cluster_percentage,
        "silhouette": silhouette,
        "davies_bouldin":
            davies_bouldin
    })

    print(
        f"{eps:>8.2f}"
        f"{n_clusters:>12}"
        f"{noise_percentage:>11.2f}%"
        f"{largest_cluster_percentage:>13.2f}%"
        f"{silhouette:>16.6f}"
        f"{davies_bouldin:>18.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n")
print(
    f"Results saved to: "
    f"{OUTPUT_CSV}"
)


# ============================================================
# BEST RESULTS
# ============================================================

valid_results = results_df.dropna(
    subset=[
        "silhouette",
        "davies_bouldin"
    ]
)


if len(valid_results) > 0:

    best_silhouette = (
        valid_results.loc[
            valid_results[
                "silhouette"
            ].idxmax()
        ]
    )

    best_db = (
        valid_results.loc[
            valid_results[
                "davies_bouldin"
            ].idxmin()
        ]
    )

    print("\n")
    print("=" * 90)
    print("BEST DBSCAN CONFIGURATIONS")
    print("=" * 90)

    print("\nBest Silhouette:")

    print(
        f"eps = "
        f"{best_silhouette['eps']:.2f}"
    )

    print(
        f"clusters = "
        f"{int(best_silhouette['clusters'])}"
    )

    print(
        f"noise = "
        f"{best_silhouette['noise_percentage']:.2f}%"
    )

    print(
        f"Silhouette = "
        f"{best_silhouette['silhouette']:.6f}"
    )

    print(
        f"Davies-Bouldin = "
        f"{best_silhouette['davies_bouldin']:.6f}"
    )


    print("\nBest Davies-Bouldin:")

    print(
        f"eps = "
        f"{best_db['eps']:.2f}"
    )

    print(
        f"clusters = "
        f"{int(best_db['clusters'])}"
    )

    print(
        f"noise = "
        f"{best_db['noise_percentage']:.2f}%"
    )

    print(
        f"Silhouette = "
        f"{best_db['silhouette']:.6f}"
    )

    print(
        f"Davies-Bouldin = "
        f"{best_db['davies_bouldin']:.6f}"
    )


print("\n")
print("=" * 90)
print("DBSCAN ANALYSIS COMPLETE")
print("=" * 90)