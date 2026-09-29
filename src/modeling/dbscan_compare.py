import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_FILE = "venus_clustered_k4.csv"
SCALER_FILE = "scaler_parameters.csv"

FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]

# Lower epsilon range to investigate
EPS_VALUES = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

MIN_SAMPLES = 10

# A cluster must contain at least this percentage
# of the total dataset to count as "substantial".
SUBSTANTIAL_CLUSTER_THRESHOLD = 1.0


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

df = pd.read_csv(TRAINING_FILE)

print(f"Samples: {len(df):,}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler parameters...")

scaler_df = pd.read_csv(SCALER_FILE)

means = scaler_df["mean"].values
scales = scaler_df["scale"].values


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

X_original = df[FEATURES].values.astype(np.float64)

X = (X_original - means) / scales

print("Standardization complete.")


# ============================================================
# DBSCAN SEARCH
# ============================================================

results = []
cluster_details = []


print("\n")
print("=" * 115)
print("DBSCAN LOWER-EPSILON PARAMETER SEARCH")
print("=" * 115)

print(
    f"\nSubstantial cluster threshold: "
    f"{SUBSTANTIAL_CLUSTER_THRESHOLD:.1f}% of dataset"
)

print("\n")
print(
    f"{'eps':>6} "
    f"{'clusters':>10} "
    f"{'substantial':>13} "
    f"{'noise %':>10} "
    f"{'largest %':>12} "
    f"{'silhouette':>14} "
    f"{'DB index':>12}"
)

print("-" * 115)


for eps in EPS_VALUES:

    print(f"\nRunning DBSCAN with eps={eps:.2f}...")

    model = DBSCAN(
        eps=eps,
        min_samples=MIN_SAMPLES,
        n_jobs=-1
    )

    labels = model.fit_predict(X)

    # --------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------

    noise_mask = labels == -1

    noise_count = np.sum(noise_mask)
    noise_percentage = 100 * noise_count / len(labels)

    cluster_ids = sorted(
        cluster_id
        for cluster_id in np.unique(labels)
        if cluster_id != -1
    )

    n_clusters = len(cluster_ids)

    # --------------------------------------------------------
    # CLUSTER SIZE ANALYSIS
    # --------------------------------------------------------

    cluster_sizes = []

    for cluster_id in cluster_ids:

        count = np.sum(labels == cluster_id)

        percentage = 100 * count / len(labels)

        cluster_sizes.append({
            "cluster": cluster_id,
            "count": count,
            "percentage": percentage
        })

        cluster_details.append({
            "eps": eps,
            "cluster": cluster_id,
            "samples": count,
            "percentage": percentage
        })

    if cluster_sizes:

        largest_cluster_percentage = max(
            cluster["percentage"]
            for cluster in cluster_sizes
        )

    else:

        largest_cluster_percentage = 0.0

    # --------------------------------------------------------
    # SUBSTANTIAL CLUSTERS
    # --------------------------------------------------------

    substantial_clusters = [
        cluster
        for cluster in cluster_sizes
        if cluster["percentage"] >= SUBSTANTIAL_CLUSTER_THRESHOLD
    ]

    n_substantial = len(substantial_clusters)

    # --------------------------------------------------------
    # INTERNAL METRICS
    # --------------------------------------------------------

    silhouette = np.nan
    davies_bouldin = np.nan

    non_noise_mask = labels != -1

    X_non_noise = X[non_noise_mask]
    labels_non_noise = labels[non_noise_mask]

    unique_non_noise = np.unique(labels_non_noise)

    if len(unique_non_noise) >= 2:

        silhouette = silhouette_score(
            X_non_noise,
            labels_non_noise
        )

        davies_bouldin = davies_bouldin_score(
            X_non_noise,
            labels_non_noise
        )

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"  clusters={n_clusters}, "
        f"substantial={n_substantial}, "
        f"noise={noise_percentage:.2f}%, "
        f"largest={largest_cluster_percentage:.2f}%, "
        f"silhouette={silhouette:.6f}, "
        f"DB={davies_bouldin:.6f}"
    )

    # --------------------------------------------------------
    # STORE SUMMARY
    # --------------------------------------------------------

    results.append({
        "eps": eps,
        "clusters": n_clusters,
        "substantial_clusters": n_substantial,
        "noise_samples": noise_count,
        "noise_percentage": noise_percentage,
        "largest_cluster_percentage": largest_cluster_percentage,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "dbscan_lower_eps_results.csv",
    index=False
)

cluster_details_df = pd.DataFrame(cluster_details)

cluster_details_df.to_csv(
    "dbscan_lower_eps_cluster_sizes.csv",
    index=False
)


# ============================================================
# PRINT FULL TABLE
# ============================================================

print("\n")
print("=" * 115)
print("COMPLETE RESULTS")
print("=" * 115)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# IDENTIFY CANDIDATES WITH MULTIPLE SUBSTANTIAL CLUSTERS
# ============================================================

candidate_df = results_df[
    results_df["substantial_clusters"] >= 2
].copy()


print("\n")
print("=" * 115)
print("CONFIGURATIONS WITH AT LEAST 2 SUBSTANTIAL CLUSTERS")
print("=" * 115)


if len(candidate_df) == 0:

    print(
        "\nNo epsilon in the tested range produced "
        "at least 2 clusters larger than or equal to "
        f"{SUBSTANTIAL_CLUSTER_THRESHOLD:.1f}%."
    )

else:

    print(
        candidate_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# BEST CANDIDATE
# ============================================================

if len(candidate_df) > 0:

    # Prefer candidates with more substantial clusters,
    # then higher silhouette.
    best_candidate = candidate_df.sort_values(
        by=[
            "substantial_clusters",
            "silhouette"
        ],
        ascending=[
            False,
            False
        ]
    ).iloc[0]

    print("\n")
    print("=" * 115)
    print("BEST STRUCTURALLY USEFUL CANDIDATE")
    print("=" * 115)

    print(
        f"\neps: {best_candidate['eps']:.2f}"
    )

    print(
        f"Total clusters: "
        f"{int(best_candidate['clusters'])}"
    )

    print(
        f"Substantial clusters: "
        f"{int(best_candidate['substantial_clusters'])}"
    )

    print(
        f"Noise: "
        f"{best_candidate['noise_percentage']:.2f}%"
    )

    print(
        f"Largest cluster: "
        f"{best_candidate['largest_cluster_percentage']:.2f}%"
    )

    print(
        f"Silhouette: "
        f"{best_candidate['silhouette']:.6f}"
    )

    print(
        f"Davies-Bouldin: "
        f"{best_candidate['davies_bouldin']:.6f}"
    )


# ============================================================
# BEST PURE METRIC RESULTS
# ============================================================

valid_silhouette = results_df.dropna(
    subset=["silhouette"]
)

if len(valid_silhouette) > 0:

    best_silhouette = valid_silhouette.loc[
        valid_silhouette["silhouette"].idxmax()
    ]

    print("\n")
    print("=" * 115)
    print("BEST SILHOUETTE")
    print("=" * 115)

    print(
        f"\neps={best_silhouette['eps']:.2f}"
    )

    print(
        f"Silhouette="
        f"{best_silhouette['silhouette']:.6f}"
    )

    print(
        f"Clusters="
        f"{int(best_silhouette['clusters'])}"
    )

    print(
        f"Substantial clusters="
        f"{int(best_silhouette['substantial_clusters'])}"
    )


if len(valid_silhouette) > 0:

    best_db = valid_silhouette.loc[
        valid_silhouette["davies_bouldin"].idxmin()
    ]

    print("\n")
    print("=" * 115)
    print("BEST DAVIES-BOULDIN")
    print("=" * 115)

    print(
        f"\neps={best_db['eps']:.2f}"
    )

    print(
        f"Davies-Bouldin="
        f"{best_db['davies_bouldin']:.6f}"
    )

    print(
        f"Clusters="
        f"{int(best_db['clusters'])}"
    )

    print(
        f"Substantial clusters="
        f"{int(best_db['substantial_clusters'])}"
    )


# ============================================================
# FILE SUMMARY
# ============================================================

print("\n")
print("=" * 115)
print("SEARCH COMPLETE")
print("=" * 115)

print("\nFiles created:")
print("1. dbscan_lower_eps_results.csv")
print("2. dbscan_lower_eps_cluster_sizes.csv")