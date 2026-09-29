import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)


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

K = 4

# Keep this manageable for hierarchical clustering.
SAMPLE_SIZE = 3000

RANDOM_STATE = 42

LINKAGE_METHOD = "ward"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

df = pd.read_csv(TRAINING_FILE)

print(f"Total samples: {len(df):,}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler parameters...")

scaler_df = pd.read_csv(SCALER_FILE)

means = scaler_df["mean"].values.astype(np.float64)
scales = scaler_df["scale"].values.astype(np.float64)


# ============================================================
# STANDARDIZE
# ============================================================

X_original = df[FEATURES].values.astype(np.float64)

X = (X_original - means) / scales

print("Standardization complete.")


# ============================================================
# SELECT REPRESENTATIVE SAMPLE
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(SAMPLE_SIZE, len(X))

indices = rng.choice(
    len(X),
    size=sample_size,
    replace=False
)

X_sample = X[indices]
X_original_sample = X_original[indices]


print(
    f"\nUsing {sample_size:,} representative samples "
    "for hierarchical clustering."
)


# ============================================================
# HIERARCHICAL LINKAGE
# ============================================================

print("\nCalculating Ward linkage...")

Z = linkage(
    X_sample,
    method=LINKAGE_METHOD
)

print("Ward linkage complete.")


# ============================================================
# DENDROGRAM
# ============================================================

print("\nCreating dendrogram...")

plt.figure(figsize=(14, 8))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=40,
    show_leaf_counts=True,
    show_contracted=True
)

plt.title(
    "Venus Feature-Space Hierarchical Clustering"
)

plt.xlabel(
    "Merged clusters"
)

plt.ylabel(
    "Ward linkage distance"
)

plt.tight_layout()

plt.savefig(
    "hierarchical_dendrogram.png",
    dpi=200
)

plt.close()

print(
    "Saved: hierarchical_dendrogram.png"
)


# ============================================================
# EXTRACT K=4 HIERARCHICAL CLUSTERS
# ============================================================

print("\nExtracting K=4 hierarchical clusters...")

hier_labels = fcluster(
    Z,
    t=K,
    criterion="maxclust"
)

# Convert labels from 1..K to 0..K-1
hier_labels = hier_labels - 1

print("Hierarchical clustering complete.")


# ============================================================
# CLUSTER SIZES
# ============================================================

print("\n")
print("=" * 90)
print("HIERARCHICAL CLUSTER SIZES")
print("=" * 90)

hier_counts = np.bincount(
    hier_labels,
    minlength=K
)

hier_percentages = (
    hier_counts / len(hier_labels)
) * 100

for cluster_id in range(K):

    print(
        f"Cluster {cluster_id}: "
        f"{hier_counts[cluster_id]:,} "
        f"({hier_percentages[cluster_id]:.2f}%)"
    )


# ============================================================
# HIERARCHICAL FEATURE MEANS
# ============================================================

print("\n")
print("=" * 90)
print("HIERARCHICAL CLUSTER FEATURE MEANS")
print("=" * 90)

hier_rows = []

for cluster_id in range(K):

    mask = hier_labels == cluster_id

    values = X_original_sample[mask]

    feature_means = values.mean(axis=0)

    print(f"\nCluster {cluster_id}")

    print(
        f"  radar:        {feature_means[0]:.3f}"
    )

    print(
        f"  height:       {feature_means[1]:.3f}"
    )

    print(
        f"  emissivity:   {feature_means[2]:.3f}"
    )

    print(
        f"  slope:        {feature_means[3]:.3f}"
    )

    print(
        f"  reflectivity: {feature_means[4]:.3f}"
    )

    hier_rows.append({
        "cluster": cluster_id,
        "samples": int(np.sum(mask)),
        "percentage": 100 * np.sum(mask) / len(hier_labels),
        "radar_mean": feature_means[0],
        "height_mean": feature_means[1],
        "emissivity_mean": feature_means[2],
        "slope_mean": feature_means[3],
        "reflectivity_mean": feature_means[4]
    })


hier_means_df = pd.DataFrame(hier_rows)

hier_means_df.to_csv(
    "hierarchical_cluster_means.csv",
    index=False
)


# ============================================================
# HIERARCHICAL METRICS
# ============================================================

print("\n")
print("=" * 90)
print("HIERARCHICAL VALIDATION")
print("=" * 90)

hier_silhouette = silhouette_score(
    X_sample,
    hier_labels
)

hier_db = davies_bouldin_score(
    X_sample,
    hier_labels
)

print(
    f"\nSilhouette: "
    f"{hier_silhouette:.6f}"
)

print(
    f"Davies-Bouldin: "
    f"{hier_db:.6f}"
)


# ============================================================
# K-MEANS ON THE SAME 3,000 SAMPLES
# ============================================================

print("\n")
print("=" * 90)
print("K-MEANS K=4 ON THE SAME 3,000 SAMPLES")
print("=" * 90)

kmeans = KMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300
)

kmeans_labels = kmeans.fit_predict(
    X_sample
)

print("K-Means complete.")


# ============================================================
# K-MEANS METRICS
# ============================================================

kmeans_silhouette = silhouette_score(
    X_sample,
    kmeans_labels
)

kmeans_db = davies_bouldin_score(
    X_sample,
    kmeans_labels
)

kmeans_inertia = kmeans.inertia_


# ============================================================
# ARI
# ============================================================

ari = adjusted_rand_score(
    kmeans_labels,
    hier_labels
)


# ============================================================
# K-MEANS CLUSTER SIZES
# ============================================================

kmeans_counts = np.bincount(
    kmeans_labels,
    minlength=K
)

kmeans_percentages = (
    kmeans_counts / len(kmeans_labels)
) * 100


print("\nK-Means cluster sizes:")

for cluster_id in range(K):

    print(
        f"Cluster {cluster_id}: "
        f"{kmeans_counts[cluster_id]:,} "
        f"({kmeans_percentages[cluster_id]:.2f}%)"
    )


# ============================================================
# COMPARISON
# ============================================================

print("\n")
print("=" * 90)
print("K-MEANS vs HIERARCHICAL")
print("=" * 90)

comparison = pd.DataFrame([
    {
        "method": "K-Means K=4",
        "samples": sample_size,
        "silhouette": kmeans_silhouette,
        "davies_bouldin": kmeans_db,
        "inertia": kmeans_inertia,
        "ARI_vs_other": ari
    },
    {
        "method": "Hierarchical Ward K=4",
        "samples": sample_size,
        "silhouette": hier_silhouette,
        "davies_bouldin": hier_db,
        "inertia": np.nan,
        "ARI_vs_other": ari
    }
])

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    "kmeans_vs_hierarchical_sample.csv",
    index=False
)

pd.DataFrame({
    "sample_index": indices,
    "hierarchical_cluster": hier_labels,
    "kmeans_cluster": kmeans_labels
}).to_csv(
    "hierarchical_vs_kmeans_labels.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("HIERARCHICAL ANALYSIS COMPLETE")
print("=" * 90)

print("\nFiles created:")

print("1. hierarchical_dendrogram.png")
print("2. hierarchical_cluster_means.csv")
print("3. kmeans_vs_hierarchical_sample.csv")
print("4. hierarchical_vs_kmeans_labels.csv")

print("\nKey values:")

print(
    f"Hierarchical Silhouette: "
    f"{hier_silhouette:.6f}"
)

print(
    f"Hierarchical Davies-Bouldin: "
    f"{hier_db:.6f}"
)

print(
    f"K-Means Silhouette: "
    f"{kmeans_silhouette:.6f}"
)

print(
    f"K-Means Davies-Bouldin: "
    f"{kmeans_db:.6f}"
)

print(
    f"K-Means vs Hierarchical ARI: "
    f"{ari:.6f}"
)