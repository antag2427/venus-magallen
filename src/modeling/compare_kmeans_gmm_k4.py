import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


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

N_CLUSTERS = 4
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

df = pd.read_csv(INPUT_CSV)

missing = [f for f in FEATURES if f not in df.columns]

if missing:
    raise ValueError(
        f"Missing required feature columns: {missing}"
    )

X = df[FEATURES].values.astype(np.float64)

print(f"Samples: {len(X):,}")
print(f"Features: {X.shape[1]}")


# ============================================================
# LOAD THE EXACT SAME SCALER USED FOR K-MEANS
# ============================================================

print("\nLoading scaler parameters...")

scaler_df = pd.read_csv(SCALER_CSV)

required_columns = {"feature", "mean", "scale"}

if not required_columns.issubset(scaler_df.columns):
    raise ValueError(
        "Scaler CSV must contain: feature, mean, scale"
    )

scaler_df = scaler_df.set_index("feature")

means = np.array(
    [scaler_df.loc[f, "mean"] for f in FEATURES],
    dtype=np.float64
)

scales = np.array(
    [scaler_df.loc[f, "scale"] for f in FEATURES],
    dtype=np.float64
)

if np.any(scales == 0):
    raise ValueError(
        "At least one feature has a zero scaling factor."
    )

X_scaled = (X - means) / scales

print("Standardization complete.")


# ============================================================
# K-MEANS
# ============================================================

print("\n============================================")
print("TRAINING K-MEANS K=4")
print("============================================")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300
)

kmeans_labels = kmeans.fit_predict(X_scaled)

print("K-Means training complete.")


# ============================================================
# K-MEANS METRICS
# ============================================================

print("\nK-Means evaluation...")

kmeans_silhouette = silhouette_score(
    X_scaled,
    kmeans_labels
)

kmeans_db = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

kmeans_inertia = kmeans.inertia_


# ============================================================
# K-MEANS CLUSTER DISTRIBUTION
# ============================================================

kmeans_counts = np.bincount(
    kmeans_labels,
    minlength=N_CLUSTERS
)

kmeans_percentages = (
    kmeans_counts / len(X) * 100
)


# ============================================================
# GMM
# ============================================================

print("\n============================================")
print("TRAINING GMM K=4")
print("============================================")

gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="full",
    random_state=RANDOM_STATE,
    n_init=5,
    max_iter=300,
    reg_covar=1e-6
)

gmm_labels = gmm.fit_predict(X_scaled)

print("GMM training complete.")


# ============================================================
# GMM METRICS
# ============================================================

print("\nGMM evaluation...")

gmm_silhouette = silhouette_score(
    X_scaled,
    gmm_labels
)

gmm_db = davies_bouldin_score(
    X_scaled,
    gmm_labels
)

gmm_bic = gmm.bic(X_scaled)
gmm_aic = gmm.aic(X_scaled)


# ============================================================
# GMM CLUSTER DISTRIBUTION
# ============================================================

gmm_counts = np.bincount(
    gmm_labels,
    minlength=N_CLUSTERS
)

gmm_percentages = (
    gmm_counts / len(X) * 100
)


# ============================================================
# PRINT DIRECT COMPARISON
# ============================================================

print("\n")
print("============================================================")
print("              K-MEANS vs GMM — K=4")
print("============================================================")

print(
    f"{'Metric':<25}"
    f"{'K-Means':>15}"
    f"{'GMM':>15}"
)

print("-" * 55)

print(
    f"{'Samples':<25}"
    f"{len(X):>15,}"
    f"{len(X):>15,}"
)

print(
    f"{'Silhouette':<25}"
    f"{kmeans_silhouette:>15.6f}"
    f"{gmm_silhouette:>15.6f}"
)

print(
    f"{'Davies-Bouldin':<25}"
    f"{kmeans_db:>15.6f}"
    f"{gmm_db:>15.6f}"
)

print(
    f"{'Inertia':<25}"
    f"{kmeans_inertia:>15.2f}"
    f"{'N/A':>15}"
)

print(
    f"{'BIC':<25}"
    f"{'N/A':>15}"
    f"{gmm_bic:>15.2f}"
)

print(
    f"{'AIC':<25}"
    f"{'N/A':>15}"
    f"{gmm_aic:>15.2f}"
)

print("============================================================")


# ============================================================
# CLUSTER DISTRIBUTIONS
# ============================================================

print("\nK-Means cluster distribution:")

for cluster in range(N_CLUSTERS):
    print(
        f"Cluster {cluster}: "
        f"{kmeans_counts[cluster]:,} "
        f"({kmeans_percentages[cluster]:.2f}%)"
    )


print("\nGMM cluster distribution:")

for cluster in range(N_CLUSTERS):
    print(
        f"Cluster {cluster}: "
        f"{gmm_counts[cluster]:,} "
        f"({gmm_percentages[cluster]:.2f}%)"
    )


# ============================================================
# K-MEANS CENTROIDS IN ORIGINAL UNITS
# ============================================================

kmeans_centers_original = (
    kmeans.cluster_centers_ * scales + means
)

kmeans_centers_df = pd.DataFrame(
    kmeans_centers_original,
    columns=FEATURES
)

kmeans_centers_df.index.name = "cluster"

print("\nK-Means centroids in original units:")
print(kmeans_centers_df)


# ============================================================
# GMM MEANS IN ORIGINAL UNITS
# ============================================================

gmm_means_original = (
    gmm.means_ * scales + means
)

gmm_means_df = pd.DataFrame(
    gmm_means_original,
    columns=FEATURES
)

gmm_means_df.index.name = "cluster"

print("\nGMM means in original units:")
print(gmm_means_df)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "model": [
        "KMeans",
        "GMM"
    ],

    "samples": [
        len(X),
        len(X)
    ],

    "silhouette": [
        kmeans_silhouette,
        gmm_silhouette
    ],

    "davies_bouldin": [
        kmeans_db,
        gmm_db
    ],

    "inertia": [
        kmeans_inertia,
        np.nan
    ],

    "bic": [
        np.nan,
        gmm_bic
    ],

    "aic": [
        np.nan,
        gmm_aic
    ]
})

comparison_df.to_csv(
    "kmeans_vs_gmm_k4_comparison.csv",
    index=False
)

print(
    "\nSaved: kmeans_vs_gmm_k4_comparison.csv"
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n============================================================")
print("FINAL INTERNAL-METRIC COMPARISON")
print("============================================================")

if kmeans_silhouette > gmm_silhouette:
    print("Silhouette winner: K-Means")
else:
    print("Silhouette winner: GMM")

if kmeans_db < gmm_db:
    print("Davies-Bouldin winner: K-Means")
else:
    print("Davies-Bouldin winner: GMM")

print("============================================================")