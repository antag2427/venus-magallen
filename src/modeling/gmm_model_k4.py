import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "venus_clustered_k4.csv"

CENTROIDS_KMEANS_CSV = "cluster_centroids_k4.csv"
SCALER_CSV = "scaler_parameters_k4.csv"

OUTPUT_MODEL = "gmm_k4_model.npz"
OUTPUT_CENTROIDS = "gmm_centroids_k4.csv"
OUTPUT_STATS = "gmm_k4_statistics.csv"


FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]

N_COMPONENTS = 4
RANDOM_STATE = 42


# ============================================================
# LOAD TRAINING SAMPLE
# ============================================================

print("Loading training sample...")

df = pd.read_csv(INPUT_CSV)

print(f"Training samples loaded: {len(df):,}")

# Make sure all required columns exist
missing = [feature for feature in FEATURES if feature not in df.columns]

if missing:
    raise ValueError(
        f"Missing required feature columns: {missing}"
    )

X = df[FEATURES].values.astype(np.float64)

print(f"Number of features: {X.shape[1]}")
print(f"Feature matrix shape: {X.shape}")


# ============================================================
# LOAD K-MEANS SCALER PARAMETERS
# ============================================================

print("\nLoading K-Means scaler parameters...")

scaler_df = pd.read_csv(SCALER_CSV)

print("\nScaler file:")
print(scaler_df)

# Expected structure:
# feature, mean, scale

if not {"feature", "mean", "scale"}.issubset(scaler_df.columns):
    raise ValueError(
        "Scaler CSV must contain columns: feature, mean, scale"
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
    raise ValueError("One or more scaler scales are zero.")

# Apply the exact same transformation used for K-Means
X_scaled = (X - means) / scales

print("\nStandardization complete.")


# ============================================================
# TRAIN GAUSSIAN MIXTURE MODEL
# ============================================================

print("\nTraining Gaussian Mixture Model...")
print(f"Components: {N_COMPONENTS}")
print("Covariance type: full")

gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type="full",
    random_state=RANDOM_STATE,
    n_init=5,
    max_iter=300,
    reg_covar=1e-6
)

gmm.fit(X_scaled)

print("GMM training complete.")


# ============================================================
# PREDICT CLUSTERS
# ============================================================

print("\nPredicting cluster assignments...")

labels = gmm.predict(X_scaled)

print("Prediction complete.")


# ============================================================
# BASIC CLUSTER COUNTS
# ============================================================

cluster_counts = np.bincount(
    labels,
    minlength=N_COMPONENTS
)

cluster_percentages = (
    cluster_counts / len(labels) * 100
)

print("\nCluster distribution:")
print("--------------------------------")

for cluster in range(N_COMPONENTS):
    print(
        f"Cluster {cluster}: "
        f"{cluster_counts[cluster]:,} samples "
        f"({cluster_percentages[cluster]:.2f}%)"
    )


# ============================================================
# SILHOUETTE SCORE
# ============================================================

print("\nCalculating Silhouette score...")

silhouette = silhouette_score(
    X_scaled,
    labels
)

print(f"Silhouette score: {silhouette:.6f}")


# ============================================================
# DAVIES-BOULDIN INDEX
# ============================================================

print("\nCalculating Davies-Bouldin index...")

davies_bouldin = davies_bouldin_score(
    X_scaled,
    labels
)

print(
    f"Davies-Bouldin index: "
    f"{davies_bouldin:.6f}"
)


# ============================================================
# GMM MODEL QUALITY: BIC AND AIC
# ============================================================

print("\nCalculating GMM information criteria...")

bic = gmm.bic(X_scaled)
aic = gmm.aic(X_scaled)

print(f"BIC: {bic:.2f}")
print(f"AIC: {aic:.2f}")


# ============================================================
# GMM MEANS IN ORIGINAL UNITS
# ============================================================

print("\nConverting GMM means back to original units...")

gmm_means_scaled = gmm.means_

gmm_means_original = (
    gmm_means_scaled * scales + means
)

centroids_df = pd.DataFrame(
    gmm_means_original,
    columns=FEATURES
)

centroids_df.index.name = "cluster"

print("\nGMM cluster means:")
print(centroids_df)


# ============================================================
# SAVE GMM PARAMETERS
# ============================================================

print("\nSaving GMM model parameters...")

np.savez(
    OUTPUT_MODEL,

    # Model parameters
    weights=gmm.weights_,
    means_scaled=gmm.means_,
    covariances_scaled=gmm.covariances_,

    # Standardization parameters
    scaler_means=means,
    scaler_scales=scales,

    # Metadata
    n_components=np.array([N_COMPONENTS]),
)

print(f"Saved: {OUTPUT_MODEL}")


# ============================================================
# SAVE ORIGINAL-UNIT CENTROIDS
# ============================================================

centroids_df.to_csv(
    OUTPUT_CENTROIDS
)

print(f"Saved: {OUTPUT_CENTROIDS}")


# ============================================================
# SAVE SUMMARY STATISTICS
# ============================================================

stats = pd.DataFrame({
    "metric": [
        "number_of_components",
        "number_of_samples",
        "silhouette",
        "davies_bouldin",
        "bic",
        "aic"
    ],

    "value": [
        N_COMPONENTS,
        len(X),
        silhouette,
        davies_bouldin,
        bic,
        aic
    ]
})

stats.to_csv(
    OUTPUT_STATS,
    index=False
)

print(f"Saved: {OUTPUT_STATS}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================")
print("GMM K=4 COMPLETE")
print("============================================")

print(f"Samples:           {len(X):,}")
print(f"Components:        {N_COMPONENTS}")
print(f"Silhouette:        {silhouette:.6f}")
print(f"Davies-Bouldin:    {davies_bouldin:.6f}")
print(f"BIC:               {bic:.2f}")
print(f"AIC:               {aic:.2f}")

print("\nCluster percentages:")

for cluster in range(N_COMPONENTS):
    print(
        f"Cluster {cluster}: "
        f"{cluster_percentages[cluster]:.2f}%"
    )

print("\nOutput files:")
print(f"  {OUTPUT_MODEL}")
print(f"  {OUTPUT_CENTROIDS}")
print(f"  {OUTPUT_STATS}")

print("\nDone.")