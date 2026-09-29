import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from paths import CLUSTERING_OUTPUT_DIR, PROJECT_ROOT


TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "venus_training_sample.csv"
)

OUTPUT_RUNS = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_runs.csv"
)

OUTPUT_REFERENCE_ARI = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_reference_ari.csv"
)

OUTPUT_PAIRWISE_ARI = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_pairwise_ari.csv"
)

OUTPUT_CENTROIDS = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_centroids.csv"
)

FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity",
]

K = 4

SEEDS = [
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    20,
    42,
    100,
]

REFERENCE_SEED = 42

N_INIT = 10
MAX_ITER = 300


print("=" * 100)
print("K-MEANS K=4 STABILITY ANALYSIS")
print("=" * 100)

print("\nTraining file:")
print(TRAINING_FILE)

if not TRAINING_FILE.exists():
    raise FileNotFoundError(
        f"Training sample not found:\n{TRAINING_FILE}"
    )

df = pd.read_csv(TRAINING_FILE)

print(f"\nSamples: {len(df):,}")
print(f"Features: {len(FEATURES)}")

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

X_original = df[FEATURES].to_numpy(dtype=np.float64)

if not np.isfinite(X_original).all():
    raise ValueError(
        "Feature matrix contains NaN or infinite values."
    )


print("\nStandardizing features...")

scaler = StandardScaler()
X = scaler.fit_transform(X_original)

print("Standardization complete.")
print("Standardized shape:", X.shape)


all_labels = {}
all_centroids = {}
run_results = []


for seed in SEEDS:

    print(
        f"\nRunning K-Means with random_state={seed}..."
    )

    model = KMeans(
        n_clusters=K,
        random_state=seed,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )

    labels = model.fit_predict(X)

    all_labels[seed] = labels

    centroids_original = scaler.inverse_transform(
        model.cluster_centers_
    )

    all_centroids[seed] = centroids_original

    inertia = model.inertia_

    counts = np.bincount(
        labels,
        minlength=K,
    )

    percentages = (
        counts / len(labels)
    ) * 100

    print(f"Inertia: {inertia:.6f}")

    for cluster_id in range(K):
        print(
            f"  Cluster {cluster_id}: "
            f"{counts[cluster_id]:,} "
            f"({percentages[cluster_id]:.2f}%)"
        )

    row = {
        "seed": seed,
        "inertia": inertia,
    }

    for cluster_id in range(K):
        row[
            f"cluster_{cluster_id}_samples"
        ] = counts[cluster_id]

        row[
            f"cluster_{cluster_id}_percentage"
        ] = percentages[cluster_id]

    run_results.append(row)


run_results_df = pd.DataFrame(run_results)


reference_labels = all_labels[REFERENCE_SEED]

print("\n" + "=" * 100)
print(
    f"ARI AGAINST REFERENCE SEED = {REFERENCE_SEED}"
)
print("=" * 100)

ari_reference_results = []

for seed in SEEDS:

    ari = adjusted_rand_score(
        reference_labels,
        all_labels[seed],
    )

    print(
        f"Seed {seed:>3}: "
        f"ARI = {ari:.6f}"
    )

    ari_reference_results.append(
        {
            "seed": seed,
            "reference_seed": REFERENCE_SEED,
            "ari": ari,
        }
    )


ari_reference_df = pd.DataFrame(
    ari_reference_results
)


print("\n" + "=" * 100)
print("PAIRWISE ARI")
print("=" * 100)

pairwise_rows = []

for i, seed_a in enumerate(SEEDS):

    for seed_b in SEEDS[i + 1:]:

        ari = adjusted_rand_score(
            all_labels[seed_a],
            all_labels[seed_b],
        )

        pairwise_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ari": ari,
            }
        )

        print(
            f"Seed {seed_a:>3} vs "
            f"{seed_b:>3}: "
            f"ARI = {ari:.6f}"
        )


pairwise_df = pd.DataFrame(
    pairwise_rows
)

pairwise_ari_values = (
    pairwise_df["ari"].to_numpy()
)


print("\n" + "=" * 100)
print("PAIRWISE ARI SUMMARY")
print("=" * 100)

minimum_ari = np.min(pairwise_ari_values)
mean_ari = np.mean(pairwise_ari_values)
median_ari = np.median(pairwise_ari_values)
maximum_ari = np.max(pairwise_ari_values)

print(f"Minimum ARI:  {minimum_ari:.6f}")
print(f"Mean ARI:     {mean_ari:.6f}")
print(f"Median ARI:   {median_ari:.6f}")
print(f"Maximum ARI:  {maximum_ari:.6f}")


print("\n" + "=" * 100)
print("CENTROIDS FOR EACH RUN - ORIGINAL UNITS")
print("=" * 100)

print("\nNOTE:")
print("K-Means cluster IDs are arbitrary across runs.")
print("ARI is therefore the primary stability measure.")

centroid_rows = []

for seed in SEEDS:

    centroids = all_centroids[seed]

    for cluster_id in range(K):

        centroid_rows.append(
            {
                "seed": seed,
                "cluster": cluster_id,
                "radar": centroids[cluster_id, 0],
                "height": centroids[cluster_id, 1],
                "emissivity": centroids[cluster_id, 2],
                "slope": centroids[cluster_id, 3],
                "reflectivity": centroids[cluster_id, 4],
            }
        )


centroids_df = pd.DataFrame(
    centroid_rows
)


CLUSTERING_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

run_results_df.to_csv(
    OUTPUT_RUNS,
    index=False,
)

ari_reference_df.to_csv(
    OUTPUT_REFERENCE_ARI,
    index=False,
)

pairwise_df.to_csv(
    OUTPUT_PAIRWISE_ARI,
    index=False,
)

centroids_df.to_csv(
    OUTPUT_CENTROIDS,
    index=False,
)


print("\n" + "=" * 100)
print("STABILITY INTERPRETATION")
print("=" * 100)

if mean_ari >= 0.90 and minimum_ari >= 0.80:

    print("\nVERY STRONG STABILITY")
    print(
        "The K=4 solution is highly consistent "
        "across different random seeds."
    )

elif mean_ari >= 0.75 and minimum_ari >= 0.60:

    print("\nGOOD STABILITY")
    print(
        "The K=4 solution is reasonably stable "
        "across different random seeds."
    )

elif mean_ari >= 0.50:

    print("\nMODERATE STABILITY")
    print(
        "The K=4 solution shows meaningful but "
        "imperfect stability across random seeds."
    )

else:

    print("\nWEAK STABILITY")
    print(
        "The K=4 solution changes substantially "
        "between different random seeds."
    )


print("\n" + "=" * 100)
print("STABILITY ANALYSIS COMPLETE")
print("=" * 100)

print("\nFiles created:")
print(OUTPUT_RUNS)
print(OUTPUT_REFERENCE_ARI)
print(OUTPUT_PAIRWISE_ARI)
print(OUTPUT_CENTROIDS)
