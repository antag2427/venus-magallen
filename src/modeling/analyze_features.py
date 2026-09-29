import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



df = pd.read_csv("venus_clustered_clean.csv")

print("Dataset loaded.")
print("Number of pixels:", len(df))

features = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]

X = df[features]



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


pca = PCA()

X_pca = pca.fit_transform(X_scaled)



print("\nPCA explained variance:")

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1
):

    print(
        f"PC{i}: {variance:.4f} "
        f"({variance * 100:.2f}%)"
    )


print(
    "\nTotal variance explained by PC1 + PC2:",
    f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100:.2f}%"
)



loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[
        f"PC{i}"
        for i in range(1, len(features) + 1)
    ]
)


print("\nPCA loadings:")

print(loadings)


kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters_2 = kmeans.fit_predict(X_scaled)




df["cluster_k2"] = clusters_2


centroids_scaled = kmeans.cluster_centers_



centroids_original = scaler.inverse_transform(
    centroids_scaled
)


centroid_df = pd.DataFrame(
    centroids_original,
    columns=features
)


centroid_df.index.name = "cluster"


print("\nCluster centroids in original units:")

print(
    centroid_df
)




cluster_means = df.groupby(
    "cluster_k2"
)[features].mean()


print("\nCluster feature means:")

print(
    cluster_means
)



plt.figure(figsize=(9, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters_2,
    s=5,
    alpha=0.5
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Venus Pixels in PCA Space — K=2"
)

plt.colorbar(
    label="Cluster"
)

plt.tight_layout()

plt.show()



plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(features) + 1),
    pca.explained_variance_ratio_,
    marker="o"
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Explained Variance Ratio"
)

plt.title(
    "PCA Explained Variance"
)

plt.xticks(
    range(1, len(features) + 1)
)

plt.grid(True)

plt.tight_layout()

plt.show()




centroid_df.to_csv(
    "cluster_centroids_k2.csv"
)

loadings.to_csv(
    "pca_loadings.csv"
)


print("\nAnalysis files saved:")
print("cluster_centroids_k2.csv")
print("pca_loadings.csv")