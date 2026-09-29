import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv("venus_clustered_clean.csv")


features = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]


print("\n================ FEATURE STATISTICS ================\n")

stats = df[features].describe().T

print(stats)



print("\n================ MEDIAN / IQR ================\n")

for feature in features:

    median = df[feature].median()

    q1 = df[feature].quantile(0.25)

    q3 = df[feature].quantile(0.75)

    iqr = q3 - q1

    print(f"\n{feature}")

    print(f"Median: {median}")

    print(f"Q1:     {q1}")

    print(f"Q3:     {q3}")

    print(f"IQR:    {iqr}")



for feature in features:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature],
        bins=100
    )

    plt.xlabel(feature)

    plt.ylabel("Number of Pixels")

    plt.title(
        f"Distribution of {feature}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()



correlation = df[features].corr()


print("\n================ CORRELATION MATRIX ================\n")

print(correlation)



plt.figure(figsize=(8, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    label="Correlation"
)

plt.xticks(
    range(len(features)),
    features,
    rotation=45
)

plt.yticks(
    range(len(features)),
    features
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.show()



if "cluster" in df.columns:

    print(
        "\n================ CLUSTER STATISTICS ================\n"
    )

    cluster_stats = df.groupby(
        "cluster"
    )[features].agg(
        ["mean", "median", "std"]
    )

    print(cluster_stats)



print(
    "\n================ RADAR DISTRIBUTION ================\n"
)

radar_percentiles = df["radar"].quantile(
    [
        0,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00
    ]
)

print(radar_percentiles)



stats.to_csv(
    "feature_statistics.csv"
)

correlation.to_csv(
    "feature_correlations.csv"
)

print(
    "\nSaved:"
)

print("feature_statistics.csv")

print("feature_correlations.csv")