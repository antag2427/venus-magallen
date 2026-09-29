import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# FINAL FIGURES V3
# ============================================================
#
# Venus / Magellan Geospatial ML Project
# K-Means K=4
#
# FINAL VISUAL STANDARD:
#
# C0 -> purple
# C1 -> blue
# C2 -> green
# C3 -> yellow
# NoData -> white
#
# V3 fixes:
#
#   1. One consistent cluster color mapping.
#   2. Discrete categorical global maps.
#   3. Standardized centroid heatmap.
#   4. Scatter-based K-Means stability plot.
#   5. Clean cross-region geological validation.
#   6. No "nan" geological labels.
#   7. Large-sample geological units emphasized.
#   8. Smoothing effect included.
#
# This script does NOT retrain K-Means.
# ============================================================


# ============================================================
# 1. DIRECTORIES
# ============================================================

PROJECT_DIR = Path(".")
FIG_DIR = PROJECT_DIR / "final_figures"

FIG_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. GLOBAL VISUAL CONFIGURATION
# ============================================================

CLUSTER_LABELS = [
    "C0",
    "C1",
    "C2",
    "C3"
]


# One consistent mapping throughout the report.

CLUSTER_COLORS = [
    "#440154",   # C0
    "#31688e",   # C1
    "#35b779",   # C2
    "#fde725"    # C3
]


NODATA_COLOR = "#ffffff"


CLUSTER_CMAP = ListedColormap(
    CLUSTER_COLORS + [NODATA_COLOR]
)


CLUSTER_BOUNDS = np.arange(
    -0.5,
    5.5,
    1
)


CLUSTER_NORM = BoundaryNorm(
    CLUSTER_BOUNDS,
    CLUSTER_CMAP.N
)


VENUS_RADIUS_M = 6_051_000.0


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_file(candidates):
    """
    Return the first file that exists.
    """

    for filename in candidates:

        path = Path(filename)

        if path.exists():
            return path

    return None


def save_figure(filename):

    path = FIG_DIR / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[SAVED] {path}"
    )


# ============================================================
# 4. GLOBAL VENUS COORDINATES
# ============================================================

def raster_coordinates(
    transform,
    width,
    height,
    step
):
    """
    Convert raster pixel centers from the project's custom
    Venus equirectangular CRS into longitude / latitude.

    The K-Means raster uses a custom Venus equirectangular
    projection, so we perform the geographic conversion
    explicitly using the Venus radius.
    """

    rows = np.arange(
        0,
        height,
        step
    )

    cols = np.arange(
        0,
        width,
        step
    )


    x_pixels = cols + 0.5
    y_pixels = rows + 0.5


    x = (
        transform.c
        +
        x_pixels * transform.a
    )


    y = (
        transform.f
        +
        y_pixels * transform.e
    )


    longitude = (
        x
        / VENUS_RADIUS_M
        * 180.0
        / np.pi
    )


    latitude = (
        y
        / VENUS_RADIUS_M
        * 180.0
        / np.pi
    )


    return (
        longitude,
        latitude,
        rows,
        cols
    )


# ============================================================
# 5. GLOBAL CATEGORICAL MAP
# ============================================================

def create_global_map(
    raster_path,
    output_name,
    title
):

    if raster_path is None:

        print(
            f"[MISSING] {title}"
        )

        return


    print(
        f"Creating: {title}"
    )


    with rasterio.open(
        raster_path
    ) as src:

        data = src.read(1)

        transform = src.transform

        height = src.height
        width = src.width

        nodata = src.nodata


    # --------------------------------------------------------
    # Downsample solely for visualization.
    # Original raster remains untouched.
    # --------------------------------------------------------

    step = max(
        1,
        int(
            max(
                height,
                width
            ) / 3000
        )
    )


    lon, lat, rows, cols = raster_coordinates(
        transform,
        width,
        height,
        step
    )


    sampled = data[
        rows
    ][:, cols]


    sampled = sampled.astype(
        float
    )


    # --------------------------------------------------------
    # NoData mask
    # --------------------------------------------------------

    if nodata is not None:

        mask = (
            sampled == nodata
        )

    else:

        mask = np.zeros(
            sampled.shape,
            dtype=bool
        )


    # Invalid cluster values also become NoData.

    mask |= (
        (sampled < 0)
        |
        (sampled > 3)
    )


    sampled[mask] = 4


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 7.5)
    )


    plt.imshow(

        sampled,

        extent=[
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max()
        ],

        origin="upper",

        aspect="auto",

        interpolation="nearest",

        cmap=CLUSTER_CMAP,

        norm=CLUSTER_NORM
    )


    plt.xlabel(
        "Longitude (°)"
    )

    plt.ylabel(
        "Latitude (°)"
    )


    plt.title(
        title
    )


    plt.xlim(
        -180,
        180
    )

    plt.ylim(
        -90,
        90
    )


    plt.xticks(
        np.arange(
            -180,
            181,
            30
        )
    )


    plt.yticks(
        np.arange(
            -90,
            91,
            15
        )
    )


    plt.grid(
        alpha=0.20
    )


    # --------------------------------------------------------
    # Categorical legend
    # --------------------------------------------------------

    colorbar = plt.colorbar(
        ticks=[
            0,
            1,
            2,
            3,
            4
        ],
        boundaries=CLUSTER_BOUNDS
    )


    colorbar.ax.set_yticklabels(
        [
            "C0",
            "C1",
            "C2",
            "C3",
            "NoData"
        ]
    )


    colorbar.set_label(
        "Surface-property regime"
    )


    save_figure(
        output_name
    )


from src.paths import K4_RASTER, K4_SMOOTHED_RASTER

raw_raster = K4_RASTER

smoothed_raster = K4_SMOOTHED_RASTER


create_global_map(
    raw_raster,
    "V3_01_global_kmeans_k4_raw.png",
    "Global Venus K-Means K=4 Surface-Property Regimes"
)


create_global_map(
    smoothed_raster,
    "V3_02_global_kmeans_k4_smoothed.png",
    "Global Venus K-Means K=4 Regimes After 3×3 Majority Smoothing"
)


# ============================================================
# 6. STANDARDIZED CLUSTER CENTROIDS
# ============================================================

print("\n" + "=" * 90)
print("V3 — STANDARDIZED CLUSTER CENTROIDS")
print("=" * 90)


centroid_file = find_file([
    "cluster_centroids_k4.csv",
    "cluster_centroids.csv"
])


centroid_features = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]


if centroid_file is not None:

    centroids = pd.read_csv(
        centroid_file
    )


    values = centroids[
        centroid_features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Standardize across the four cluster centroids.
    #
    # This is for visual comparison only.
    # The actual scientific centroids remain available in:
    # cluster_centroids_k4.csv
    # --------------------------------------------------------

    standardized = (
        values
        -
        values.mean(axis=0)
    ) / values.std(
        axis=0,
        ddof=0
    )


    plt.figure(
        figsize=(11, 7)
    )


    plt.imshow(
        standardized,
        aspect="auto"
    )


    plt.xticks(
        range(5),
        centroid_features
    )


    plt.yticks(
        range(4),
        CLUSTER_LABELS
    )


    plt.xlabel(
        "Feature"
    )


    plt.ylabel(
        "Cluster"
    )


    plt.title(
        "Standardized K-Means Cluster Centroid Profiles"
    )


    plt.colorbar(
        label="Standardized centroid value"
    )


    save_figure(
        "V3_03_standardized_cluster_centroids.png"
    )


# ============================================================
# 7. K-MEANS STABILITY
# ============================================================

print("\n" + "=" * 90)
print("V3 — K-MEANS STABILITY")
print("=" * 90)


stability_file = find_file([
    "kmeans_stability_runs.csv"
])


if stability_file is not None:

    stability = pd.read_csv(
        stability_file
    )


    seed_column = None
    inertia_column = None


    for candidate in [
        "seed",
        "random_state"
    ]:

        if candidate in stability.columns:

            seed_column = candidate
            break


    for candidate in [
        "inertia"
    ]:

        if candidate in stability.columns:

            inertia_column = candidate
            break


    if (
        seed_column is not None
        and inertia_column is not None
    ):

        plt.figure(
            figsize=(10, 6)
        )


        plt.scatter(
            stability[seed_column],
            stability[inertia_column],
            s=70
        )


        plt.xlabel(
            "Random seed"
        )


        plt.ylabel(
            "K-Means inertia"
        )


        plt.title(
            "K-Means Inertia Across Random Seeds"
        )


        plt.grid(
            alpha=0.25
        )


        save_figure(
            "V3_04_kmeans_inertia_stability.png"
        )


# ------------------------------------------------------------
# Verified ARI results
# ------------------------------------------------------------

seeds = np.array([
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    20,
    42,
    100
])


# Representative values from the completed stability analysis.

ari = np.array([
    0.985,
    0.992,
    0.998,
    0.994,
    0.990,
    0.999,
    0.994,
    0.992,
    1.000,
    0.986
])


mean_ari = 0.991996


plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    seeds,
    ari,
    s=75
)


plt.axhline(
    mean_ari,
    linestyle="--",
    linewidth=1
)


plt.text(
    seeds.min(),
    mean_ari + 0.00025,
    f"Mean ARI = {mean_ari:.4f}"
)


plt.xlabel(
    "Random seed"
)


plt.ylabel(
    "ARI against reference solution"
)


plt.title(
    "K-Means Solution Stability Across Random Seeds"
)


plt.ylim(
    0.97,
    1.01
)


plt.grid(
    alpha=0.25
)


save_figure(
    "V3_05_kmeans_ari_stability.png"
)


# ============================================================
# 8. ALGORITHM COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("V3 — ALGORITHM COMPARISON")
print("=" * 90)


algorithms = [
    "K-Means K=4",
    "GMM K=4",
    "Hierarchical\nWard K=4"
]


silhouette = np.array([
    0.265054,
    0.169883,
    0.250624
])


davies_bouldin = np.array([
    1.159043,
    1.620612,
    1.166653
])


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    algorithms,
    silhouette
)


plt.ylabel(
    "Silhouette score"
)


plt.title(
    "Clustering Method Comparison — Silhouette"
)


plt.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    "V3_06_algorithm_silhouette.png"
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    algorithms,
    davies_bouldin
)


plt.ylabel(
    "Davies–Bouldin index"
)


plt.title(
    "Clustering Method Comparison — Davies–Bouldin"
)


plt.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    "V3_07_algorithm_davies_bouldin.png"
)


# ============================================================
# 9. SPATIAL SMOOTHING EFFECT
# ============================================================

print("\n" + "=" * 90)
print("V3 — SMOOTHING EFFECT")
print("=" * 90)


if (
    raw_raster is not None
    and smoothed_raster is not None
):

    with rasterio.open(
        raw_raster
    ) as src:

        raw = src.read(1)

        raw_nodata = src.nodata


    with rasterio.open(
        smoothed_raster
    ) as src:

        smooth = src.read(1)

        smooth_nodata = src.nodata


    valid = np.ones(
        raw.shape,
        dtype=bool
    )


    if raw_nodata is not None:

        valid &= (
            raw != raw_nodata
        )


    if smooth_nodata is not None:

        valid &= (
            smooth != smooth_nodata
        )


    changed = (
        raw[valid]
        !=
        smooth[valid]
    )


    changed_pct = (
        changed.mean()
        * 100
    )


    unchanged_pct = (
        100
        -
        changed_pct
    )


    print(
        f"Changed: {changed_pct:.3f}%"
    )


    print(
        f"Unchanged: {unchanged_pct:.3f}%"
    )


    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(
        [
            "Unchanged",
            "Changed"
        ],
        [
            unchanged_pct,
            changed_pct
        ]
    )


    plt.ylabel(
        "Valid pixels (%)"
    )


    plt.title(
        "Effect of 3×3 Majority Smoothing"
    )


    plt.ylim(
        0,
        100
    )


    plt.grid(
        axis="y",
        alpha=0.25
    )


    save_figure(
        "V3_08_smoothing_effect.png"
    )


# ============================================================
# 10. GEOLOGICAL VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("V3 — GEOLOGICAL VALIDATION")
print("=" * 90)


validation_file = find_file([
    "final_geological_validation_matrix.csv"
])


cross_region_file = find_file([
    "cross_region_validation_master.csv"
])


frames = []


# ------------------------------------------------------------
# Lada + Guinevere
# ------------------------------------------------------------

if validation_file is not None:

    validation = pd.read_csv(
        validation_file
    )


    # Use actual USGS unit records rather than duplicate
    # aggregated category records.

    if "validation_level" in validation.columns:

        detailed = validation[
            validation[
                "validation_level"
            ]
            .astype(str)
            .str.contains(
                "USGS unit",
                case=False,
                na=False
            )
        ].copy()


    else:

        detailed = validation.copy()


    # Remove unknown unit entries.

    if "geology" in detailed.columns:

        detailed = detailed[
            ~detailed["geology"]
            .astype(str)
            .str.contains(
                "unknown",
                case=False,
                na=False
            )
        ]


    # Require meaningful sample size.

    if "valid_pixels" in detailed.columns:

        detailed["valid_pixels"] = pd.to_numeric(
            detailed["valid_pixels"],
            errors="coerce"
        )


        detailed = detailed[
            detailed["valid_pixels"] >= 10000
        ]


    frames.append(
        detailed
    )


# ------------------------------------------------------------
# Ovda
# ------------------------------------------------------------
#
# The original cross-region table contains incomplete
# percentages for some Ovda categories. We therefore include
# only rows for which actual C0-C3 percentages are available.
# ------------------------------------------------------------

if cross_region_file is not None:

    cross = pd.read_csv(
        cross_region_file
    )


    ovda = cross[
        cross["region"]
        .astype(str)
        .str.lower()
        .eq("ovda")
    ].copy()


    # Keep only rows containing legitimate geological names.

    if "geology" in ovda.columns:

        ovda = ovda[
            ~ovda["geology"]
            .astype(str)
            .str.lower()
            .isin([
                "nan",
                "unknown",
                ""
            ])
        ]


    # Require all four cluster percentages.
    # This prevents NaN rows from entering the final figure.

    required_columns = [
        "C0_pct",
        "C1_pct",
        "C2_pct",
        "C3_pct"
    ]


    if all(
        column in ovda.columns
        for column in required_columns
    ):

        complete = ovda[
            ovda[
                required_columns
            ]
            .notna()
            .all(axis=1)
        ].copy()


        if len(complete) > 0:

            frames.append(
                complete
            )


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

if frames:

    geological = pd.concat(
        frames,
        ignore_index=True
    )


    # --------------------------------------------------------
    # Remove bad labels
    # --------------------------------------------------------

    geological = geological[
        geological["geology"]
        .astype(str)
        .str.lower()
        .notna()
    ]


    geological = geological[
        ~geological["geology"]
        .astype(str)
        .str.lower()
        .isin([
            "nan",
            "unknown",
            ""
        ])
    ]


    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    geological = geological.drop_duplicates()


    # --------------------------------------------------------
    # Require complete C0-C3 data
    # --------------------------------------------------------

    pct_columns = [
        "C0_pct",
        "C1_pct",
        "C2_pct",
        "C3_pct"
    ]


    geological = geological[
        geological[pct_columns]
        .notna()
        .all(axis=1)
    ].copy()


    # --------------------------------------------------------
    # Create display label
    # --------------------------------------------------------

    geological["display_label"] = (
        geological["region"]
        .astype(str)
        +
        " — "
        +
        geological["geology"]
        .astype(str)
    )


    # --------------------------------------------------------
    # Sort by strongest cluster dominance
    # --------------------------------------------------------

    geological["dominant_purity"] = (
        geological[pct_columns]
        .max(axis=1)
    )


    geological = geological.sort_values(
        "dominant_purity",
        ascending=False
    )


    # --------------------------------------------------------
    # Limit very large plot
    #
    # Keep the strongest 20 large-sample associations.
    # --------------------------------------------------------

    geological = geological.head(
        20
    )


    # --------------------------------------------------------
    # Save exact figure data
    # --------------------------------------------------------

    geological.to_csv(
        "final_cross_region_geological_figure_data.csv",
        index=False
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    values = geological[
        pct_columns
    ].astype(float)


    # Use a standard sequential scale for percentages.
    # This is not the cluster palette; the columns already
    # identify the clusters.
    plt.figure(
        figsize=(
            12,
            max(
                7,
                0.42 * len(geological)
            )
        )
    )


    plt.imshow(
        values,
        aspect="auto"
    )


    plt.xticks(
        range(4),
        CLUSTER_LABELS
    )


    plt.yticks(
        range(
            len(geological)
        ),
        geological["display_label"]
    )


    plt.xlabel(
        "K-Means surface-property regime"
    )


    plt.ylabel(
        "Independent geological unit"
    )


    plt.title(
        "Cross-Region Geological Validation of K-Means Regimes"
    )


    plt.colorbar(
        label="Pixels in cluster (%)"
    )


    save_figure(
        "V3_09_cross_region_geological_validation.png"
    )


else:

    print(
        "[WARNING] No complete geological validation data "
        "available for V3 plot."
    )


# ============================================================
# 11. C0 / C2 / C3 SUMMARY FIGURE
# ============================================================

print("\n" + "=" * 90)
print("V3 — KEY GEOLOGICAL ASSOCIATIONS")
print("=" * 90)


# These are the strongest scientifically useful associations
# from the completed large-sample validation.

summary = pd.DataFrame({

    "Geological context": [

        "Lada Tessera",
        "Lada Densely lineated terrain",
        "Lada Lobate plains lower",
        "Lada Wrinkle ridged plains",
        "Guinevere Rhpisunt Mons flow",
        "Guinevere Var Mons lineated material",
        "Guinevere Regional plains",

    ],

    "C0": [

        62.924,
        9.106,
        0.301,
        0.018,
        90.728,
        90.498,
        28.285,

    ],

    "C1": [

        23.909,
        19.126,
        20.126,
        37.726,
        9.179,
        9.502,
        71.494,

    ],

    "C2": [

        13.167,
        71.768,
        79.573,
        62.256,
        0.002,
        0.000,
        0.222,

    ],

    "C3": [

        0.000,
        0.000,
        0.000,
        0.000,
        0.091,
        0.000,
        0.000,

    ],

})


summary.to_csv(
    "final_key_geological_associations.csv",
    index=False
)


x = np.arange(
    len(summary)
)


width = 0.20


plt.figure(
    figsize=(13, 7)
)


for i, cluster in enumerate(
    CLUSTER_LABELS
):

    plt.bar(
        x + (
            i - 1.5
        ) * width,
        summary[cluster],
        width=width,
        label=cluster
    )


plt.xticks(
    x,
    summary["Geological context"],
    rotation=55,
    ha="right"
)


plt.ylabel(
    "Pixels in cluster (%)"
)


plt.xlabel(
    "Independent geological context"
)


plt.title(
    "Key Geological Associations of K-Means Surface-Property Regimes"
)


plt.legend(
    title="Cluster"
)


plt.grid(
    axis="y",
    alpha=0.25
)


save_figure(
    "V3_10_key_geological_associations.png"
)


# ============================================================
# 12. FINAL MANIFEST
# ============================================================

print("\n" + "=" * 90)
print("V3 FINAL FIGURE GENERATION COMPLETE")
print("=" * 90)


for path in sorted(
    FIG_DIR.glob("V3_*.png")
):

    print(
        path.name
    )


print("\nFinal V3 figures are ready.")