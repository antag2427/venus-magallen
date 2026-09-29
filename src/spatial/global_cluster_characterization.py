import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER
from rasterio.enums import Resampling
from scipy import ndimage


# ============================================================
# CONFIGURATION
# ============================================================

CLUSTER_RASTER = str(K4_RASTER)

VENUS_RADIUS = 6051000.0

NODATA = 255

CLUSTERS = [0, 1, 2, 3]

LATITUDE_BAND_SIZE = 10
LONGITUDE_BAND_SIZE = 10

CHUNK_ROWS = 512

# Used only for coarse connected-component analysis
COMPONENT_DOWNSAMPLE = 16

CONNECTIVITY = np.ones(
    (3, 3),
    dtype=np.uint8
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pixel_center_latitudes(
    row_start,
    row_end,
    transform
):
    """
    Convert raster row centers to Venus latitude.

    The raster uses:

        y = R * latitude_radians

    Therefore:

        latitude_radians = y / R
    """

    rows = np.arange(
        row_start,
        row_end
    )

    ys = (
        transform.f +
        transform.e *
        (rows + 0.5)
    )

    latitudes = np.rad2deg(
        ys / VENUS_RADIUS
    )

    return latitudes


def pixel_center_longitudes(
    width,
    transform
):
    """
    Convert raster column centers to longitude.
    """

    cols = np.arange(width)

    xs = (
        transform.c +
        transform.a *
        (cols + 0.5)
    )

    longitudes = np.rad2deg(
        xs / VENUS_RADIUS
    )

    # Wrap into [-180, 180)
    longitudes = (
        (longitudes + 180.0)
        % 360.0
    ) - 180.0

    return longitudes


def spherical_row_areas(
    latitudes,
    transform
):
    """
    Calculate the physical area of one raster pixel
    in each row.

    For an equirectangular raster:

        A = RÂ² * Î”Î» *
            [sin(Ï†â‚‚) - sin(Ï†â‚)]

    where Ï† is latitude in radians.

    IMPORTANT:
    The projected coordinate values are already
    R * angle_in_radians, so we do NOT apply another
    degree-to-radian conversion to the pixel size.
    """

    # Projected y boundaries
    y_center = (
        np.deg2rad(latitudes) *
        VENUS_RADIUS
    )

    half_height = (
        abs(transform.e) / 2.0
    )

    y1 = y_center - half_height
    y2 = y_center + half_height

    phi1 = y1 / VENUS_RADIUS
    phi2 = y2 / VENUS_RADIUS

    # Projected x pixel width
    delta_lambda = (
        abs(transform.a) /
        VENUS_RADIUS
    )

    areas = (
        VENUS_RADIUS ** 2
        * delta_lambda
        * (
            np.sin(phi2) -
            np.sin(phi1)
        )
    )

    return np.abs(areas)


# ============================================================
# OPEN RASTER
# ============================================================

print("=" * 110)
print("CORRECTED GLOBAL K-MEANS K=4 GEOGRAPHIC PROFILE")
print("=" * 110)

with rasterio.open(
    CLUSTER_RASTER
) as src:

    width = src.width
    height = src.height

    transform = src.transform
    crs = src.crs

    print("\nRaster information")
    print("-" * 70)

    print(f"Width:          {width:,}")
    print(f"Height:         {height:,}")
    print(
        f"Total pixels:   "
        f"{width * height:,}"
    )

    print(f"Resolution:     {src.res}")

    print(f"NoData:         {src.nodata}")

    print(f"Venus radius:   {VENUS_RADIUS:,.0f} m")


# ============================================================
# THEORETICAL VENUS SURFACE AREA
# ============================================================

theoretical_venus_area = (
    4.0 *
    np.pi *
    VENUS_RADIUS ** 2
)

print("\n")
print("=" * 110)
print("THEORETICAL VENUS SURFACE AREA")
print("=" * 110)

print(
    f"\n4Ï€RÂ² = "
    f"{theoretical_venus_area / 1e6:,.2f} kmÂ²"
)


# ============================================================
# PRECOMPUTE LONGITUDES
# ============================================================

longitudes = pixel_center_longitudes(
    width,
    transform
)


# ============================================================
# HISTOGRAM EDGES
# ============================================================

lat_edges = np.arange(
    -90,
    90 + LATITUDE_BAND_SIZE,
    LATITUDE_BAND_SIZE
)

lon_edges = np.arange(
    -180,
    180 + LONGITUDE_BAND_SIZE,
    LONGITUDE_BAND_SIZE
)


latitude_histograms = {
    cluster: np.zeros(
        len(lat_edges) - 1,
        dtype=np.int64
    )
    for cluster in CLUSTERS
}


longitude_histograms = {
    cluster: np.zeros(
        len(lon_edges) - 1,
        dtype=np.int64
    )
    for cluster in CLUSTERS
}


# ============================================================
# GLOBAL ACCUMULATORS
# ============================================================

pixel_counts = {
    cluster: 0
    for cluster in CLUSTERS
}

area_m2 = {
    cluster: 0.0
    for cluster in CLUSTERS
}

latitude_weighted_sum = {
    cluster: 0.0
    for cluster in CLUSTERS
}

latitude_squared_weighted_sum = {
    cluster: 0.0
    for cluster in CLUSTERS
}

# Circular longitude statistics
longitude_sin_sum = {
    cluster: 0.0
    for cluster in CLUSTERS
}

longitude_cos_sum = {
    cluster: 0.0
    for cluster in CLUSTERS
}


# Geographic extrema
min_latitude = {
    cluster: np.inf
    for cluster in CLUSTERS
}

max_latitude = {
    cluster: -np.inf
    for cluster in CLUSTERS
}


# ============================================================
# EXACT GLOBAL SCAN
# ============================================================

print("\n")
print("=" * 110)
print("SCANNING FULL CLASSIFICATION")
print("=" * 110)

with rasterio.open(
    CLUSTER_RASTER
) as src:

    for row_start in range(
        0,
        height,
        CHUNK_ROWS
    ):

        row_end = min(
            row_start + CHUNK_ROWS,
            height
        )

        window = rasterio.windows.Window(
            0,
            row_start,
            width,
            row_end - row_start
        )

        data = src.read(
            1,
            window=window
        )

        latitudes = pixel_center_latitudes(
            row_start,
            row_end,
            transform
        )

        row_areas = spherical_row_areas(
            latitudes,
            transform
        )

        for cluster_id in CLUSTERS:

            mask = (
                data == cluster_id
            )

            if not np.any(mask):
                continue

            row_indices, col_indices = np.where(
                mask
            )

            pixel_lats = latitudes[
                row_indices
            ]

            pixel_lons = longitudes[
                col_indices
            ]

            pixel_areas = row_areas[
                row_indices
            ]

            count = len(
                pixel_lats
            )

            pixel_counts[
                cluster_id
            ] += count

            total_area = np.sum(
                pixel_areas
            )

            area_m2[
                cluster_id
            ] += total_area

            # ------------------------------------------------
            # Latitude
            # ------------------------------------------------

            latitude_weighted_sum[
                cluster_id
            ] += np.sum(
                pixel_lats *
                pixel_areas
            )

            latitude_squared_weighted_sum[
                cluster_id
            ] += np.sum(
                (pixel_lats ** 2) *
                pixel_areas
            )

            # ------------------------------------------------
            # Longitude
            #
            # Circular statistics:
            #
            # mean angle = atan2(mean(sin Î¸),
            #                    mean(cos Î¸))
            # ------------------------------------------------

            lon_rad = np.deg2rad(
                pixel_lons
            )

            longitude_sin_sum[
                cluster_id
            ] += np.sum(
                np.sin(lon_rad) *
                pixel_areas
            )

            longitude_cos_sum[
                cluster_id
            ] += np.sum(
                np.cos(lon_rad) *
                pixel_areas
            )

            # ------------------------------------------------
            # Geographic extrema
            # ------------------------------------------------

            local_min = np.min(
                pixel_lats
            )

            local_max = np.max(
                pixel_lats
            )

            if local_min < min_latitude[
                cluster_id
            ]:

                min_latitude[
                    cluster_id
                ] = local_min

            if local_max > max_latitude[
                cluster_id
            ]:

                max_latitude[
                    cluster_id
                ] = local_max

            # ------------------------------------------------
            # Latitude histogram
            # ------------------------------------------------

            lat_hist, _ = np.histogram(
                pixel_lats,
                bins=lat_edges
            )

            latitude_histograms[
                cluster_id
            ] += lat_hist

            # ------------------------------------------------
            # Longitude histogram
            # ------------------------------------------------

            lon_hist, _ = np.histogram(
                pixel_lons,
                bins=lon_edges
            )

            longitude_histograms[
                cluster_id
            ] += lon_hist

        if (
            row_start %
            (CHUNK_ROWS * 10)
            == 0
        ):

            print(
                f"Processed rows "
                f"{row_start:,} - "
                f"{row_end:,} / "
                f"{height:,}"
            )


# ============================================================
# TOTALS
# ============================================================

total_valid_pixels = sum(
    pixel_counts.values()
)

total_classified_area = sum(
    area_m2.values()
)

total_valid_percentage = (
    100.0 *
    total_valid_pixels /
    (width * height)
)

classified_surface_percentage = (
    100.0 *
    total_classified_area /
    theoretical_venus_area
)


# ============================================================
# GLOBAL CLUSTER SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("CORRECTED GLOBAL CLUSTER SUMMARY")
print("=" * 110)

print(
    f"\nValid classified pixels: "
    f"{total_valid_pixels:,}"
)

print(
    f"Valid pixel percentage: "
    f"{total_valid_percentage:.2f}%"
)

print(
    f"Classified physical area: "
    f"{total_classified_area / 1e6:,.2f} kmÂ²"
)

print(
    f"Classified Venus surface: "
    f"{classified_surface_percentage:.2f}%"
)


global_rows = []


for cluster_id in CLUSTERS:

    pixels = pixel_counts[
        cluster_id
    ]

    area = area_m2[
        cluster_id
    ]

    pixel_percentage = (
        100.0 *
        pixels /
        total_valid_pixels
    )

    valid_area_percentage = (
        100.0 *
        area /
        total_classified_area
    )

    whole_venus_percentage = (
        100.0 *
        area /
        theoretical_venus_area
    )

    mean_latitude = (
        latitude_weighted_sum[
            cluster_id
        ] /
        area
    )

    mean_latitude_squared = (
        latitude_squared_weighted_sum[
            cluster_id
        ] /
        area
    )

    latitude_variance = (
        mean_latitude_squared -
        mean_latitude ** 2
    )

    latitude_std = np.sqrt(
        max(
            latitude_variance,
            0.0
        )
    )

    # Circular longitude mean
    sin_mean = (
        longitude_sin_sum[
            cluster_id
        ] /
        area
    )

    cos_mean = (
        longitude_cos_sum[
            cluster_id
        ] /
        area
    )

    circular_longitude = np.rad2deg(
        np.arctan2(
            sin_mean,
            cos_mean
        )
    )

    # Resultant length:
    #
    # close to 1 = concentrated
    # close to 0 = globally distributed
    #

    longitude_resultant = np.sqrt(
        sin_mean ** 2 +
        cos_mean ** 2
    )

    print(
        f"\nCluster {cluster_id}"
    )

    print(
        f"  Pixels: "
        f"{pixels:,}"
    )

    print(
        f"  Pixel share of valid map: "
        f"{pixel_percentage:.2f}%"
    )

    print(
        f"  Physical area: "
        f"{area / 1e6:,.2f} kmÂ²"
    )

    print(
        f"  Share of classified area: "
        f"{valid_area_percentage:.2f}%"
    )

    print(
        f"  Share of theoretical Venus surface: "
        f"{whole_venus_percentage:.2f}%"
    )

    print(
        f"  Area-weighted mean latitude: "
        f"{mean_latitude:.3f}Â°"
    )

    print(
        f"  Latitude std: "
        f"{latitude_std:.3f}Â°"
    )

    print(
        f"  Latitude range: "
        f"{min_latitude[cluster_id]:.3f}Â° "
        f"to "
        f"{max_latitude[cluster_id]:.3f}Â°"
    )

    print(
        f"  Circular longitude mean: "
        f"{circular_longitude:.3f}Â°"
    )

    print(
        f"  Longitude concentration "
        f"(resultant length): "
        f"{longitude_resultant:.5f}"
    )

    global_rows.append({
        "cluster": cluster_id,
        "pixels": pixels,
        "pixel_percentage": pixel_percentage,
        "area_km2": area / 1e6,
        "classified_area_percentage":
            valid_area_percentage,
        "venus_surface_percentage":
            whole_venus_percentage,
        "mean_latitude":
            mean_latitude,
        "latitude_std":
            latitude_std,
        "min_latitude":
            min_latitude[cluster_id],
        "max_latitude":
            max_latitude[cluster_id],
        "circular_mean_longitude":
            circular_longitude,
        "longitude_resultant":
            longitude_resultant
    })


global_df = pd.DataFrame(
    global_rows
)

global_df.to_csv(
    "global_cluster_summary_corrected.csv",
    index=False
)


# ============================================================
# LATITUDE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 110)
print("LATITUDE-BAND DISTRIBUTION")
print("=" * 110)

latitude_rows = []

for i in range(
    len(lat_edges) - 1
):

    lat_min = lat_edges[i]
    lat_max = lat_edges[i + 1]

    band_counts = {
        cluster: latitude_histograms[
            cluster
        ][i]
        for cluster in CLUSTERS
    }

    total_band = sum(
        band_counts.values()
    )

    if total_band == 0:
        continue

    row = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "total_pixels": total_band
    }

    for cluster_id in CLUSTERS:

        count = band_counts[
            cluster_id
        ]

        row[
            f"cluster_{cluster_id}_pixels"
        ] = count

        row[
            f"cluster_{cluster_id}_percentage"
        ] = (
            100.0 *
            count /
            total_band
        )

    latitude_rows.append(
        row
    )


latitude_df = pd.DataFrame(
    latitude_rows
)

latitude_df.to_csv(
    "global_cluster_latitude_corrected.csv",
    index=False
)


# ============================================================
# LONGITUDE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 110)
print("LONGITUDE-BAND DISTRIBUTION")
print("=" * 110)

longitude_rows = []

for i in range(
    len(lon_edges) - 1
):

    lon_min = lon_edges[i]
    lon_max = lon_edges[i + 1]

    band_counts = {
        cluster: longitude_histograms[
            cluster
        ][i]
        for cluster in CLUSTERS
    }

    total_band = sum(
        band_counts.values()
    )

    if total_band == 0:
        continue

    row = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "total_pixels": total_band
    }

    for cluster_id in CLUSTERS:

        count = band_counts[
            cluster_id
        ]

        row[
            f"cluster_{cluster_id}_pixels"
        ] = count

        row[
            f"cluster_{cluster_id}_percentage"
        ] = (
            100.0 *
            count /
            total_band
        )

    longitude_rows.append(
        row
    )


longitude_df = pd.DataFrame(
    longitude_rows
)

longitude_df.to_csv(
    "global_cluster_longitude_corrected.csv",
    index=False
)


# ============================================================
# COARSE SPATIAL COHERENCE
# ============================================================

print("\n")
print("=" * 110)
print("COARSE GLOBAL SPATIAL COHERENCE")
print("=" * 110)

with rasterio.open(
    CLUSTER_RASTER
) as src:

    small_height = max(
        1,
        src.height //
        COMPONENT_DOWNSAMPLE
    )

    small_width = max(
        1,
        src.width //
        COMPONENT_DOWNSAMPLE
    )

    small = src.read(
        1,
        out_shape=(
            small_height,
            small_width
        ),
        resampling=Resampling.mode
    )


component_rows = []


for cluster_id in CLUSTERS:

    print(
        f"\nCluster {cluster_id}"
    )

    mask = (
        small == cluster_id
    )

    labeled, num_components = (
        ndimage.label(
            mask,
            structure=CONNECTIVITY
        )
    )

    if num_components == 0:

        print(
            "  No components."
        )

        continue

    component_sizes = np.bincount(
        labeled.ravel()
    )[1:]

    component_sizes.sort()

    component_sizes = (
        component_sizes[::-1]
    )

    total_coarse_pixels = np.sum(
        mask
    )

    largest_component = (
        component_sizes[0]
    )

    largest_share = (
        100.0 *
        largest_component /
        total_coarse_pixels
    )

    median_component = np.median(
        component_sizes
    )

    print(
        f"  Components: "
        f"{num_components:,}"
    )

    print(
        f"  Largest component: "
        f"{largest_component:,} "
        f"coarse pixels"
    )

    print(
        f"  Largest component share: "
        f"{largest_share:.2f}%"
    )

    print(
        f"  Median component: "
        f"{median_component:.1f} "
        f"coarse pixels"
    )

    top_n = min(
        10,
        len(component_sizes)
    )

    for rank in range(top_n):

        size = component_sizes[
            rank
        ]

        print(
            f"    #{rank + 1}: "
            f"{size:,}"
        )

        component_rows.append({
            "cluster": cluster_id,
            "rank": rank + 1,
            "component_size_coarse_pixels":
                int(size)
        })


component_df = pd.DataFrame(
    component_rows
)

component_df.to_csv(
    "global_cluster_components_corrected.csv",
    index=False
)


# ============================================================
# PRINT LATITUDE TABLE
# ============================================================

print("\n")
print("=" * 110)
print("LATITUDE-BAND COMPOSITION")
print("=" * 110)

print(
    latitude_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.3f}"
    )
)


# ============================================================
# PRINT LONGITUDE TABLE
# ============================================================

print("\n")
print("=" * 110)
print("LONGITUDE-BAND COMPOSITION")
print("=" * 110)

print(
    longitude_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.3f}"
    )
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 110)
print("AREA CONSISTENCY CHECK")
print("=" * 110)

print(
    f"\nTheoretical Venus area: "
    f"{theoretical_venus_area / 1e6:,.2f} kmÂ²"
)

print(
    f"Classified raster area: "
    f"{total_classified_area / 1e6:,.2f} kmÂ²"
)

print(
    f"Classified fraction of Venus: "
    f"{classified_surface_percentage:.2f}%"
)

print(
    "\nCluster areas should sum to the "
    "classified raster area."
)

print(
    f"Cluster-area sum: "
    f"{sum(area_m2.values()) / 1e6:,.2f} kmÂ²"
)


# ============================================================
# FINAL FILES
# ============================================================

print("\n")
print("=" * 110)
print("CORRECTED GLOBAL CHARACTERIZATION COMPLETE")
print("=" * 110)

print("\nFiles created:")

print(
    "1. global_cluster_summary_corrected.csv"
)

print(
    "2. global_cluster_latitude_corrected.csv"
)

print(
    "3. global_cluster_longitude_corrected.csv"
)

print(
    "4. global_cluster_components_corrected.csv"
)

print("\nImportant:")

print(
    "Physical area now uses the correct spherical "
    "cell-area equation for the Equirectangular Venus grid."
)

print(
    "Longitude location uses circular statistics, "
    "not an ordinary arithmetic longitude mean."
)

print(
    "The original K-Means classification raster "
    "is not modified."
)

print(
    "Global area percentages use spherical surface-area "
    "weighting for the equirectangular Venus raster."
)

print(
    "Connected-component statistics are based on a "
    "16x downsampled classification and are therefore "
    "coarse spatial-coherence indicators."
)
