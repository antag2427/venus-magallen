import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER


# ============================================================
# CONFIGURATION
# ============================================================

CLUSTER_RASTER = str(K4_RASTER)

VENUS_RADIUS = 6051000.0

CLUSTERS = [0, 1, 2, 3]

CHUNK_ROWS = 256


# ============================================================
# REGIONS
# ============================================================
#
# Longitudes are supplied in the traditional Venus
# 0â€“360 degree east-positive convention used by the maps.
#
# The raster itself uses -180 to +180 internally.
#
# IMPORTANT:
# These are GEOLOGICAL CONTEXT REGIONS / QUADRANGLE WINDOWS.
# They are NOT treated as pure geological units.
#
# Their descriptions are based on published geological maps.
# ============================================================

REGIONS = [

    {
        "name": "Ovda Regio",
        "quadrangle": "V-35",
        "lon_min": 90.0,
        "lon_max": 120.0,
        "lat_min": -25.0,
        "lat_max": 0.0,
        "context": (
            "Mixed highland and lowland region; "
            "northern part includes high-standing Aphrodite "
            "Terra/Ovda-Thetis plateau and southern part "
            "contains lower plains."
        ),
        "context_class": "mixed_highland_lowland"
    },

    {
        "name": "Niobe Planitia",
        "quadrangle": "V-23",
        "lon_min": 90.0,
        "lon_max": 120.0,
        "lat_min": 0.0,
        "lat_max": 25.0,
        "context": (
            "Highland-to-lowland transition; southern margin "
            "contains Ovda Regio and Haasttse-baad Tessera, "
            "with Niobe/Sogolon lowlands to the north."
        ),
        "context_class": "highland_lowland_transition"
    },

    {
        "name": "Aino Planitia",
        "quadrangle": "V-46",
        "lon_min": 60.0,
        "lon_max": 90.0,
        "lat_min": -50.0,
        "lat_max": -25.0,
        "context": (
            "Southern lowland dominated by volcanic plains, "
            "with volcanoes and coronae."
        ),
        "context_class": "lowland_volcanic"
    },

    {
        "name": "Lavinia Planitia",
        "quadrangle": "V-55",
        "lon_min": 330.0,
        "lon_max": 360.0,
        "lat_min": -50.0,
        "lat_max": -25.0,
        "context": (
            "Large southern lowland/basin dominated by "
            "volcanic deposits."
        ),
        "context_class": "lowland_basin"
    },

    {
        "name": "Guinevere Planitia",
        "quadrangle": "V-30",
        "lon_min": 300.0,
        "lon_max": 330.0,
        "lat_min": 0.0,
        "lat_max": 25.0,
        "context": (
            "Northern lowland dominated by plains and "
            "volcanic flow materials, with smaller "
            "upland/tessera exposures."
        ),
        "context_class": "lowland_volcanic"
    },

    {
        "name": "Metis Mons",
        "quadrangle": "V-6",
        "lon_min": 240.0,
        "lon_max": 300.0,
        "lat_min": 50.0,
        "lat_max": 75.0,
        "context": (
            "Mixed northern region containing plains, "
            "tesserae, coronae, volcanoes, and structural belts."
        ),
        "context_class": "mixed"
    }

]


# ============================================================
# CONVERT 0-360 LONGITUDE TO -180/+180
# ============================================================

def normalize_longitude(lon):

    return ((lon + 180.0) % 360.0) - 180.0


# ============================================================
# SPHERICAL AREA
# ============================================================

def pixel_row_area(
    latitudes,
    transform
):
    """
    Physical area of one raster pixel for each latitude row.

    For an equirectangular projection:

        A = RÂ² Î”Î»
            [sin(phi2) - sin(phi1)]

    Projected pixel dimensions are already expressed as
    distances on the Venus equirectangular grid.
    """

    # Center y coordinates in metres
    y_center = (
        np.deg2rad(latitudes)
        * VENUS_RADIUS
    )

    half_height = (
        abs(transform.e) / 2.0
    )

    y1 = y_center - half_height
    y2 = y_center + half_height

    phi1 = (
        y1 / VENUS_RADIUS
    )

    phi2 = (
        y2 / VENUS_RADIUS
    )

    delta_lambda = (
        abs(transform.a)
        / VENUS_RADIUS
    )

    area = (
        VENUS_RADIUS ** 2
        * delta_lambda
        * (
            np.sin(phi2)
            - np.sin(phi1)
        )
    )

    return np.abs(area)


# ============================================================
# REGION MASK CALCULATION
# ============================================================

def process_region(
    src,
    region,
    longitudes
):

    # Convert longitude convention
    lon_min = normalize_longitude(
        region["lon_min"]
    )

    lon_max = normalize_longitude(
        region["lon_max"]
    )

    # --------------------------------------------------------
    # Determine longitude mask
    # --------------------------------------------------------

    if region["lon_max"] >= 180:

        # Example:
        # 330â€“360 becomes -30â€“0
        lon_mask = (
            (longitudes >= lon_min)
            &
            (longitudes <= lon_max)
        )

    else:

        lon_mask = (
            (longitudes >= lon_min)
            &
            (longitudes <= lon_max)
        )

    col_indices = np.where(
        lon_mask
    )[0]

    if len(col_indices) == 0:

        raise RuntimeError(
            f"No longitude columns found for "
            f"{region['name']}"
        )

    col_start = col_indices.min()
    col_end = col_indices.max() + 1

    # --------------------------------------------------------
    # Latitude rows
    # --------------------------------------------------------

    # Raster y direction may be negative.
    #
    # We calculate center latitude for every row and
    # select the requested geographic interval.

    rows = np.arange(
        src.height
    )

    ys = (
        src.transform.f
        + src.transform.e
        * (rows + 0.5)
    )

    latitudes = np.rad2deg(
        ys / VENUS_RADIUS
    )

    row_mask = (
        (latitudes >= region["lat_min"])
        &
        (latitudes <= region["lat_max"])
    )

    row_indices = np.where(
        row_mask
    )[0]

    if len(row_indices) == 0:

        raise RuntimeError(
            f"No latitude rows found for "
            f"{region['name']}"
        )

    row_start = row_indices.min()
    row_end = row_indices.max() + 1

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    pixel_counts = {
        cluster: 0
        for cluster in CLUSTERS
    }

    physical_areas = {
        cluster: 0.0
        for cluster in CLUSTERS
    }

    # --------------------------------------------------------
    # Process by chunks
    # --------------------------------------------------------

    for chunk_start in range(
        row_start,
        row_end,
        CHUNK_ROWS
    ):

        chunk_end = min(
            chunk_start + CHUNK_ROWS,
            row_end
        )

        window = rasterio.windows.Window(
            col_start,
            chunk_start,
            col_end - col_start,
            chunk_end - chunk_start
        )

        data = src.read(
            1,
            window=window
        )

        # Latitude for this chunk
        chunk_rows = np.arange(
            chunk_start,
            chunk_end
        )

        chunk_ys = (
            src.transform.f
            + src.transform.e
            * (chunk_rows + 0.5)
        )

        chunk_lats = np.rad2deg(
            chunk_ys / VENUS_RADIUS
        )

        # Exact requested longitude mask
        chunk_longitudes = longitudes[
            col_start:col_end
        ]

        if region["lon_max"] >= 180:

            valid_lon_mask = (
                (chunk_longitudes >= lon_min)
                &
                (chunk_longitudes <= lon_max)
            )

        else:

            valid_lon_mask = (
                (chunk_longitudes >= lon_min)
                &
                (chunk_longitudes <= lon_max)
            )

        # Also ensure requested latitude
        valid_lat_mask = (
            (chunk_lats >= region["lat_min"])
            &
            (chunk_lats <= region["lat_max"])
        )

        # ----------------------------------------------------
        # Build 2-D geographic mask
        # ----------------------------------------------------

        geographic_mask = (
            valid_lat_mask[:, None]
            &
            valid_lon_mask[None, :]
        )

        # ----------------------------------------------------
        # Physical area by row
        # ----------------------------------------------------

        areas = pixel_row_area(
            chunk_lats,
            src.transform
        )

        # ----------------------------------------------------
        # Cluster statistics
        # ----------------------------------------------------

        for cluster_id in CLUSTERS:

            cluster_mask = (
                data == cluster_id
            ) & geographic_mask

            if not np.any(cluster_mask):

                continue

            rows_local, cols_local = (
                np.where(cluster_mask)
            )

            count = len(
                rows_local
            )

            pixel_counts[
                cluster_id
            ] += count

            physical_areas[
                cluster_id
            ] += np.sum(
                areas[rows_local]
            )

    total_pixels = sum(
        pixel_counts.values()
    )

    total_area = sum(
        physical_areas.values()
    )

    # --------------------------------------------------------
    # Convert to percentages
    # --------------------------------------------------------

    result = {
        "region": region["name"],
        "quadrangle": region["quadrangle"],
        "context_class": region["context_class"],
        "context_description": region["context"],
        "total_valid_pixels": total_pixels,
        "total_area_km2": (
            total_area / 1e6
        )
    }

    for cluster_id in CLUSTERS:

        count = pixel_counts[
            cluster_id
        ]

        area = physical_areas[
            cluster_id
        ]

        if total_pixels > 0:

            pixel_percentage = (
                100.0
                * count
                / total_pixels
            )

        else:

            pixel_percentage = np.nan

        if total_area > 0:

            area_percentage = (
                100.0
                * area
                / total_area
            )

        else:

            area_percentage = np.nan

        result[
            f"cluster_{cluster_id}_pixels"
        ] = count

        result[
            f"cluster_{cluster_id}_pixel_percentage"
        ] = pixel_percentage

        result[
            f"cluster_{cluster_id}_area_km2"
        ] = area / 1e6

        result[
            f"cluster_{cluster_id}_area_percentage"
        ] = area_percentage

    return result


# ============================================================
# OPEN RASTER
# ============================================================

print("=" * 110)
print("GLOBAL REGIONAL GEOLOGICAL CROSS-VALIDATION")
print("=" * 110)

print("\nOpening K=4 classification...")

with rasterio.open(
    CLUSTER_RASTER
) as src:

    print(
        f"Raster: "
        f"{src.width:,} x "
        f"{src.height:,}"
    )

    print(
        f"Resolution: "
        f"{src.res}"
    )

    print(
        f"NoData: "
        f"{src.nodata}"
    )

    # --------------------------------------------------------
    # Pixel-center longitudes
    # --------------------------------------------------------

    columns = np.arange(
        src.width
    )

    xs = (
        src.transform.c
        + src.transform.a
        * (columns + 0.5)
    )

    longitudes = np.rad2deg(
        xs / VENUS_RADIUS
    )

    longitudes = (
        (longitudes + 180.0)
        % 360.0
    ) - 180.0

    # --------------------------------------------------------
    # Process all regions
    # --------------------------------------------------------

    results = []

    for region in REGIONS:

        print("\n")
        print("-" * 110)

        print(
            f"Processing: "
            f"{region['name']} "
            f"({region['quadrangle']})"
        )

        print(
            f"Longitude: "
            f"{region['lon_min']}â€“"
            f"{region['lon_max']}Â°E"
        )

        print(
            f"Latitude: "
            f"{region['lat_min']}â€“"
            f"{region['lat_max']}Â°"
        )

        print(
            f"Context: "
            f"{region['context_class']}"
        )

        result = process_region(
            src,
            region,
            longitudes
        )

        results.append(
            result
        )

        print(
            f"Valid pixels: "
            f"{result['total_valid_pixels']:,}"
        )

        print(
            f"Physical area: "
            f"{result['total_area_km2']:,.2f} kmÂ²"
        )

        for cluster_id in CLUSTERS:

            print(
                f"  C{cluster_id}: "
                f"{result[f'cluster_{cluster_id}_area_percentage']:.2f}%"
            )


# ============================================================
# RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# GLOBAL REFERENCE
# ============================================================

print("\n")
print("=" * 110)
print("GLOBAL BASELINE")
print("=" * 110)

global_baseline = {
    0: 24.03,
    1: 58.11,
    2: 9.17,
    3: 1.55
}

print(
    "\nReference percentages are the corrected "
    "global physical-area percentages from the "
    "previous analysis."
)

for cluster_id in CLUSTERS:

    print(
        f"C{cluster_id}: "
        f"{global_baseline[cluster_id]:.2f}%"
    )


# ============================================================
# CALCULATE REGIONAL ENRICHMENT
# ============================================================

for cluster_id in CLUSTERS:

    results_df[
        f"cluster_{cluster_id}_enrichment"
    ] = (
        results_df[
            f"cluster_{cluster_id}_area_percentage"
        ]
        /
        global_baseline[cluster_id]
    )


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n")
print("=" * 110)
print("REGIONAL CLUSTER COMPOSITION")
print("=" * 110)

display_columns = [
    "region",
    "quadrangle",
    "context_class",
    "total_area_km2"
]

for cluster_id in CLUSTERS:

    display_columns.append(
        f"cluster_{cluster_id}_area_percentage"
    )

    display_columns.append(
        f"cluster_{cluster_id}_enrichment"
    )


print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.3f}"
    )
)


# ============================================================
# FIND C3-RICH REGIONS
# ============================================================

print("\n")
print("=" * 110)
print("C3 ENRICHMENT BY REGION")
print("=" * 110)

c3_sorted = results_df.sort_values(
    "cluster_3_enrichment",
    ascending=False
)

for _, row in c3_sorted.iterrows():

    print(
        f"{row['region']} "
        f"({row['quadrangle']}): "
        f"C3 = "
        f"{row['cluster_3_area_percentage']:.2f}% "
        f"â†’ "
        f"{row['cluster_3_enrichment']:.2f}x "
        f"global baseline"
    )


# ============================================================
# FIND C2-RICH REGIONS
# ============================================================

print("\n")
print("=" * 110)
print("C2 ENRICHMENT BY REGION")
print("=" * 110)

c2_sorted = results_df.sort_values(
    "cluster_2_enrichment",
    ascending=False
)

for _, row in c2_sorted.iterrows():

    print(
        f"{row['region']} "
        f"({row['quadrangle']}): "
        f"C2 = "
        f"{row['cluster_2_area_percentage']:.2f}% "
        f"â†’ "
        f"{row['cluster_2_enrichment']:.2f}x "
        f"global baseline"
    )


# ============================================================
# CONTEXT-CLASS SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("SUMMARY BY GEOLOGICAL CONTEXT CLASS")
print("=" * 110)

context_rows = []

for context_class, group in results_df.groupby(
    "context_class"
):

    row = {
        "context_class": context_class,
        "regions": len(group)
    }

    for cluster_id in CLUSTERS:

        # Area-weighted percentage across all
        # regions in this context group.
        weights = group[
            "total_area_km2"
        ].values

        values = group[
            f"cluster_{cluster_id}_area_percentage"
        ].values

        weighted_mean = np.average(
            values,
            weights=weights
        )

        row[
            f"cluster_{cluster_id}_percentage"
        ] = weighted_mean

    context_rows.append(
        row
    )


context_df = pd.DataFrame(
    context_rows
)

print(
    context_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.3f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "global_regional_geology_validation.csv",
    index=False
)

context_df.to_csv(
    "global_geological_context_summary.csv",
    index=False
)


# ============================================================
# FINAL INTERPRETATION GUIDANCE
# ============================================================

print("\n")
print("=" * 110)
print("INTERPRETATION GUIDANCE")
print("=" * 110)

print(
    """
This analysis is a REGIONAL GEOLOGICAL CONTEXT test.

The quadrangles are not treated as homogeneous
geological units.

Therefore:

1. A high C3 percentage in a highland/mixed region
   supports the hypothesis that C3 is associated with
   highland/tectonically complex surface-property
   environments.

2. A low C3 percentage in volcanic lowland regions
   would provide additional supporting evidence.

3. A high C2 percentage in specific high-latitude
   regions would support the geographic concentration
   observed in the global analysis.

4. Regional enrichment does NOT prove that a cluster
   corresponds to a geological unit.

5. The strongest evidence comes from combining:
      global distribution
      regional comparison
      feature-space centroids
      spatial coherence
      geological polygon validation
      algorithm stability
"""
)


# ============================================================
# FILE SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("GLOBAL GEOLOGICAL CROSS-VALIDATION COMPLETE")
print("=" * 110)

print("\nFiles created:")

print(
    "1. global_regional_geology_validation.csv"
)

print(
    "2. global_geological_context_summary.csv"
)
