import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER, OVDA_GEOLOGY_FILE
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from shapely.geometry import box
from shapely.ops import transform


# ============================================================
# FILES
# ============================================================

CLUSTER_FILE = str(K4_RASTER)

GEOLOGY_FILE = str(OVDA_GEOLOGY_FILE)


# ============================================================
# PARAMETERS
# ============================================================

VENUS_RADIUS = 6_051_000.0

MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0

NODATA = 255


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def lonlat_to_venus_xy(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    x_projected = (
        VENUS_RADIUS
        *
        np.deg2rad(x)
    )

    y_projected = (
        VENUS_RADIUS
        *
        np.deg2rad(y)
    )

    return x_projected, y_projected


def transform_geometry(geometry):

    return transform(
        lonlat_to_venus_xy,
        geometry
    )


# ============================================================
# OPEN CLUSTER RASTER
# ============================================================

with rasterio.open(CLUSTER_FILE) as src:

    transform_raster = src.transform

    raster_crs = src.crs

    raster_width = src.width

    raster_height = src.height


    # --------------------------------------------------------
    # Ovda boundaries in projected coordinates
    # --------------------------------------------------------

    min_x, min_y = lonlat_to_venus_xy(
        MIN_LON,
        MIN_LAT
    )

    max_x, max_y = lonlat_to_venus_xy(
        MAX_LON,
        MAX_LAT
    )


    ovda_box = box(
        min_x,
        min_y,
        max_x,
        max_y
    )


    # --------------------------------------------------------
    # Raster window
    # --------------------------------------------------------

    window = from_bounds(

        min_x,
        min_y,
        max_x,
        max_y,

        transform=transform_raster

    ).round_offsets().round_lengths()


    cluster_data = src.read(
        1,
        window=window
    )


    window_transform = rasterio.windows.transform(
        window,
        transform_raster
    )


# ============================================================
# CREATE WHOLE-OVDA BASELINE
# ============================================================

valid = (
    cluster_data != NODATA
)

valid_values = cluster_data[valid]


print("=" * 70)
print("OVDA BASELINE CLUSTER DISTRIBUTION")
print("=" * 70)


baseline = {}


for cluster in range(4):

    count = np.sum(
        valid_values == cluster
    )

    percentage = (
        count
        /
        len(valid_values)
        *
        100
    )

    baseline[cluster] = percentage

    print(
        f"Cluster {cluster}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# LOAD TECTONOMORPHIC UNITS
# ============================================================

gdf = gpd.read_file(

    GEOLOGY_FILE,

    layer="Tectonomorphic_map"

)


# ============================================================
# TRANSFORM GEOMETRIES
# ============================================================

gdf["geometry"] = (

    gdf["geometry"]
    .apply(transform_geometry)

)


# ============================================================
# KEEP OVDA UNITS
# ============================================================

gdf = gdf[
    gdf.geometry.intersects(
        ovda_box
    )
].copy()


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH GEOLOGICAL UNIT
# ============================================================

for _, feature in gdf.iterrows():

    name = str(
        feature["Name"]
    )

    geometry = feature.geometry


    # --------------------------------------------------------
    # Rasterize this geological unit
    # --------------------------------------------------------

    geology_mask = rasterize(

        [
            (
                geometry,
                1
            )
        ],

        out_shape=cluster_data.shape,

        transform=window_transform,

        fill=0,

        dtype="uint8"

    )


    mask = (

        (geology_mask == 1)

        &

        valid

    )


    values = cluster_data[
        mask
    ]


    total = len(values)


    if total == 0:

        continue


    record = {

        "geological_unit": name,

        "pixels": total

    }


    print("\n" + "-" * 60)

    print(
        "Geological unit:",
        name
    )

    print(
        "Pixels:",
        f"{total:,}"
    )


    # --------------------------------------------------------
    # Geological unit -> cluster
    # --------------------------------------------------------

    for cluster in range(4):

        count = np.sum(
            values == cluster
        )

        percentage = (
            count
            /
            total
            *
            100
        )


        # ----------------------------------------------------
        # Enrichment compared with whole Ovda
        # ----------------------------------------------------

        enrichment = (
            percentage
            /
            baseline[cluster]
        )


        record[
            f"cluster_{cluster}_percent"
        ] = percentage


        record[
            f"cluster_{cluster}_enrichment"
        ] = enrichment


        print(

            f"Cluster {cluster}: "
            f"{percentage:.2f}% "
            f"(enrichment = {enrichment:.2f}x)"

        )


    results.append(
        record
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(

    "geology_validation_metrics.csv",

    index=False

)


# ============================================================
# PRINT COMPLETE TABLE
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL GEOLOGICAL VALIDATION TABLE"
)

print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


print("\nSaved:")

print(
    "geology_validation_metrics.csv"
)
