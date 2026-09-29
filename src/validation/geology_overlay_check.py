import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER, OVDA_GEOLOGY_FILE
from rasterio.windows import from_bounds

from shapely.geometry import box
from shapely.ops import transform


# ============================================================
# FILES
# ============================================================

CLUSTER_FILE = str(K4_RASTER)

GEOLOGY_FILE = str(OVDA_GEOLOGY_FILE)


# ============================================================
# VENUS PARAMETERS
# ============================================================

VENUS_RADIUS = 6_051_000.0


# ============================================================
# OVDA REGION
# ============================================================

MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def lonlat_to_venus_xy(x, y):

    x = np.asarray(x)

    y = np.asarray(y)

    projected_x = (
        VENUS_RADIUS
        *
        np.deg2rad(x)
    )

    projected_y = (
        VENUS_RADIUS
        *
        np.deg2rad(y)
    )

    return projected_x, projected_y


def transform_geometry(geometry):

    return transform(
        lonlat_to_venus_xy,
        geometry
    )


# ============================================================
# 1. OPEN CLUSTER RASTER
# ============================================================

print("=" * 70)

print("GEOLOGICAL OVERLAY SANITY CHECK")

print("=" * 70)


with rasterio.open(
    CLUSTER_FILE
) as src:

    raster_transform = src.transform

    raster_crs = src.crs

    raster_width = src.width

    raster_height = src.height


    # --------------------------------------------------------
    # Convert Ovda boundaries to projected metres
    # --------------------------------------------------------

    min_x, min_y = lonlat_to_venus_xy(
        MIN_LON,
        MIN_LAT
    )

    max_x, max_y = lonlat_to_venus_xy(
        MAX_LON,
        MAX_LAT
    )


    # --------------------------------------------------------
    # Create bounding box
    # --------------------------------------------------------

    ovda_box = box(
        min_x,
        min_y,
        max_x,
        max_y
    )


    # --------------------------------------------------------
    # Get raster window
    # --------------------------------------------------------

    window = from_bounds(

        min_x,
        min_y,
        max_x,
        max_y,

        transform=raster_transform

    ).round_offsets().round_lengths()


    # --------------------------------------------------------
    # Read Ovda classification
    #
    # We downsample for plotting only.
    # --------------------------------------------------------

    scale = 4

    out_height = int(
        window.height / scale
    )

    out_width = int(
        window.width / scale
    )


    cluster_map = src.read(

        1,

        window=window,

        out_shape=(
            out_height,
            out_width
        ),

        resampling=rasterio.enums.Resampling.nearest

    )


    # --------------------------------------------------------
    # Calculate display transform
    # --------------------------------------------------------

    display_transform = (
        rasterio.windows.transform(
            window,
            raster_transform
        )
        *
        rasterio.Affine.scale(
            window.width / out_width,
            window.height / out_height
        )
    )


# ============================================================
# 2. MASK NODATA
# ============================================================

cluster_map = np.ma.masked_equal(
    cluster_map,
    255
)


# ============================================================
# 3. LOAD TECTONOMORPHIC MAP
# ============================================================

print("\nLoading tectonomorphic polygons...")


tectono = gpd.read_file(

    GEOLOGY_FILE,

    layer="Tectonomorphic_map"

)


print(
    "Original features:",
    len(tectono)
)


print(
    "Original CRS:",
    tectono.crs
)


# ============================================================
# 4. TRANSFORM GEOLOGICAL POLYGONS
# ============================================================

tectono["geometry"] = (

    tectono["geometry"]
    .apply(transform_geometry)

)


# ============================================================
# 5. KEEP ONLY POLYGONS INTERSECTING OVDA
# ============================================================

tectono = tectono[
    tectono.geometry.intersects(
        ovda_box
    )
].copy()


print(
    "Polygons inside/intersecting Ovda:",
    len(tectono)
)


# ============================================================
# 6. CONVERT POLYGONS TO DISPLAY COORDINATES
#
# We need longitude/latitude for plotting.
# ============================================================

def projected_to_lonlat(x, y):

    x = np.asarray(x)

    y = np.asarray(y)

    longitude = (
        x
        /
        VENUS_RADIUS
        *
        180.0
        /
        np.pi
    )

    latitude = (
        y
        /
        VENUS_RADIUS
        *
        180.0
        /
        np.pi
    )

    return longitude, latitude


def projected_geometry_to_lonlat(geometry):

    return transform(
        projected_to_lonlat,
        geometry
    )


tectono["geometry_lonlat"] = (

    tectono.geometry
    .apply(projected_geometry_to_lonlat)

)


plt.figure(
    figsize=(14, 10)
)


plt.imshow(

    cluster_map,

    cmap="tab10",

    interpolation="nearest",

    extent=[
        MIN_LON,
        MAX_LON,
        MIN_LAT,
        MAX_LAT
    ],

    origin="upper"

)



for _, feature in tectono.iterrows():

    geometry = feature[
        "geometry_lonlat"
    ]

    name = str(
        feature["Name"]
    )


    try:

        x, y = geometry.exterior.xy

        plt.plot(
            x,
            y,
            linewidth=1.5,
            label=name
        )

    except AttributeError:

        # MultiPolygon or complex geometry.
        #
        # Plot each polygon separately.

        for polygon in geometry.geoms:

            x, y = polygon.exterior.xy

            plt.plot(
                x,
                y,
                linewidth=1.5
            )



for _, feature in tectono.iterrows():

    geometry = feature[
        "geometry_lonlat"
    ]

    name = str(
        feature["Name"]
    )


    representative_point = (
        geometry.representative_point()
    )


    plt.text(

        representative_point.x,

        representative_point.y,

        name,

        fontsize=7

    )

plt.xlabel(
    "Longitude (degrees East)"
)

plt.ylabel(
    "Latitude (degrees)"
)


plt.title(
    "K=4 Clusters with Tectonomorphic Geological Boundaries"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.show()

