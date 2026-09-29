import numpy as np
import matplotlib.pyplot as plt
import rasterio


# ============================================================
# INPUT
# ============================================================

from src.paths import K4_RASTER

CLUSTER_FILE = K4_RASTER

NODATA = 255

VENUS_RADIUS = 6_051_000.0


# ============================================================
# REGION OF INTEREST
#
# Ovda Regio quadrangle:
#
# Longitude: 90E to 120E
# Latitude:   0N to 25S
#
# ============================================================

MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0


# ============================================================
# 1. OPEN CLUSTER MAP
# ============================================================

with rasterio.open(CLUSTER_FILE) as src:

    print("=" * 60)
    print("OVDA REGIO CLUSTER VALIDATION")
    print("=" * 60)

    print("Raster size:")
    print(src.height, src.width)

    print("\nCRS:")
    print(src.crs)

    print("\nResolution:")
    print(src.res)

    print("\nTransform:")
    print(src.transform)


    # --------------------------------------------------------
    # Convert geographic coordinates to raster pixel indices.
    #
    # Because the CRS is a simple equirectangular Venus
    # projection:
    #
    # x = R * longitude
    # y = R * latitude
    #
    # longitude and latitude are converted from degrees
    # to radians first.
    # --------------------------------------------------------

    min_x = (
        MIN_LON
        *
        np.pi
        /
        180.0
        *
        VENUS_RADIUS
    )

    max_x = (
        MAX_LON
        *
        np.pi
        /
        180.0
        *
        VENUS_RADIUS
    )

    min_y = (
        MIN_LAT
        *
        np.pi
        /
        180.0
        *
        VENUS_RADIUS
    )

    max_y = (
        MAX_LAT
        *
        np.pi
        /
        180.0
        *
        VENUS_RADIUS
    )


    # --------------------------------------------------------
    # Convert coordinates to pixels.
    # --------------------------------------------------------

    row_top, col_left = src.index(
        min_x,
        max_y
    )

    row_bottom, col_right = src.index(
        max_x,
        min_y
    )


    # --------------------------------------------------------
    # Make sure indices stay inside the raster.
    # --------------------------------------------------------

    row_top = max(
        0,
        row_top
    )

    col_left = max(
        0,
        col_left
    )

    row_bottom = min(
        src.height,
        row_bottom
    )

    col_right = min(
        src.width,
        col_right
    )


    print("\nPixel window:")

    print(
        "Rows:",
        row_top,
        "to",
        row_bottom
    )

    print(
        "Columns:",
        col_left,
        "to",
        col_right
    )


    # --------------------------------------------------------
    # Read only the Ovda region.
    # --------------------------------------------------------

    window = rasterio.windows.Window(

        col_left,

        row_top,

        col_right - col_left,

        row_bottom - row_top

    )


    # --------------------------------------------------------
    # Downsample for visualization if necessary.
    # --------------------------------------------------------

    scale = 4

    out_height = max(
        1,
        int(window.height / scale)
    )

    out_width = max(
        1,
        int(window.width / scale)
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


# ============================================================
# 2. MASK NODATA
# ============================================================

cluster_map = np.ma.masked_equal(
    cluster_map,
    NODATA
)


# ============================================================
# 3. DISPLAY
# ============================================================

plt.figure(
    figsize=(12, 8)
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


# ============================================================
# 4. COLORBAR
# ============================================================

colorbar = plt.colorbar()

colorbar.set_label(
    "K-Means Cluster ID"
)


# ============================================================
# 5. AXES
# ============================================================

plt.xlabel(
    "Longitude (degrees East)"
)

plt.ylabel(
    "Latitude (degrees)"
)


# ============================================================
# 6. TITLE
# ============================================================

plt.title(
    "K=4 Clustering — Ovda Regio Quadrangle"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.show()