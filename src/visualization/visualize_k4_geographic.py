import numpy as np
import matplotlib.pyplot as plt
import rasterio

from src.paths import K4_RASTER


# ============================================================
# INPUT
# ============================================================

CLUSTER_FILE = K4_RASTER

NODATA = 255

VENUS_RADIUS = 6_051_000.0


# ============================================================
# 1. OPEN K=4 CLUSTER RASTER
# ============================================================

print("=" * 60)
print("K=4 GEOGRAPHIC VISUALIZATION")
print("=" * 60)


with rasterio.open(CLUSTER_FILE) as src:

    print("\nCRS:")
    print(src.crs)

    print("\nWidth:", src.width)

    print("Height:", src.height)

    print("\nResolution:", src.res)

    print("\nTransform:")
    print(src.transform)

    print("\nNoData:", src.nodata)


    # --------------------------------------------------------
    # Reduce resolution only for display.
    #
    # The actual GeoTIFF remains full resolution.
    # --------------------------------------------------------

    scale = 8

    display_height = src.height // scale

    display_width = src.width // scale


    cluster_map = src.read(
        1,
        out_shape=(
            display_height,
            display_width
        ),
        resampling=rasterio.enums.Resampling.nearest
    )


    # --------------------------------------------------------
    # Calculate the corresponding transform for the
    # downsampled display image.
    # --------------------------------------------------------

    display_transform = (
        src.transform
        *
        src.transform.scale(
            src.width / display_width,
            src.height / display_height
        )
    )


# ============================================================
# 2. MASK NODATA
# ============================================================

cluster_map = np.ma.masked_equal(
    cluster_map,
    NODATA
)


# ============================================================
# 3. GET IMAGE BOUNDARIES IN PROJECTED METRES
# ============================================================

left_x = display_transform.c

top_y = display_transform.f


right_x = (
    left_x
    +
    display_width
    *
    display_transform.a
)


bottom_y = (
    top_y
    +
    display_height
    *
    display_transform.e
)


print("\nProjected extent:")

print(
    "X:",
    left_x,
    "to",
    right_x
)

print(
    "Y:",
    bottom_y,
    "to",
    top_y
)


# ============================================================
# 4. CONVERT VENUS PROJECTED COORDINATES
#    TO LONGITUDE / LATITUDE
#
# Your CRS is an equirectangular Venus projection.
#
# x = R * longitude_in_radians
# y = R * latitude_in_radians
#
# Therefore:
#
# longitude = x / R * 180/pi
# latitude  = y / R * 180/pi
# ============================================================

min_lon = (
    left_x
    /
    VENUS_RADIUS
    *
    180.0
    /
    np.pi
)


max_lon = (
    right_x
    /
    VENUS_RADIUS
    *
    180.0
    /
    np.pi
)


min_lat = (
    bottom_y
    /
    VENUS_RADIUS
    *
    180.0
    /
    np.pi
)


max_lat = (
    top_y
    /
    VENUS_RADIUS
    *
    180.0
    /
    np.pi
)


# ============================================================
# 5. PRINT GEOGRAPHIC EXTENT
# ============================================================

print("\nApproximate Venus geographic extent:")

print(
    f"Longitude: {min_lon:.3f}° "
    f"to {max_lon:.3f}°"
)

print(
    f"Latitude: {min_lat:.3f}° "
    f"to {max_lat:.3f}°"
)


# ============================================================
# 6. DISPLAY MAP
# ============================================================

plt.figure(
    figsize=(16, 8)
)


plt.imshow(

    cluster_map,

    cmap="tab10",

    interpolation="nearest",

    extent=[
        min_lon,
        max_lon,
        min_lat,
        max_lat
    ],

    origin="upper"

)


# ============================================================
# 7. COLORBAR
# ============================================================

colorbar = plt.colorbar()

colorbar.set_label(
    "Cluster ID"
)


# ============================================================
# 8. AXES
# ============================================================

plt.xlabel(
    "Longitude (degrees)"
)

plt.ylabel(
    "Latitude (degrees)"
)


# ============================================================
# 9. TITLE
# ============================================================

plt.title(
    "K=4 Venus Terrain Clustering"
)


# ============================================================
# 10. GRID
# ============================================================

plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()

plt.show()