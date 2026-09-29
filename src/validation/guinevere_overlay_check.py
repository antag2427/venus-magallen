import os
import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER, GUINEVERE_GDB
import matplotlib.pyplot as plt

from shapely.ops import transform as shapely_transform


# ============================================================
# CONFIGURATION
# ============================================================

CLUSTER_RASTER = str(K4_RASTER)

GDB_PATH = str(GUINEVERE_GDB)

USGS_LAYER = "V30Units16"

OUTPUT_ALL = "guinevere_usgs_overlay_all.png"
OUTPUT_CATEGORIES = "guinevere_usgs_overlay_categories.png"
OUTPUT_TESSERA = "guinevere_tessera_overlay.png"


VENUS_RASTER_RADIUS = 6051000.0
USGS_VENUS_RADIUS = 6051800.0

CLUSTERS = [0, 1, 2, 3]


# ============================================================
# USGS TYPE â†’ CATEGORY
# ============================================================

UNIT_CATEGORY = {

    "t": "Tessera_Upland",
    "ul": "Tessera_Upland",

    "pGr": "Plains",
    "pGlm": "Plains",

    "fpf": "Volcanic_Flow",
    "fl": "Volcanic_Flow",
    "fA1": "Volcanic_Flow",
    "fA2": "Volcanic_Flow",
    "fT": "Volcanic_Flow",
    "fV": "Volcanic_Flow",
    "fU": "Volcanic_Flow",
    "fR": "Volcanic_Flow",
    "fVl": "Volcanic_Flow",

    "v": "Volcanic_Edifice",

    "c": "Impact_Crater"
}


# ============================================================
# USGS MERCATOR â†’ VENUS LONGITUDE/LATITUDE
# ============================================================

def usgs_mercator_to_lonlat(x, y, z=None):

    x = np.asarray(x)
    y = np.asarray(y)

    lon = (
        x /
        USGS_VENUS_RADIUS
    )

    lat = (
        2.0 *
        np.arctan(
            np.exp(
                y /
                USGS_VENUS_RADIUS
            )
        )
        -
        np.pi / 2.0
    )

    return (
        np.rad2deg(lon),
        np.rad2deg(lat)
    )


# ============================================================
# VENUS LONGITUDE/LATITUDE â†’ K4 EQUIRECTANGULAR
# ============================================================

def lonlat_to_k4(x, y, z=None):

    lon = np.deg2rad(
        np.asarray(x)
    )

    lat = np.deg2rad(
        np.asarray(y)
    )

    return (
        VENUS_RASTER_RADIUS * lon,
        VENUS_RASTER_RADIUS * lat
    )


# ============================================================
# COMBINED TRANSFORMATION
# ============================================================

def usgs_to_k4(x, y, z=None):

    lon, lat = (
        usgs_mercator_to_lonlat(
            x,
            y
        )
    )

    return lonlat_to_k4(
        lon,
        lat
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 110)
print("GUINEVERE USGS / K-MEANS OVERLAY SANITY CHECK")
print("=" * 110)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(
    CLUSTER_RASTER
):

    raise FileNotFoundError(
        CLUSTER_RASTER
    )

if not os.path.exists(
    GDB_PATH
):

    raise FileNotFoundError(
        GDB_PATH
    )


# ============================================================
# READ RASTER
# ============================================================

print("\nOpening K=4 raster...")

with rasterio.open(
    CLUSTER_RASTER
) as src:

    print(
        f"Raster size: "
        f"{src.width:,} x "
        f"{src.height:,}"
    )

    print(
        f"Raster CRS:\n{src.crs}"
    )

    print(
        f"Raster bounds:\n{src.bounds}"
    )

    raster = src.read(
        1
    )

    raster_transform = src.transform


# ============================================================
# READ USGS POLYGONS
# ============================================================

print("\nLoading USGS V-30 polygons...")

units = gpd.read_file(
    GDB_PATH,
    layer=USGS_LAYER
)

print(
    f"USGS polygons: "
    f"{len(units):,}"
)

print(
    f"USGS CRS:\n{units.crs}"
)


# ============================================================
# CATEGORY ASSIGNMENT
# ============================================================

units["category"] = (
    units["Type"]
    .astype(str)
    .map(UNIT_CATEGORY)
)

unknown = units[
    units["category"].isna()
]

if len(unknown) > 0:

    print("\nWARNING: unknown USGS types:")

    print(
        unknown["Type"]
        .value_counts()
        .to_string()
    )


# ============================================================
# TRANSFORM POLYGONS
# ============================================================

print("\nTransforming USGS polygons...")

units["geometry_k4"] = units.geometry.apply(
    lambda geometry:
        shapely_transform(
            usgs_to_k4,
            geometry
        )
        if geometry is not None
        and not geometry.is_empty
        else None
)


# ============================================================
# CALCULATE TRANSFORMED BOUNDS
# ============================================================

valid_geometries = units[
    units["geometry_k4"].notna()
]

minx = valid_geometries.geometry_k4.bounds.minx.min()
miny = valid_geometries.geometry_k4.bounds.miny.min()
maxx = valid_geometries.geometry_k4.bounds.maxx.max()
maxy = valid_geometries.geometry_k4.bounds.maxy.max()


print("\n")
print("=" * 110)
print("TRANSFORMED USGS EXTENT")
print("=" * 110)

print(
    f"\nUSGS polygons in K4 coordinates:"
)

print(
    f"X: {minx:,.0f} to "
    f"{maxx:,.0f} m"
)

print(
    f"Y: {miny:,.0f} to "
    f"{maxy:,.0f} m"
)


# ============================================================
# CONVERT EXTENT TO LONGITUDE/LATITUDE
# ============================================================

print("\nApproximate geographic extent:")

min_lon = np.rad2deg(
    minx /
    VENUS_RASTER_RADIUS
)

max_lon = np.rad2deg(
    maxx /
    VENUS_RASTER_RADIUS
)

min_lat = np.rad2deg(
    miny /
    VENUS_RASTER_RADIUS
)

max_lat = np.rad2deg(
    maxy /
    VENUS_RASTER_RADIUS
)

print(
    f"Longitude: "
    f"{min_lon:.3f}Â° to "
    f"{max_lon:.3f}Â°"
)

print(
    f"Latitude: "
    f"{min_lat:.3f}Â° to "
    f"{max_lat:.3f}Â°"
)


# ============================================================
# PREPARE GUINEVERE RASTER WINDOW
# ============================================================

print("\nPreparing Guinevere raster...")

# Geographic bounds:
#
# 300â€“330Â°E
# 0â€“25Â°N
#

lon_grid = (
    np.rad2deg(
        (
            raster_transform.c
            +
            raster_transform.a *
            (
                np.arange(
                    raster.shape[1]
                ) +
                0.5
            )
        )
        /
        VENUS_RASTER_RADIUS
    )
)

lon_grid = (
    (
        lon_grid +
        180.0
    )
    % 360.0
) - 180.0

lon_grid_360 = (
    lon_grid %
    360.0
)

lat_grid = np.rad2deg(
    (
        raster_transform.f
        +
        raster_transform.e *
        (
            np.arange(
                raster.shape[0]
            ) +
            0.5
        )
    )
    /
    VENUS_RASTER_RADIUS
)


lon_mask = (
    (lon_grid_360 >= 300.0)
    &
    (lon_grid_360 <= 330.0)
)

lat_mask = (
    (lat_grid >= 0.0)
    &
    (lat_grid <= 25.0)
)

cols = np.where(
    lon_mask
)[0]

rows = np.where(
    lat_mask
)[0]


col_start = cols.min()
col_end = cols.max() + 1

row_start = rows.min()
row_end = rows.max() + 1


guinevere = raster[
    row_start:row_end,
    col_start:col_end
]


guinevere_transform = (
    rasterio.windows.Window(
        col_start,
        row_start,
        col_end - col_start,
        row_end - row_start
    )
)


print(
    f"Guinevere raster dimensions: "
    f"{guinevere.shape[1]:,} x "
    f"{guinevere.shape[0]:,}"
)


# ============================================================
# PLOT 1 â€” ALL USGS POLYGONS
# ============================================================

print("\nCreating all-unit overlay...")

fig, ax = plt.subplots(
    figsize=(14, 10)
)


# ------------------------------------------------------------
# Raster display
# ------------------------------------------------------------

extent = [

    raster_transform.c
    +
    raster_transform.a *
    col_start,

    raster_transform.c
    +
    raster_transform.a *
    col_end,

    raster_transform.f
    +
    raster_transform.e *
    row_end,

    raster_transform.f
    +
    raster_transform.e *
    row_start
]


masked_guinevere = np.ma.masked_where(
    ~np.isin(
        guinevere,
        CLUSTERS
    ),
    guinevere
)


ax.imshow(
    masked_guinevere,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    alpha=0.75
)


# ------------------------------------------------------------
# Plot polygon boundaries
# ------------------------------------------------------------

for _, row in units.iterrows():

    geometry = row.geometry_k4

    if geometry is None:
        continue

    gpd.GeoSeries(
        [geometry]
    ).boundary.plot(
        ax=ax,
        linewidth=0.5
    )


ax.set_title(
    "Guinevere V-30: K-Means K=4 + USGS Geological Boundaries"
)

ax.set_xlabel(
    "Equirectangular Venus X (m)"
)

ax.set_ylabel(
    "Equirectangular Venus Y (m)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ALL,
    dpi=200
)

plt.close()


print(
    f"Saved: {OUTPUT_ALL}"
)


# ============================================================
# PLOT 2 â€” CATEGORIES
# ============================================================

print(
    "\nCreating geological-category overlay..."
)


category_styles = {
    "Tessera_Upland": {
        "linewidth": 2.5
    },

    "Plains": {
        "linewidth": 1.0
    },

    "Volcanic_Flow": {
        "linewidth": 1.0
    },

    "Volcanic_Edifice": {
        "linewidth": 2.0
    },

    "Impact_Crater": {
        "linewidth": 2.5
    }
}


fig, ax = plt.subplots(
    figsize=(14, 10)
)


ax.imshow(
    masked_guinevere,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    alpha=0.75
)


for category, style in category_styles.items():

    subset = units[
        units["category"] ==
        category
    ]

    if len(subset) == 0:
        continue

    print(
        f"{category}: "
        f"{len(subset)} polygons"
    )

    for _, row in subset.iterrows():

        geometry = row.geometry_k4

        if geometry is None:
            continue

        gpd.GeoSeries(
            [geometry]
        ).boundary.plot(
            ax=ax,
            linewidth=style["linewidth"],
            label=category
        )


# ------------------------------------------------------------
# Remove duplicate legend entries
# ------------------------------------------------------------

handles, labels = (
    ax.get_legend_handles_labels()
)

unique = {}

for handle, label in zip(
    handles,
    labels
):

    if label not in unique:

        unique[label] = handle


if unique:

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper right"
    )


ax.set_title(
    "Guinevere V-30: K-Means K=4 + Geological Categories"
)

ax.set_xlabel(
    "Equirectangular Venus X (m)"
)

ax.set_ylabel(
    "Equirectangular Venus Y (m)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_CATEGORIES,
    dpi=200
)

plt.close()


print(
    f"Saved: {OUTPUT_CATEGORIES}"
)


# ============================================================
# PLOT 3 â€” TESSERA/UPLAND EMPHASIS
# ============================================================

print(
    "\nCreating tessera/upland emphasis map..."
)


fig, ax = plt.subplots(
    figsize=(14, 10)
)


ax.imshow(
    masked_guinevere,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    alpha=0.65
)


tessera = units[
    units["category"] ==
    "Tessera_Upland"
]


print(
    f"Tessera/upland polygons: "
    f"{len(tessera)}"
)


for _, row in tessera.iterrows():

    geometry = row.geometry_k4

    if geometry is None:
        continue

    gpd.GeoSeries(
        [geometry]
    ).boundary.plot(
        ax=ax,
        linewidth=2.5
    )


ax.set_title(
    "Guinevere V-30: K-Means K=4 + USGS Tessera/Upland"
)

ax.set_xlabel(
    "Equirectangular Venus X (m)"
)

ax.set_ylabel(
    "Equirectangular Venus Y (m)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_TESSERA,
    dpi=200
)

plt.close()


print(
    f"Saved: {OUTPUT_TESSERA}"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 110)
print("OVERLAY SANITY CHECK COMPLETE")
print("=" * 110)

print("\nGenerated maps:")

print(
    f"1. {OUTPUT_ALL}"
)

print(
    f"2. {OUTPUT_CATEGORIES}"
)

print(
    f"3. {OUTPUT_TESSERA}"
)

print("\nInspect the maps visually.")

print(
    "\nThe USGS polygon boundaries should fall "
    "over the expected Guinevere classification."
)

print(
    "\nMost importantly, confirm that the tessera/upland "
    "boundaries are spatially registered rather than "
    "systematically shifted."
)

