import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box
from shapely.ops import transform

import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds


# ============================================================
# FILES
# ============================================================

from src.paths import K4_RASTER, OVDA_GEOLOGY_FILE

CLUSTER_FILE = K4_RASTER

GEOLOGY_FILE = OVDA_GEOLOGY_FILE


# ============================================================
# LAYERS
# ============================================================

TECTONOMORPHIC_LAYER = "Tectonomorphic_map"

TESSERA_LAYER = "IntraTesseraPlains_map"

BRUSHED_LAYER = "BrushedUnit_map"

CRATER_LAYER = "LargeCraters_map"


# ============================================================
# VENUS PARAMETERS
# ============================================================

# Radius used by YOUR K=4 raster CRS.
VENUS_RADIUS = 6_051_000.0


# ============================================================
# OVDA REGIO BOUNDARY
# ============================================================

MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0


# ============================================================
# CLUSTER NODATA
# ============================================================

CLUSTER_NODATA = 255


# ============================================================
# FUNCTION:
# VENUS LONGITUDE/LATITUDE -> YOUR RASTER CRS
# ============================================================

def lonlat_to_venus_xy(x, y):

    """
    Convert Venus longitude/latitude in degrees into the
    projected metre coordinates used by venus_clusters_k4_full.tif.

    Your raster uses an equirectangular Venus projection
    centered at longitude 0 and latitude 0.

    x = R * longitude_in_radians
    y = R * latitude_in_radians
    """

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


# ============================================================
# FUNCTION:
# TRANSFORM A GEOMETRY FROM LON/LAT TO RASTER CRS
# ============================================================

def transform_geometry(geometry):

    return transform(
        lonlat_to_venus_xy,
        geometry
    )


# ============================================================
# 1. OPEN K=4 CLUSTER RASTER
# ============================================================

print("=" * 70)
print("VENUS GEOLOGICAL VALIDATION")
print("=" * 70)


with rasterio.open(CLUSTER_FILE) as src:

    raster_crs = src.crs

    raster_transform = src.transform

    raster_width = src.width

    raster_height = src.height

    raster_nodata = src.nodata


    print("\nK=4 raster:")
    print("Width:", raster_width)
    print("Height:", raster_height)
    print("CRS:", raster_crs)
    print("Resolution:", src.res)
    print("NoData:", raster_nodata)


    # ========================================================
    # 2. CREATE OVDA REGION IN PROJECTED METRES
    # ========================================================

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


    # ========================================================
    # 3. CREATE SMALL RASTER WINDOW FOR OVDA
    #
    # We do NOT load the entire 176-million-pixel raster.
    # Only the Ovda region is needed.
    # ========================================================

    window = from_bounds(

        min_x,
        min_y,
        max_x,
        max_y,

        transform=raster_transform

    ).round_offsets().round_lengths()


    print("\nOvda raster window:")
    print(window)


    # ========================================================
    # 4. READ ONLY OVDA CLUSTER DATA
    # ========================================================

    cluster_data = src.read(

        1,

        window=window

    )


    # Transform corresponding to the window.
    window_transform = rasterio.windows.transform(
        window,
        raster_transform
    )


print(
    "\nOvda cluster array shape:",
    cluster_data.shape
)


# ============================================================
# 5. MASK CLUSTER NODATA
# ============================================================

cluster_valid = (
    cluster_data != CLUSTER_NODATA
)


# ============================================================
# 6. LOAD TECTONOMORPHIC DATA
# ============================================================

print("\n" + "=" * 70)

print("LOADING TECTONOMORPHIC MAP")

print("=" * 70)


tectono = gpd.read_file(

    GEOLOGY_FILE,

    layer=TECTONOMORPHIC_LAYER

)


print(
    "Features:",
    len(tectono)
)

print(
    "Original CRS:",
    tectono.crs
)


# ============================================================
# 7. TRANSFORM GEOLOGICAL GEOMETRIES
# ============================================================

tectono["geometry"] = (
    tectono["geometry"]
    .apply(transform_geometry)
)


# ============================================================
# 8. CLIP TO OVDA WINDOW
# ============================================================

tectono = tectono[
    tectono.geometry.intersects(
        ovda_box
    )
].copy()


print(
    "Features intersecting Ovda:",
    len(tectono)
)


# ============================================================
# 9. RASTERIZE TECTONOMORPHIC UNITS
#
# Each polygon receives a unique numeric ID.
# ============================================================

tectono["geo_id"] = (
    np.arange(len(tectono))
    + 1
)


shapes = [

    (
        geometry,
        int(geo_id)
    )

    for geometry, geo_id
    in zip(
        tectono.geometry,
        tectono.geo_id
    )

]


tectono_raster = rasterize(

    shapes,

    out_shape=cluster_data.shape,

    transform=window_transform,

    fill=0,

    dtype="int16"

)


# ============================================================
# 10. CALCULATE CLUSTER × GEOLOGY MATRIX
# ============================================================

rows = []


for _, feature in tectono.iterrows():

    geo_id = int(
        feature["geo_id"]
    )

    geo_name = str(
        feature["Name"]
    )


    # --------------------------------------------------------
    # Pixels belonging to this geological unit
    # --------------------------------------------------------

    geology_mask = (
        tectono_raster == geo_id
    )


    valid_mask = (
        geology_mask
        &
        cluster_valid
    )


    cluster_values = (
        cluster_data[valid_mask]
    )


    total = len(
        cluster_values
    )


    if total == 0:

        print(
            f"\n{geo_name}: "
            "no overlapping valid cluster pixels"
        )

        continue


    print("\n" + "-" * 50)

    print(
        "Geological unit:",
        geo_name
    )

    print(
        "Valid overlapping pixels:",
        f"{total:,}"
    )


    record = {

        "geological_unit": geo_name,

        "pixel_count": total

    }


    for cluster in range(4):

        count = np.sum(
            cluster_values == cluster
        )


        percentage = (
            count
            /
            total
            *
            100
        )


        record[
            f"cluster_{cluster}_percent"
        ] = percentage


        print(

            f"Cluster {cluster}: "
            f"{count:,} pixels "
            f"({percentage:.2f}%)"

        )


    rows.append(
        record
    )


# ============================================================
# 11. SAVE TECTONOMORPHIC VALIDATION
# ============================================================

tectono_results = pd.DataFrame(
    rows
)


tectono_results.to_csv(

    "tectonomorphic_cluster_validation.csv",

    index=False

)


print("\n" + "=" * 70)

print(
    "TECTONOMORPHIC VALIDATION SAVED"
)

print(
    "tectonomorphic_cluster_validation.csv"
)

print("=" * 70)


# ============================================================
# 12. INTRA-TESSERA PLAINS
# ============================================================

print("\n" + "=" * 70)

print(
    "INTRA-TESSERA PLAINS VALIDATION"
)

print("=" * 70)


tessera = gpd.read_file(

    GEOLOGY_FILE,

    layer=TESSERA_LAYER

)


print(
    "Original features:",
    len(tessera)
)


tessera["geometry"] = (
    tessera["geometry"]
    .apply(transform_geometry)
)


tessera_shapes = [

    (
        geometry,
        1
    )

    for geometry
    in tessera.geometry

]


tessera_raster = rasterize(

    tessera_shapes,

    out_shape=cluster_data.shape,

    transform=window_transform,

    fill=0,

    dtype="uint8"

)


tessera_mask = (

    (tessera_raster == 1)

    &

    cluster_valid

)


tessera_clusters = (
    cluster_data[tessera_mask]
)


tessera_total = len(
    tessera_clusters
)


print(
    "\nValid Intra-tessera plains pixels:",
    f"{tessera_total:,}"
)


if tessera_total > 0:

    tessera_rows = []


    for cluster in range(4):

        count = np.sum(
            tessera_clusters == cluster
        )


        percentage = (

            count
            /
            tessera_total
            *
            100

        )


        print(

            f"Cluster {cluster}: "
            f"{count:,} pixels "
            f"({percentage:.2f}%)"

        )


        tessera_rows.append({

            "feature": "Intra-tessera plains",

            "cluster": cluster,

            "pixels": count,

            "percentage": percentage

        })


    pd.DataFrame(
        tessera_rows
    ).to_csv(

        "intra_tessera_cluster_validation.csv",

        index=False

    )


# ============================================================
# 13. BRUSHED UNIT
# ============================================================

print("\n" + "=" * 70)

print(
    "BRUSHED UNIT VALIDATION"
)

print("=" * 70)


brushed = gpd.read_file(

    GEOLOGY_FILE,

    layer=BRUSHED_LAYER

)


print(
    "Original features:",
    len(brushed)
)


# ------------------------------------------------------------
# Keep the feature IDs from the source.
# ------------------------------------------------------------

brushed["geometry"] = (
    brushed["geometry"]
    .apply(transform_geometry)
)


for _, feature in brushed.iterrows():

    feature_id = feature["id"]

    geometry = feature.geometry


    brushed_mask = rasterize(

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

        (brushed_mask == 1)

        &

        cluster_valid

    )


    values = cluster_data[mask]


    total = len(values)


    print(
        f"\nBrushed unit ID {feature_id}"
    )

    print(
        "Valid pixels:",
        f"{total:,}"
    )


    if total == 0:

        continue


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


        print(

            f"Cluster {cluster}: "
            f"{count:,} pixels "
            f"({percentage:.2f}%)"

        )


# ============================================================
# 14. LARGE CRATERS
# ============================================================

print("\n" + "=" * 70)

print(
    "LARGE CRATER CLUSTER ASSOCIATION"
)

print("=" * 70)


craters = gpd.read_file(

    GEOLOGY_FILE,

    layer=CRATER_LAYER

)


for _, crater in craters.iterrows():

    x_lon = crater["INSIDE_X"]

    y_lat = crater["INSIDE_Y"]


    # Convert longitude/latitude to projected coordinates.

    x, y = lonlat_to_venus_xy(
        x_lon,
        y_lat
    )


    # Convert projected position to raster row/column.

    with rasterio.open(CLUSTER_FILE) as src:

        row, col = src.index(
            x,
            y
        )


        # ----------------------------------------------------
        # Check bounds.
        # ----------------------------------------------------

        if (

            row < 0

            or row >= src.height

            or col < 0

            or col >= src.width

        ):

            print(
                f"\nCrater at "
                f"{x_lon:.3f}, {y_lat:.3f}: "
                "outside raster"
            )

            continue


        cluster = src.read(
            1,
            window=rasterio.windows.Window(
                col,
                row,
                1,
                1
            )
        )[0, 0]


    print(

        f"\nCrater at "
        f"{x_lon:.3f}°E, "
        f"{y_lat:.3f}°: "

        f"Cluster {cluster}"

    )


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "GEOLOGICAL VALIDATION COMPLETE"
)

print("=" * 70)