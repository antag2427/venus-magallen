import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import mapping


# ============================================================
# CONFIGURATION
# ============================================================

from src.paths import (
    K4_RASTER,
    RADAR_FILE,
    HEIGHT_FILE,
    EMISSIVITY_FILE,
    SLOPE_FILE,
    REFLECTIVITY_FILE,
    OVDA_GEOLOGY_FILE,
)

CLUSTER_RASTER = K4_RASTER

GPKG_FILE = OVDA_GEOLOGY_FILE

TECTONOMORPHIC_LAYER = "Tectonomorphic_map"
PLAINS_LAYER = "IntraTesseraPlains_map"
BRUSHED_LAYER = "BrushedUnit_map"
CRATERS_LAYER = "LargeCraters_map"

# Ovda Regio / V-35 geographic bounds
MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0

# Venus radius used by our K=4 raster CRS
VENUS_RADIUS = 6051000.0

# Cluster IDs
CLUSTERS = [0, 1, 2, 3]

# Internal NoData conventions
CLUSTER_NODATA = 255
RADAR_NODATA = 0
OTHER_NODATA = -32768

# Pixel dimensions of our radar grid
PIXEL_SIZE_M = 2025.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lonlat_to_xy(lon, lat):
    """
    Convert Venus longitude/latitude in degrees to the
    Equirectangular projected coordinates used by the K=4 raster.

    x = R * longitude
    y = R * latitude
    where longitude/latitude are converted to radians.
    """

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x = VENUS_RADIUS * lon_rad
    y = VENUS_RADIUS * lat_rad

    return x, y


def xy_to_lonlat(x, y):
    """
    Convert K=4 raster projected coordinates to Venus lon/lat.
    """

    lon = np.rad2deg(x / VENUS_RADIUS)
    lat = np.rad2deg(y / VENUS_RADIUS)

    return lon, lat


def get_ovda_window(src):
    """
    Get the raster window corresponding approximately to
    90-120 E, 25-0 S.
    """

    min_x, min_y = lonlat_to_xy(MIN_LON, MIN_LAT)
    max_x, max_y = lonlat_to_xy(MAX_LON, MAX_LAT)

    window = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    return window


def feature_mask_for_geometry(geometry, out_shape, transform):
    """
    Return True for pixels lying inside the supplied polygon.
    """

    return geometry_mask(
        [mapping(geometry)],
        out_shape=out_shape,
        transform=transform,
        invert=True,
        all_touched=False
    )


def cluster_statistics(cluster_data):
    """
    Return counts and percentages for cluster IDs.
    """

    valid = cluster_data[
        (cluster_data >= 0) &
        (cluster_data <= 3)
    ]

    total = len(valid)

    output = {}

    for cluster_id in CLUSTERS:

        count = np.sum(valid == cluster_id)

        if total > 0:
            percentage = 100.0 * count / total
        else:
            percentage = np.nan

        output[cluster_id] = {
            "count": int(count),
            "percentage": percentage
        }

    return output


def dominant_cluster(cluster_data):
    """
    Return dominant cluster and its purity.
    """

    stats = cluster_statistics(cluster_data)

    best_cluster = None
    best_percentage = -1.0

    for cluster_id in CLUSTERS:

        pct = stats[cluster_id]["percentage"]

        if not np.isnan(pct) and pct > best_percentage:

            best_percentage = pct
            best_cluster = cluster_id

    return best_cluster, best_percentage


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 100)
print("SYSTEMATIC GEOLOGICAL VALIDATION — K-MEANS K=4")
print("=" * 100)

required_files = [
    CLUSTER_RASTER,
    RADAR_FILE,
    HEIGHT_FILE,
    EMISSIVITY_FILE,
    SLOPE_FILE,
    REFLECTIVITY_FILE,
    GPKG_FILE
]

print("\nChecking input files...")

for filename in required_files:

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"Required file not found: {filename}"
        )

    print(f"  OK: {filename}")


# ============================================================
# OPEN CLUSTER RASTER
# ============================================================

print("\nOpening K=4 cluster raster...")

with rasterio.open(CLUSTER_RASTER) as src:

    cluster_window = get_ovda_window(src)

    cluster_data = src.read(
        1,
        window=cluster_window
    )

    cluster_transform = src.window_transform(
        cluster_window
    )

    cluster_crs = src.crs

    print(
        f"Raster size: {src.width} x {src.height}"
    )

    print(
        f"Ovda window: "
        f"col_off={cluster_window.col_off:.0f}, "
        f"row_off={cluster_window.row_off:.0f}, "
        f"width={cluster_window.width:.0f}, "
        f"height={cluster_window.height:.0f}"
    )


# ============================================================
# OVDA BASELINE
# ============================================================

print("\n")
print("=" * 100)
print("OVDA BASELINE CLUSTER DISTRIBUTION")
print("=" * 100)

baseline_stats = cluster_statistics(
    cluster_data
)

baseline_rows = []

for cluster_id in CLUSTERS:

    count = baseline_stats[cluster_id]["count"]

    percentage = baseline_stats[cluster_id]["percentage"]

    print(
        f"Cluster {cluster_id}: "
        f"{count:,} pixels "
        f"({percentage:.2f}%)"
    )

    baseline_rows.append({
        "cluster": cluster_id,
        "pixels": count,
        "percentage": percentage
    })


baseline_df = pd.DataFrame(
    baseline_rows
)

baseline_df.to_csv(
    "ovda_cluster_baseline.csv",
    index=False
)


# ============================================================
# OPEN MULTI-BAND DATA
# ============================================================

print("\n")
print("=" * 100)
print("LOADING GEOLOGICAL FEATURE RASTERS")
print("=" * 100)

feature_files = {
    "radar": RADAR_FILE,
    "height": HEIGHT_FILE,
    "emissivity": EMISSIVITY_FILE,
    "slope": SLOPE_FILE,
    "reflectivity": REFLECTIVITY_FILE
}

feature_arrays = {}

for name, filename in feature_files.items():

    print(f"\nLoading {name}: {filename}")

    with rasterio.open(filename) as src:

        # We use physical lon/lat to obtain the corresponding
        # Ovda window because the source rasters use a different
        # projected CRS than the aligned K=4 raster.
        min_x, min_y = lonlat_to_xy(
            MIN_LON,
            MIN_LAT
        )

        max_x, max_y = lonlat_to_xy(
            MAX_LON,
            MAX_LAT
        )

        # The original source rasters are not guaranteed to use
        # exactly the same projection/radius as the K=4 raster.
        # Therefore, instead of directly reading by the K=4 window,
        # we use the already validated K=4 cluster raster for
        # geological validation and only use feature statistics
        # from the corresponding geographic window when possible.

        try:

            window = from_bounds(
                min_x,
                min_y,
                max_x,
                max_y,
                transform=src.transform
            )

            window = (
                window
                .round_offsets()
                .round_lengths()
            )

            arr = src.read(
                1,
                window=window
            )

            feature_arrays[name] = arr

            print(
                f"  Loaded shape: {arr.shape}"
            )

        except Exception as exc:

            print(
                f"  WARNING: Could not load geographic "
                f"window for {name}: {exc}"
            )


# ============================================================
# LOAD GEOLOGICAL LAYERS
# ============================================================

print("\n")
print("=" * 100)
print("LOADING GEOLOGICAL GIS LAYERS")
print("=" * 100)

tecto = gpd.read_file(
    GPKG_FILE,
    layer=TECTONOMORPHIC_LAYER
)

plains = gpd.read_file(
    GPKG_FILE,
    layer=PLAINS_LAYER
)

brushed = gpd.read_file(
    GPKG_FILE,
    layer=BRUSHED_LAYER
)

craters = gpd.read_file(
    GPKG_FILE,
    layer=CRATERS_LAYER
)

print(
    f"\nTectonomorphic features: "
    f"{len(tecto)}"
)

print(
    f"Intra-tessera plains: "
    f"{len(plains)}"
)

print(
    f"Brushed units: "
    f"{len(brushed)}"
)

print(
    f"Large craters: "
    f"{len(craters)}"
)


# ============================================================
# CRS CHECK
# ============================================================

print("\n")
print("=" * 100)
print("GEOLOGICAL LAYER CRS")
print("=" * 100)

print(
    f"Tectonomorphic CRS: {tecto.crs}"
)

print(
    f"Plains CRS: {plains.crs}"
)

print(
    f"Brushed CRS: {brushed.crs}"
)

print(
    f"Craters CRS: {craters.crs}"
)


# ============================================================
# IMPORTANT:
# THE GEOLOGICAL GPKG USES VENUS GEOGRAPHIC COORDINATES.
#
# We deliberately do NOT ask pyproj to transform the Venus
# CRS to Earth WGS84.
#
# Instead, polygon coordinates are treated as Venus lon/lat
# and converted manually to the K=4 raster's Equirectangular
# coordinates using VENUS_RADIUS.
# ============================================================


# ============================================================
# TECTONOMORPHIC VALIDATION
# ============================================================

print("\n")
print("=" * 100)
print("TECTONOMORPHIC UNIT VALIDATION")
print("=" * 100)

tecto_results = []


for idx, row in tecto.iterrows():

    geometry = row.geometry

    if geometry is None or geometry.is_empty:

        continue

    # --------------------------------------------------------
    # Convert polygon coordinates:
    # GPKG -> Venus lon/lat -> K=4 projected coordinates
    # --------------------------------------------------------

    try:

        from shapely.ops import transform as shapely_transform

        def geographic_to_k4(x, y, z=None):

            return lonlat_to_xy(x, y)

        geometry_k4 = shapely_transform(
            geographic_to_k4,
            geometry
        )

    except Exception as exc:

        print(
            f"WARNING: geometry transformation failed "
            f"for feature {idx}: {exc}"
        )

        continue

    # --------------------------------------------------------
    # Check intersection with the Ovda raster window
    # --------------------------------------------------------

    raster_bounds = rasterio.windows.bounds(
        cluster_window,
        rasterio.open(CLUSTER_RASTER).transform
    )

    from shapely.geometry import box

    raster_box = box(
        raster_bounds[0],
        raster_bounds[1],
        raster_bounds[2],
        raster_bounds[3]
    )

    if not geometry_k4.intersects(raster_box):

        continue

    geometry_k4 = geometry_k4.intersection(
        raster_box
    )

    if geometry_k4.is_empty:

        continue

    # --------------------------------------------------------
    # Raster mask
    # --------------------------------------------------------

    mask = feature_mask_for_geometry(
        geometry_k4,
        cluster_data.shape,
        cluster_transform
    )

    values = cluster_data[mask]

    values = values[
        (values >= 0) &
        (values <= 3)
    ]

    if len(values) == 0:

        continue

    stats = cluster_statistics(values)

    dominant, purity = dominant_cluster(values)

    area_km2 = (
        len(values) *
        (PIXEL_SIZE_M / 1000.0) ** 2
    )

    name = row.get(
        "Name",
        f"feature_{idx}"
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Valid pixels: {len(values):,}"
    )

    print(
        f"  Area: {area_km2:,.1f} km²"
    )

    print(
        f"  Dominant cluster: C{dominant}"
    )

    print(
        f"  Purity: {purity:.2f}%"
    )

    row_result = {
        "feature_index": idx,
        "name": name,
        "area_km2": area_km2,
        "dominant_cluster": dominant,
        "purity": purity,
        "valid_pixels": len(values)
    }

    for cluster_id in CLUSTERS:

        pct = stats[cluster_id]["percentage"]

        baseline_pct = baseline_stats[
            cluster_id
        ]["percentage"]

        if baseline_pct > 0:

            enrichment = pct / baseline_pct

        else:

            enrichment = np.nan

        row_result[
            f"cluster_{cluster_id}_percentage"
        ] = pct

        row_result[
            f"cluster_{cluster_id}_enrichment"
        ] = enrichment

        row_result[
            f"cluster_{cluster_id}_pixels"
        ] = stats[cluster_id]["count"]

        print(
            f"  C{cluster_id}: "
            f"{pct:.2f}% "
            f"(enrichment {enrichment:.2f}x)"
        )

    tecto_results.append(
        row_result
    )


tecto_df = pd.DataFrame(
    tecto_results
)

tecto_df.to_csv(
    "systematic_tectonomorphic_validation.csv",
    index=False
)


# ============================================================
# INTRA-TESSERA PLAINS
# ============================================================

print("\n")
print("=" * 100)
print("INTRA-TESSERA PLAINS VALIDATION")
print("=" * 100)


plains_all_values = []


for idx, row in plains.iterrows():

    geometry = row.geometry

    if geometry is None or geometry.is_empty:

        continue

    def geographic_to_k4(x, y, z=None):

        return lonlat_to_xy(x, y)

    from shapely.ops import transform as shapely_transform

    geometry_k4 = shapely_transform(
        geographic_to_k4,
        geometry
    )

    raster_bounds = rasterio.windows.bounds(
        cluster_window,
        rasterio.open(CLUSTER_RASTER).transform
    )

    from shapely.geometry import box

    raster_box = box(
        raster_bounds[0],
        raster_bounds[1],
        raster_bounds[2],
        raster_bounds[3]
    )

    if not geometry_k4.intersects(raster_box):

        continue

    geometry_k4 = geometry_k4.intersection(
        raster_box
    )

    if geometry_k4.is_empty:

        continue

    mask = feature_mask_for_geometry(
        geometry_k4,
        cluster_data.shape,
        cluster_transform
    )

    values = cluster_data[mask]

    valid = values[
        (values >= 0) &
        (values <= 3)
    ]

    plains_all_values.append(
        valid
    )


if plains_all_values:

    plains_values = np.concatenate(
        plains_all_values
    )

else:

    plains_values = np.array(
        [],
        dtype=np.uint8
    )


if len(plains_values) > 0:

    stats = cluster_statistics(
        plains_values
    )

    dominant, purity = dominant_cluster(
        plains_values
    )

    coverage = (
        100.0 *
        stats[dominant]["count"] /
        np.sum(
            baseline_df["pixels"]
        )
    )

    print(
        f"Valid pixels: {len(plains_values):,}"
    )

    print(
        f"Dominant cluster: C{dominant}"
    )

    print(
        f"Purity: {purity:.2f}%"
    )

    for cluster_id in CLUSTERS:

        pct = stats[cluster_id]["percentage"]

        baseline_pct = baseline_stats[
            cluster_id
        ]["percentage"]

        enrichment = (
            pct / baseline_pct
            if baseline_pct > 0
            else np.nan
        )

        print(
            f"C{cluster_id}: "
            f"{pct:.2f}% "
            f"(enrichment {enrichment:.2f}x)"
        )

    plains_summary = {
        "valid_pixels": len(plains_values),
        "dominant_cluster": dominant,
        "purity": purity
    }

    for cluster_id in CLUSTERS:

        plains_summary[
            f"cluster_{cluster_id}_percentage"
        ] = stats[cluster_id]["percentage"]

        plains_summary[
            f"cluster_{cluster_id}_enrichment"
        ] = (
            stats[cluster_id]["percentage"] /
            baseline_stats[cluster_id]["percentage"]
            if baseline_stats[cluster_id]["percentage"] > 0
            else np.nan
        )

    pd.DataFrame(
        [plains_summary]
    ).to_csv(
        "systematic_intra_tessera_plains_validation.csv",
        index=False
    )


# ============================================================
# BRUSHED UNITS
# ============================================================

print("\n")
print("=" * 100)
print("BRUSHED UNIT VALIDATION")
print("=" * 100)

brushed_results = []


for idx, row in brushed.iterrows():

    geometry = row.geometry

    if geometry is None or geometry.is_empty:

        continue

    def geographic_to_k4(x, y, z=None):

        return lonlat_to_xy(x, y)

    geometry_k4 = shapely_transform(
        geographic_to_k4,
        geometry
    )

    raster_bounds = rasterio.windows.bounds(
        cluster_window,
        rasterio.open(CLUSTER_RASTER).transform
    )

    raster_box = box(
        raster_bounds[0],
        raster_bounds[1],
        raster_bounds[2],
        raster_bounds[3]
    )

    if not geometry_k4.intersects(raster_box):

        continue

    geometry_k4 = geometry_k4.intersection(
        raster_box
    )

    if geometry_k4.is_empty:

        continue

    mask = feature_mask_for_geometry(
        geometry_k4,
        cluster_data.shape,
        cluster_transform
    )

    values = cluster_data[mask]

    values = values[
        (values >= 0) &
        (values <= 3)
    ]

    if len(values) == 0:

        continue

    stats = cluster_statistics(values)

    dominant, purity = dominant_cluster(values)

    area_km2 = (
        len(values) *
        (PIXEL_SIZE_M / 1000.0) ** 2
    )

    unit_id = row.get(
        "id",
        idx
    )

    print(
        f"\nBrushed unit {unit_id}"
    )

    print(
        f"  Area: {area_km2:.1f} km²"
    )

    print(
        f"  Dominant cluster: C{dominant}"
    )

    print(
        f"  Purity: {purity:.2f}%"
    )

    result = {
        "feature_index": idx,
        "id": unit_id,
        "area_km2": area_km2,
        "dominant_cluster": dominant,
        "purity": purity,
        "valid_pixels": len(values)
    }

    for cluster_id in CLUSTERS:

        pct = stats[cluster_id]["percentage"]

        baseline_pct = baseline_stats[
            cluster_id
        ]["percentage"]

        enrichment = (
            pct / baseline_pct
            if baseline_pct > 0
            else np.nan
        )

        result[
            f"cluster_{cluster_id}_percentage"
        ] = pct

        result[
            f"cluster_{cluster_id}_enrichment"
        ] = enrichment

    brushed_results.append(
        result
    )


pd.DataFrame(
    brushed_results
).to_csv(
    "systematic_brushed_unit_validation.csv",
    index=False
)


# ============================================================
# LARGE CRATER VALIDATION
# ============================================================

print("\n")
print("=" * 100)
print("LARGE CRATER VALIDATION")
print("=" * 100)

crater_results = []


with rasterio.open(CLUSTER_RASTER) as src:

    transform = src.transform

    for idx, row in craters.iterrows():

        geometry = row.geometry

        if geometry is None or geometry.is_empty:

            continue

        lon = geometry.x
        lat = geometry.y

        x, y = lonlat_to_xy(
            lon,
            lat
        )

        try:

            raster_col, raster_row = (
                ~transform
            ) * (x, y)

        except Exception:

            continue

        raster_col = int(
            np.floor(raster_col)
        )

        raster_row = int(
            np.floor(raster_row)
        )

        col_start = max(
            0,
            raster_col - 1
        )

        col_end = min(
            src.width - 1,
            raster_col + 1
        )

        row_start = max(
            0,
            raster_row - 1
        )

        row_end = min(
            src.height - 1,
            raster_row + 1
        )

        width = (
            col_end -
            col_start +
            1
        )

        height = (
            row_end -
            row_start +
            1
        )

        window = rasterio.windows.Window(
            col_start,
            row_start,
            width,
            height
        )

        values = src.read(
            1,
            window=window
        )

        values = values[
            (values >= 0) &
            (values <= 3)
        ]

        if len(values) == 0:

            continue

        nearest_cluster = int(
            src.read(
                1,
                window=rasterio.windows.Window(
                    raster_col,
                    raster_row,
                    1,
                    1
                )
            )[0, 0]
        )

        crater_id = row.get(
            "Id",
            idx
        )

        print(
            f"\nCrater {crater_id}"
        )

        print(
            f"  Longitude: {lon:.4f}"
        )

        print(
            f"  Latitude:  {lat:.4f}"
        )

        print(
            f"  Central cluster: "
            f"C{nearest_cluster}"
        )

        crater_results.append({
            "feature_index": idx,
            "id": crater_id,
            "longitude": lon,
            "latitude": lat,
            "central_cluster": nearest_cluster
        })


pd.DataFrame(
    crater_results
).to_csv(
    "systematic_crater_validation.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("SYSTEMATIC GEOLOGICAL VALIDATION COMPLETE")
print("=" * 100)

print("\nFiles created:")

print("1. ovda_cluster_baseline.csv")
print("2. systematic_tectonomorphic_validation.csv")
print("3. systematic_intra_tessera_plains_validation.csv")
print("4. systematic_brushed_unit_validation.csv")
print("5. systematic_crater_validation.csv")

print("\n")
print("Interpretation rule:")

print(
    "Purity tells us how concentrated a geological feature "
    "is in a cluster."
)

print(
    "Coverage tells us how much of a cluster is represented "
    "by that geological feature."
)

print(
    "Enrichment compares the feature's cluster proportion "
    "against the overall Ovda baseline."
)

print(
    "\nDo NOT treat high purity alone as proof that a cluster "
    "is a geological unit."
)

print(
    "Use purity + coverage + enrichment + spatial context "
    "together."
)