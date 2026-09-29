import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER, LADA_GDB
import matplotlib.pyplot as plt

from rasterio.features import geometry_mask
from pyproj import Transformer
from shapely.ops import transform as shapely_transform


# ============================================================
# CONFIGURATION
# ============================================================

CLUSTER_RASTER = str(K4_RASTER)

GDB_PATH = str(LADA_GDB)

GEOLOGICAL_LAYER = "Geologic_Units_V56"

VENUS_RADIUS = 6051000.0

CLUSTERS = [0, 1, 2, 3]

# Lada V-56
LON_MIN = 0.0
LON_MAX = 60.0
LAT_MIN = -70.0
LAT_MAX = -50.0

OUTPUT_POLYGONS = "lada_usgs_polygon_validation.csv"
OUTPUT_UNITS = "lada_usgs_unit_validation_corrected.csv"
OUTPUT_CATEGORIES = "lada_usgs_category_validation_corrected.csv"
OUTPUT_C2 = "lada_c2_geology_summary_corrected.csv"

OUTPUT_OVERLAY = "lada_usgs_overlay.png"
OUTPUT_TESSERA_OVERLAY = "lada_tessera_overlay.png"


# ============================================================
# EXPLICIT USGS UNIT CLASSIFICATION
# ============================================================
#
# IMPORTANT:
# We use the actual USGS Unit code.
#
# We do NOT classify solely by searching words in UnitName.
#
# The categories are analytical groupings for this project,
# while the original USGS unit and UnitName are preserved.
# ============================================================

UNIT_CATEGORY = {

    # --------------------------------------------------------
    # Tessera / tessera-related
    # --------------------------------------------------------

    "t": "Tessera",

    "tlt": "Tessera_like",

    "itbl": "Intra_tessera_basin",

    "itbm": "Intra_tessera_basin",

    "itbu": "Intra_tessera_basin",

    # --------------------------------------------------------
    # Tectonic / lineated terrain
    # --------------------------------------------------------

    "tdl": "Densely_lineated_terrain",

    # --------------------------------------------------------
    # Plains
    # --------------------------------------------------------

    "plu": "Plains",

    "plm": "Plains",

    "pll": "Plains",

    "pr": "Plains",

    "pwr": "Plains",

    "psh": "Plains",

    # --------------------------------------------------------
    # Corona-associated plains
    # --------------------------------------------------------

    "pshc": "Shield_plains_corona",

    # --------------------------------------------------------
    # Corona
    # --------------------------------------------------------

    "cp": "Corona",

    # --------------------------------------------------------
    # Impact crater materials
    # --------------------------------------------------------

    "ce": "Impact_crater",

    "cf": "Impact_crater",

    "cp": "Impact_crater",

    "cw": "Impact_crater"
}


# ============================================================
# CHECK INPUTS
# ============================================================

print("=" * 110)
print("LADA V-56 CORRECTED USGS GEOLOGICAL VALIDATION")
print("=" * 110)

if not os.path.exists(CLUSTER_RASTER):

    raise FileNotFoundError(
        f"Cluster raster not found:\n{CLUSTER_RASTER}"
    )

if not os.path.exists(GDB_PATH):

    raise FileNotFoundError(
        f"Lada geodatabase not found:\n{GDB_PATH}"
    )

print("\nInput files found.")


# ============================================================
# OPEN K-MEANS RASTER
# ============================================================

print("\nOpening K=4 classification...")

with rasterio.open(CLUSTER_RASTER) as src:

    raster_width = src.width
    raster_height = src.height
    raster_transform = src.transform

    print(
        f"Raster size: "
        f"{raster_width:,} x "
        f"{raster_height:,}"
    )

    print(
        f"Resolution: "
        f"{src.res}"
    )

    # --------------------------------------------------------
    # Calculate geographic location of raster pixels
    # --------------------------------------------------------

    columns = np.arange(
        raster_width
    )

    xs = (
        raster_transform.c
        +
        raster_transform.a *
        (columns + 0.5)
    )

    raster_longitudes = np.rad2deg(
        xs /
        VENUS_RADIUS
    )

    raster_longitudes = (
        (
            raster_longitudes +
            180.0
        )
        % 360.0
    ) - 180.0

    raster_longitudes_360 = (
        raster_longitudes %
        360.0
    )


    rows = np.arange(
        raster_height
    )

    ys = (
        raster_transform.f
        +
        raster_transform.e *
        (rows + 0.5)
    )

    raster_latitudes = np.rad2deg(
        ys /
        VENUS_RADIUS
    )


    # --------------------------------------------------------
    # Select Lada window
    # --------------------------------------------------------

    longitude_mask = (
        (raster_longitudes_360 >= LON_MIN)
        &
        (raster_longitudes_360 <= LON_MAX)
    )

    latitude_mask = (
        (raster_latitudes >= LAT_MIN)
        &
        (raster_latitudes <= LAT_MAX)
    )

    selected_cols = np.where(
        longitude_mask
    )[0]

    selected_rows = np.where(
        latitude_mask
    )[0]

    if (
        len(selected_cols) == 0
        or
        len(selected_rows) == 0
    ):

        raise RuntimeError(
            "Could not find Lada raster window."
        )

    col_start = selected_cols.min()
    col_end = selected_cols.max() + 1

    row_start = selected_rows.min()
    row_end = selected_rows.max() + 1

    window = rasterio.windows.Window(
        col_start,
        row_start,
        col_end - col_start,
        row_end - row_start
    )

    cluster_data = src.read(
        1,
        window=window
    )

    cluster_transform = (
        src.window_transform(
            window
        )
    )

    print(
        f"\nLada raster window:"
    )

    print(
        f"  Rows: "
        f"{row_start}â€“{row_end}"
    )

    print(
        f"  Columns: "
        f"{col_start}â€“{col_end}"
    )

    print(
        f"  Shape: "
        f"{cluster_data.shape}"
    )


# ============================================================
# LADA BASELINE
# ============================================================

valid_mask = (
    (cluster_data >= 0)
    &
    (cluster_data <= 3)
)

valid_values = cluster_data[
    valid_mask
]

print("\n")
print("=" * 110)
print("LADA BASELINE")
print("=" * 110)

baseline_percentages = {}

for cluster_id in CLUSTERS:

    count = np.sum(
        valid_values ==
        cluster_id
    )

    percentage = (
        100.0 *
        count /
        len(valid_values)
    )

    baseline_percentages[
        cluster_id
    ] = percentage

    print(
        f"C{cluster_id}: "
        f"{count:,} pixels "
        f"({percentage:.2f}%)"
    )


# ============================================================
# LOAD USGS GEOLOGY
# ============================================================

print("\n")
print("=" * 110)
print("LOADING USGS GEOLOGICAL UNITS")
print("=" * 110)

units = gpd.read_file(
    GDB_PATH,
    layer=GEOLOGICAL_LAYER
)

print(
    f"\nFeatures: "
    f"{len(units):,}"
)

print(
    f"CRS:\n{units.crs}"
)


# ============================================================
# EXPLICIT CATEGORY ASSIGNMENT
# ============================================================

units["analytical_category"] = (
    units["Unit"]
    .astype(str)
    .map(UNIT_CATEGORY)
)

unknown_units = units[
    units["analytical_category"].isna()
]

print("\n")
print("=" * 110)
print("UNKNOWN USGS UNIT CODES")
print("=" * 110)

if len(unknown_units) == 0:

    print(
        "None."
    )

else:

    print(
        unknown_units[
            "Unit"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# CRS TRANSFORMATION
# ============================================================

print("\n")
print("=" * 110)
print("SETTING UP CRS TRANSFORMATION")
print("=" * 110)

source_crs = units.crs

if source_crs is None:

    raise RuntimeError(
        "USGS geological layer has no CRS."
    )


geographic_crs = (
    source_crs.geodetic_crs
)

print(
    "\nUSGS geographic CRS:"
)

print(
    geographic_crs
)


# ------------------------------------------------------------
# Lada Lambert â†’ Venus geographic lon/lat
# ------------------------------------------------------------

to_geographic = Transformer.from_crs(
    source_crs,
    geographic_crs,
    always_xy=True
)


# ============================================================
# VENUS LON/LAT â†’ K4 EQUIRECTANGULAR
# ============================================================

def geographic_to_k4(
    x,
    y,
    z=None
):

    longitude = np.asarray(
        x,
        dtype=np.float64
    )

    latitude = np.asarray(
        y,
        dtype=np.float64
    )

    longitude_rad = np.deg2rad(
        longitude
    )

    latitude_rad = np.deg2rad(
        latitude
    )

    x_out = (
        VENUS_RADIUS *
        longitude_rad
    )

    y_out = (
        VENUS_RADIUS *
        latitude_rad
    )

    return (
        x_out,
        y_out
    )


def transform_to_k4(
    geometry
):

    if (
        geometry is None
        or
        geometry.is_empty
    ):

        return None


    # Lada projected â†’ Venus lon/lat
    geographic_geometry = (
        shapely_transform(
            to_geographic.transform,
            geometry
        )
    )


    # Venus lon/lat â†’ K4 projection
    k4_geometry = (
        shapely_transform(
            geographic_to_k4,
            geographic_geometry
        )
    )


    if not k4_geometry.is_valid:

        k4_geometry = (
            k4_geometry.buffer(0)
        )


    if k4_geometry.is_empty:

        return None


    return k4_geometry


# ============================================================
# TRANSFORM ALL POLYGONS
# ============================================================

print("\n")
print("=" * 110)
print("TRANSFORMING USGS POLYGONS")
print("=" * 110)

transformed = []

failed = 0

for i, geometry in enumerate(
    units.geometry
):

    try:

        transformed_geometry = (
            transform_to_k4(
                geometry
            )
        )

        transformed.append(
            transformed_geometry
        )

    except Exception as exc:

        failed += 1

        transformed.append(
            None
        )

        print(
            f"WARNING polygon {i}: "
            f"{exc}"
        )

    if (
        (i + 1) % 100 == 0
    ):

        print(
            f"Processed "
            f"{i + 1:,} / "
            f"{len(units):,}"
        )


units[
    "geometry_k4"
] = transformed


print(
    f"\nSuccessful: "
    f"{len(units) - failed:,}"
)

print(
    f"Failed: "
    f"{failed:,}"
)


# ============================================================
# POLYGON VALIDATION
# ============================================================

print("\n")
print("=" * 110)
print("POLYGON-LEVEL VALIDATION")
print("=" * 110)

polygon_results = []


for index, row in units.iterrows():

    geometry = row[
        "geometry_k4"
    ]

    if (
        geometry is None
        or
        geometry.is_empty
    ):

        continue


    try:

        mask = geometry_mask(
            [geometry],
            out_shape=
                cluster_data.shape,
            transform=
                cluster_transform,
            invert=True,
            all_touched=False
        )

    except Exception as exc:

        print(
            f"WARNING polygon {index}: "
            f"{exc}"
        )

        continue


    values = cluster_data[
        mask
    ]

    values = values[
        (values >= 0)
        &
        (values <= 3)
    ]


    if len(values) == 0:

        continue


    result = {

        "feature_index":
            index,

        "unit":
            row["Unit"],

        "unit_name":
            row["UnitName"],

        "category":
            row["analytical_category"],

        "usgs_spherical_area_km2":
            row.get(
                "Sph_Areakm2",
                np.nan
            ),

        "valid_pixels":
            len(values)
    }


    cluster_percentages = {}


    for cluster_id in CLUSTERS:

        count = np.sum(
            values ==
            cluster_id
        )

        percentage = (
            100.0 *
            count /
            len(values)
        )

        baseline = (
            baseline_percentages[
                cluster_id
            ]
        )

        enrichment = (
            percentage /
            baseline
            if baseline > 0
            else np.nan
        )

        result[
            f"cluster_{cluster_id}_pixels"
        ] = int(count)

        result[
            f"cluster_{cluster_id}_percentage"
        ] = percentage

        result[
            f"cluster_{cluster_id}_enrichment"
        ] = enrichment

        cluster_percentages[
            cluster_id
        ] = percentage


    dominant_cluster = max(
        CLUSTERS,
        key=lambda c:
            cluster_percentages[c]
    )


    result[
        "dominant_cluster"
    ] = dominant_cluster

    result[
        "dominant_purity"
    ] = cluster_percentages[
        dominant_cluster
    ]


    polygon_results.append(
        result
    )


polygon_df = pd.DataFrame(
    polygon_results
)


print(
    f"\nValidated polygons: "
    f"{len(polygon_df):,}"
)


if len(polygon_df) == 0:

    raise RuntimeError(
        "Zero polygons overlapped the Lada raster."
    )


polygon_df.to_csv(
    OUTPUT_POLYGONS,
    index=False
)


# ============================================================
# AGGREGATE BY EXACT USGS UNIT
# ============================================================

print("\n")
print("=" * 110)
print("EXACT USGS UNIT VALIDATION")
print("=" * 110)

unit_rows = []


for (unit, unit_name, category), group in (
    polygon_df.groupby(
        [
            "unit",
            "unit_name",
            "category"
        ]
    )
):

    total_pixels = (
        group[
            "valid_pixels"
        ].sum()
    )


    if total_pixels == 0:

        continue


    result = {

        "unit":
            unit,

        "unit_name":
            unit_name,

        "category":
            category,

        "polygons":
            len(group),

        "valid_pixels":
            total_pixels
    }


    for cluster_id in CLUSTERS:

        cluster_pixels = (
            group[
                f"cluster_{cluster_id}_pixels"
            ]
            .sum()
        )

        percentage = (
            100.0 *
            cluster_pixels /
            total_pixels
        )

        enrichment = (
            percentage /
            baseline_percentages[
                cluster_id
            ]
        )

        result[
            f"cluster_{cluster_id}_percentage"
        ] = percentage

        result[
            f"cluster_{cluster_id}_enrichment"
        ] = enrichment


    dominant = max(
        CLUSTERS,
        key=lambda c:
            result[
                f"cluster_{c}_percentage"
            ]
    )

    result[
        "dominant_cluster"
    ] = dominant

    result[
        "dominant_purity"
    ] = result[
        f"cluster_{dominant}_percentage"
    ]


    unit_rows.append(
        result
    )


unit_df = pd.DataFrame(
    unit_rows
)


print(
    unit_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


unit_df.to_csv(
    OUTPUT_UNITS,
    index=False
)


# ============================================================
# AGGREGATE BY ANALYTICAL CATEGORY
# ============================================================

print("\n")
print("=" * 110)
print("CORRECTED BROAD GEOLOGICAL CATEGORIES")
print("=" * 110)

category_rows = []


for category, group in (
    polygon_df.groupby(
        "category"
    )
):

    total_pixels = (
        group[
            "valid_pixels"
        ].sum()
    )


    if total_pixels == 0:

        continue


    result = {

        "category":
            category,

        "polygons":
            len(group),

        "valid_pixels":
            total_pixels
    }


    for cluster_id in CLUSTERS:

        cluster_pixels = (
            group[
                f"cluster_{cluster_id}_pixels"
            ]
            .sum()
        )

        percentage = (
            100.0 *
            cluster_pixels /
            total_pixels
        )

        enrichment = (
            percentage /
            baseline_percentages[
                cluster_id
            ]
        )


        result[
            f"cluster_{cluster_id}_percentage"
        ] = percentage

        result[
            f"cluster_{cluster_id}_enrichment"
        ] = enrichment


    dominant = max(
        CLUSTERS,
        key=lambda c:
            result[
                f"cluster_{c}_percentage"
            ]
    )


    result[
        "dominant_cluster"
    ] = dominant

    result[
        "dominant_purity"
    ] = result[
        f"cluster_{dominant}_percentage"
    ]


    category_rows.append(
        result
    )


category_df = pd.DataFrame(
    category_rows
)


print(
    category_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


category_df.to_csv(
    OUTPUT_CATEGORIES,
    index=False
)


# ============================================================
# C2-FOCUSED ANALYSIS
# ============================================================

print("\n")
print("=" * 110)
print("C2-FOCUSED GEOLOGICAL ANALYSIS")
print("=" * 110)

c2_rows = []


for category, group in (
    polygon_df.groupby(
        "category"
    )
):

    total_pixels = (
        group[
            "valid_pixels"
        ].sum()
    )

    c2_pixels = (
        group[
            "cluster_2_pixels"
        ].sum()
    )

    c2_percentage = (
        100.0 *
        c2_pixels /
        total_pixels
    )

    c2_enrichment = (
        c2_percentage /
        baseline_percentages[2]
    )


    c2_rows.append({

        "category":
            category,

        "polygons":
            len(group),

        "valid_pixels":
            total_pixels,

        "c2_percentage":
            c2_percentage,

        "c2_enrichment":
            c2_enrichment,

        "mean_polygon_c2_percentage":
            group[
                "cluster_2_percentage"
            ].mean(),

        "median_polygon_c2_percentage":
            group[
                "cluster_2_percentage"
            ].median(),

        "maximum_polygon_c2_percentage":
            group[
                "cluster_2_percentage"
            ].max()
    })


c2_df = pd.DataFrame(
    c2_rows
)


print(
    c2_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


c2_df.to_csv(
    OUTPUT_C2,
    index=False
)


# ============================================================
# TOP C2-RICH UNITS
# ============================================================

print("\n")
print("=" * 110)
print("TOP C2-RICH USGS UNITS")
print("=" * 110)

top_units = (
    unit_df
    .sort_values(
        [
            "cluster_2_percentage",
            "valid_pixels"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(30)
)


for _, row in top_units.iterrows():

    print(
        f"{str(row['unit']):>8} | "
        f"{str(row['unit_name']):<45} | "
        f"{str(row['category']):<28} | "
        f"C2={row['cluster_2_percentage']:7.2f}% | "
        f"enrichment="
        f"{row['cluster_2_enrichment']:7.2f}x | "
        f"pixels="
        f"{row['valid_pixels']:,}"
    )


# ============================================================
# CREATE LADA OVERLAY
# ============================================================

print("\n")
print("=" * 110)
print("CREATING LADA GEOLOGICAL OVERLAY")
print("=" * 110)


# ------------------------------------------------------------
# Geographic coordinate arrays for the Lada crop
# ------------------------------------------------------------

crop_columns = np.arange(
    col_start,
    col_end
)

crop_rows = np.arange(
    row_start,
    row_end
)

crop_x = (
    raster_transform.c
    +
    raster_transform.a *
    (crop_columns + 0.5)
)

crop_y = (
    raster_transform.f
    +
    raster_transform.e *
    (crop_rows + 0.5)
)

crop_lons = np.rad2deg(
    crop_x /
    VENUS_RADIUS
)

crop_lons = (
    (
        crop_lons +
        180.0
    )
    % 360.0
) - 180.0

crop_lons_360 = (
    crop_lons %
    360.0
)

crop_lats = np.rad2deg(
    crop_y /
    VENUS_RADIUS
)


# ------------------------------------------------------------
# Geographic extent
# ------------------------------------------------------------

extent = [

    np.min(crop_lons_360),

    np.max(crop_lons_360),

    np.min(crop_lats),

    np.max(crop_lats)
]


# ------------------------------------------------------------
# Plot all USGS unit boundaries
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 10)
)


ax.imshow(
    np.ma.masked_where(
        ~np.isin(
            cluster_data,
            CLUSTERS
        ),
        cluster_data
    ),
    extent=extent,
    origin="upper",
    interpolation="nearest",
    alpha=0.75
)


for _, row in units.iterrows():

    geometry = row[
        "geometry_k4"
    ]

    if (
        geometry is None
        or
        geometry.is_empty
    ):

        continue


    geo = gpd.GeoSeries(
        [
            geometry
        ]
    )


    geo.boundary.plot(
        ax=ax,
        linewidth=0.35
    )


ax.set_title(
    "Lada V-56: K-Means K=4 + USGS Geological Units"
)

ax.set_xlabel(
    "Longitude (Â°E)"
)

ax.set_ylabel(
    "Latitude (Â°)"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_OVERLAY,
    dpi=200
)

plt.close()


# ============================================================
# TESSERA OVERLAY
# ============================================================

print(
    "\nCreating tessera-focused overlay..."
)

fig, ax = plt.subplots(
    figsize=(14, 10)
)


ax.imshow(
    np.ma.masked_where(
        ~np.isin(
            cluster_data,
            CLUSTERS
        ),
        cluster_data
    ),
    extent=extent,
    origin="upper",
    interpolation="nearest",
    alpha=0.75
)


tessera_units = units[
    units["analytical_category"].isin(
        [
            "Tessera",
            "Tessera_like",
            "Intra_tessera_basin",
            "Densely_lineated_terrain"
        ]
    )
]


for _, row in tessera_units.iterrows():

    geometry = row[
        "geometry_k4"
    ]

    if (
        geometry is None
        or
        geometry.is_empty
    ):

        continue


    gpd.GeoSeries(
        [
            geometry
        ]
    ).boundary.plot(
        ax=ax,
        linewidth=1.2
    )


ax.set_title(
    "Lada V-56: K-Means K=4 + Tessera/Lineated Geological Units"
)

ax.set_xlabel(
    "Longitude (Â°E)"
)

ax.set_ylabel(
    "Latitude (Â°)"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_TESSERA_OVERLAY,
    dpi=200
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 110)
print("LADA VALIDATION COMPLETE")
print("=" * 110)

print("\nFiles created:")

print(
    f"1. {OUTPUT_POLYGONS}"
)

print(
    f"2. {OUTPUT_UNITS}"
)

print(
    f"3. {OUTPUT_CATEGORIES}"
)

print(
    f"4. {OUTPUT_C2}"
)

print(
    f"5. {OUTPUT_OVERLAY}"
)

print(
    f"6. {OUTPUT_TESSERA_OVERLAY}"
)

print(
    "\nThe most important results are:"
)

print(
    "CORRECTED BROAD GEOLOGICAL CATEGORIES"
)

print(
    "C2-FOCUSED GEOLOGICAL ANALYSIS"
)

print(
    "TOP C2-RICH USGS UNITS"
)
