import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import geometry_mask
from rasterio.windows import bounds as window_bounds

from shapely.geometry import box
from shapely.ops import transform as shapely_transform


# ============================================================
# CONFIGURATION
# ============================================================

from src.paths import K4_RASTER, GUINEVERE_GDB

CLUSTER_RASTER = K4_RASTER

GDB_PATH = GUINEVERE_GDB

GEOLOGICAL_LAYER = "V30Units16"

VENUS_RASTER_RADIUS = 6051000.0

USGS_VENUS_RADIUS = 6051800.0

CLUSTERS = [0, 1, 2, 3]

CLUSTER_NODATA = 255

PIXEL_SIZE_M = 2025.0


# ============================================================
# USGS GEOLOGICAL UNIT DEFINITIONS
# ============================================================
#
# These definitions come from the USGS Guinevere V-30
# geological map legend / pamphlet.
#
# Individual USGS codes are retained so that we do not
# confuse them with our ML cluster IDs.
# ============================================================

UNIT_DEFINITIONS = {

    "t": {
        "description": "Tessera",
        "category": "Tessera_Upland"
    },

    "ul": {
        "description": "Lineated upland material",
        "category": "Tessera_Upland"
    },

    "pGr": {
        "description": "Guinevere regional plains",
        "category": "Plains"
    },

    "pGlm": {
        "description": "Guinevere lineated and mottled plains",
        "category": "Plains"
    },

    "fpf": {
        "description": "Plains-forming flow material",
        "category": "Volcanic_Flow"
    },

    "fl": {
        "description": "Lobate flow material",
        "category": "Volcanic_Flow"
    },

    "fA1": {
        "description": "Atanua Mons lobate flow material, member 1",
        "category": "Volcanic_Flow"
    },

    "fA2": {
        "description": "Atanua Mons lobate flow material, member 2",
        "category": "Volcanic_Flow"
    },

    "fT": {
        "description": "Tuli Mons lobate flow material",
        "category": "Volcanic_Flow"
    },

    "fV": {
        "description": "Var Mons lobate flow material",
        "category": "Volcanic_Flow"
    },

    "fU": {
        "description": "Uilata Fluctus lobate flow material",
        "category": "Volcanic_Flow"
    },

    "fR": {
        "description": "Rhpisunt Mons lobate flow material",
        "category": "Volcanic_Flow"
    },

    "fVl": {
        "description": "Var Mons lineated material",
        "category": "Volcanic_Flow"
    },

    "v": {
        "description": "Small volcanic edifice",
        "category": "Volcanic_Edifice"
    },

    "c": {
        "description": "Impact crater material",
        "category": "Impact_Crater"
    }
}


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 110)
print("USGS GUINEVERE V-30 POLYGON-LEVEL GEOLOGICAL VALIDATION")
print("=" * 110)

if not os.path.exists(CLUSTER_RASTER):

    raise FileNotFoundError(
        f"Cluster raster not found:\n{CLUSTER_RASTER}"
    )

if not os.path.exists(GDB_PATH):

    raise FileNotFoundError(
        f"USGS geodatabase not found:\n{GDB_PATH}"
    )

print("\nInput files found.")


# ============================================================
# VENUS MERCATOR → LONGITUDE/LATITUDE
# ============================================================

def usgs_mercator_to_lonlat(
    x,
    y
):
    """
    Inverse spherical Mercator conversion.

    The USGS V-30 database uses a Venus Mercator projection
    with central meridian 0 and standard parallel 0.

    x = R * longitude_radians

    y = R * ln(tan(pi/4 + latitude/2))
    """

    x = np.asarray(x)
    y = np.asarray(y)

    longitude = (
        x /
        USGS_VENUS_RADIUS
    )

    latitude = (
        2.0 *
        np.arctan(
            np.exp(
                y /
                USGS_VENUS_RADIUS
            )
        )
        - np.pi / 2.0
    )

    return (
        np.rad2deg(longitude),
        np.rad2deg(latitude)
    )


# ============================================================
# LONGITUDE/LATITUDE → K4 EQUIRECTANGULAR
# ============================================================

def lonlat_to_k4(
    x,
    y,
    z=None
):

    longitude = np.deg2rad(
        np.asarray(x)
    )

    latitude = np.deg2rad(
        np.asarray(y)
    )

    output_x = (
        VENUS_RASTER_RADIUS *
        longitude
    )

    output_y = (
        VENUS_RASTER_RADIUS *
        latitude
    )

    return (
        output_x,
        output_y
    )


# ============================================================
# OPEN CLUSTER RASTER
# ============================================================

print("\nOpening K=4 classification...")

with rasterio.open(
    CLUSTER_RASTER
) as src:

    print(
        f"Raster dimensions: "
        f"{src.width:,} x "
        f"{src.height:,}"
    )

    print(
        f"Resolution: "
        f"{src.res}"
    )

    print(
        f"CRS: "
        f"{src.crs}"
    )

    print(
        f"NoData: "
        f"{src.nodata}"
    )

    raster_transform = src.transform

    raster_width = src.width

    raster_height = src.height

    raster_bounds = src.bounds


    # ========================================================
    # LOAD USGS GEOLOGICAL LAYER
    # ========================================================

    print("\nLoading USGS geological layer...")

    units = gpd.read_file(
        GDB_PATH,
        layer=GEOLOGICAL_LAYER
    )

    print(
        f"USGS polygons: "
        f"{len(units):,}"
    )

    print(
        f"USGS CRS:\n"
        f"{units.crs}"
    )


    # ========================================================
    # CHECK TYPE FIELD
    # ========================================================

    if "Type" not in units.columns:

        raise RuntimeError(
            "Expected 'Type' field was not found."
        )

    print("\n")
    print("=" * 110)
    print("USGS UNIT TYPE COUNTS")
    print("=" * 110)

    type_counts = (
        units["Type"]
        .value_counts(dropna=False)
    )

    print(
        type_counts.to_string()
    )


    # ========================================================
    # UNKNOWN TYPES
    # ========================================================

    known_types = set(
        UNIT_DEFINITIONS.keys()
    )

    observed_types = set(
        units["Type"]
        .dropna()
        .astype(str)
        .unique()
    )

    unknown_types = (
        observed_types -
        known_types
    )


    print("\n")
    print("=" * 110)
    print("UNKNOWN USGS TYPES")
    print("=" * 110)

    if unknown_types:

        for unit_type in sorted(
            unknown_types
        ):

            print(
                f"WARNING: {unit_type}"
            )

    else:

        print(
            "All observed Type codes are "
            "recognized."
        )


    # ========================================================
    # BASELINE — GUINEVERE WINDOW
    # ========================================================

    #
    # V-30 spans approximately:
    #
    # 300°E – 330°E
    # 0°N – 25°N
    #
    # We use the same geographic window used in our
    # previous regional characterization.
    #

    print("\n")
    print("=" * 110)
    print("GUINEVERE BASELINE")
    print("=" * 110)


    columns = np.arange(
        raster_width
    )

    xs = (
        raster_transform.c +
        raster_transform.a *
        (columns + 0.5)
    )

    raster_lons = np.rad2deg(
        xs /
        VENUS_RASTER_RADIUS
    )

    raster_lons = (
        (
            raster_lons +
            180.0
        )
        % 360.0
    ) - 180.0

    raster_lons_360 = (
        raster_lons %
        360.0
    )


    rows = np.arange(
        raster_height
    )

    ys = (
        raster_transform.f +
        raster_transform.e *
        (rows + 0.5)
    )

    raster_lats = np.rad2deg(
        ys /
        VENUS_RASTER_RADIUS
    )


    guinevere_lon_mask = (
        (raster_lons_360 >= 300.0)
        &
        (raster_lons_360 <= 330.0)
    )

    guinevere_lat_mask = (
        (raster_lats >= 0.0)
        &
        (raster_lats <= 25.0)
    )

    guinevere_cols = np.where(
        guinevere_lon_mask
    )[0]

    guinevere_rows = np.where(
        guinevere_lat_mask
    )[0]


    if (
        len(guinevere_cols) == 0
        or
        len(guinevere_rows) == 0
    ):

        raise RuntimeError(
            "Could not find Guinevere raster window."
        )


    col_start = (
        guinevere_cols.min()
    )

    col_end = (
        guinevere_cols.max() +
        1
    )

    row_start = (
        guinevere_rows.min()
    )

    row_end = (
        guinevere_rows.max() +
        1
    )


    print(
        f"Raster window:"
        f" rows {row_start}–{row_end}, "
        f"columns {col_start}–{col_end}"
    )


    cluster_window = rasterio.windows.Window(
        col_start,
        row_start,
        col_end - col_start,
        row_end - row_start
    )

    guinevere_clusters = src.read(
        1,
        window=cluster_window
    )

    guinevere_transform = (
        src.window_transform(
            cluster_window
        )
    )


    valid_mask = (
        (guinevere_clusters >= 0)
        &
        (guinevere_clusters <= 3)
    )

    valid_values = (
        guinevere_clusters[
            valid_mask
        ]
    )


    baseline_counts = {}

    for cluster_id in CLUSTERS:

        baseline_counts[
            cluster_id
        ] = np.sum(
            valid_values ==
            cluster_id
        )

    total_valid = len(
        valid_values
    )


    print("\nGuinevere baseline:")

    baseline_percentages = {}

    for cluster_id in CLUSTERS:

        percentage = (
            100.0 *
            baseline_counts[
                cluster_id
            ] /
            total_valid
        )

        baseline_percentages[
            cluster_id
        ] = percentage

        print(
            f"C{cluster_id}: "
            f"{baseline_counts[cluster_id]:,} "
            f"pixels "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # TRANSFORM USGS POLYGONS
    # ========================================================

    print("\n")
    print("=" * 110)
    print("TRANSFORMING USGS POLYGONS")
    print("=" * 110)

    transformed_geometries = []

    for idx, geometry in enumerate(
        units.geometry
    ):

        if (
            geometry is None
            or
            geometry.is_empty
        ):

            transformed_geometries.append(
                None
            )

            continue


        # ----------------------------------------------------
        # USGS Mercator x/y
        # → Venus lon/lat
        # → K4 equirectangular x/y
        # ----------------------------------------------------

        geometry_k4 = shapely_transform(
            lambda x, y, z=None:
                lonlat_to_k4(
                    *usgs_mercator_to_lonlat(
                        x,
                        y
                    ),
                    z
                ),
            geometry
        )

        transformed_geometries.append(
            geometry_k4
        )


    units["geometry_k4"] = (
        transformed_geometries
    )


    # ========================================================
    # RASTER EXTENT
    # ========================================================

    k4_bounds = window_bounds(
        cluster_window,
        raster_transform
    )

    raster_box = box(
        k4_bounds[0],
        k4_bounds[1],
        k4_bounds[2],
        k4_bounds[3]
    )


    # ========================================================
    # POLYGON-LEVEL VALIDATION
    # ========================================================

    print("\n")
    print("=" * 110)
    print("POLYGON-LEVEL VALIDATION")
    print("=" * 110)


    polygon_results = []


    for idx, row in units.iterrows():

        unit_type = str(
            row["Type"]
        )

        geometry = (
            row["geometry_k4"]
        )

        if (
            geometry is None
            or
            geometry.is_empty
        ):

            continue


        # ----------------------------------------------------
        # Restrict to Guinevere raster extent
        # ----------------------------------------------------

        if not geometry.intersects(
            raster_box
        ):

            continue


        clipped = geometry.intersection(
            raster_box
        )

        if clipped.is_empty:

            continue


        # ----------------------------------------------------
        # Rasterize polygon onto K=4 grid
        # ----------------------------------------------------

        mask = geometry_mask(
            [clipped],
            out_shape=
                guinevere_clusters.shape,
            transform=
                guinevere_transform,
            invert=True,
            all_touched=False
        )


        values = guinevere_clusters[
            mask
        ]

        values = values[
            (values >= 0)
            &
            (values <= 3)
        ]


        if len(values) == 0:

            continue


        # ----------------------------------------------------
        # Cluster proportions
        # ----------------------------------------------------

        counts = {}

        percentages = {}

        enrichments = {}


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


            counts[
                cluster_id
            ] = count

            percentages[
                cluster_id
            ] = percentage

            enrichments[
                cluster_id
            ] = enrichment


        dominant_cluster = max(
            CLUSTERS,
            key=lambda c:
                percentages[c]
        )

        dominant_purity = (
            percentages[
                dominant_cluster
            ]
        )


        # ----------------------------------------------------
        # Pixel-based area estimate
        # ----------------------------------------------------

        area_km2 = (
            len(values)
            *
            (
                PIXEL_SIZE_M /
                1000.0
            ) ** 2
        )


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if unit_type in (
            UNIT_DEFINITIONS
        ):

            description = (
                UNIT_DEFINITIONS[
                    unit_type
                ]["description"]
            )

            category = (
                UNIT_DEFINITIONS[
                    unit_type
                ]["category"]
            )

        else:

            description = (
                "Unknown USGS Type"
            )

            category = (
                "Unknown"
            )


        result = {
            "feature_index": idx,
            "orig_fid": row.get(
                "ORIG_FID",
                np.nan
            ),
            "usgs_type": unit_type,
            "description": description,
            "category": category,
            "valid_pixels": len(values),
            "area_km2": area_km2,
            "dominant_cluster":
                dominant_cluster,
            "dominant_purity":
                dominant_purity
        }


        for cluster_id in CLUSTERS:

            result[
                f"cluster_{cluster_id}_pixels"
            ] = counts[
                cluster_id
            ]

            result[
                f"cluster_{cluster_id}_percentage"
            ] = percentages[
                cluster_id
            ]

            result[
                f"cluster_{cluster_id}_enrichment"
            ] = enrichments[
                cluster_id
            ]


        polygon_results.append(
            result
        )


        print(
            f"{unit_type:>4} | "
            f"{description:<55} | "
            f"dominant C{dominant_cluster} | "
            f"purity {dominant_purity:6.2f}%"
        )


    # ========================================================
    # SAVE POLYGON RESULTS
    # ========================================================

    polygon_df = pd.DataFrame(
        polygon_results
    )


    polygon_df.to_csv(
        "guinevere_usgs_polygon_validation.csv",
        index=False
    )


    # ========================================================
    # AGGREGATE BY USGS UNIT
    # ========================================================

    print("\n")
    print("=" * 110)
    print("AGGREGATION BY USGS UNIT TYPE")
    print("=" * 110)


    unit_rows = []


    for unit_type, group in (
        polygon_df.groupby(
            "usgs_type"
        )
    ):

        category = group[
            "category"
        ].iloc[0]

        description = group[
            "description"
        ].iloc[0]


        total_pixels = (
            group[
                "valid_pixels"
            ].sum()
        )


        row = {
            "usgs_type": unit_type,
            "description": description,
            "category": category,
            "polygons": len(group),
            "valid_pixels": total_pixels
        }


        for cluster_id in CLUSTERS:

            total_cluster_pixels = (
                group[
                    f"cluster_{cluster_id}_pixels"
                ].sum()
            )

            percentage = (
                100.0 *
                total_cluster_pixels /
                total_pixels
            )

            enrichment = (
                percentage /
                baseline_percentages[
                    cluster_id
                ]
            )


            row[
                f"cluster_{cluster_id}_percentage"
            ] = percentage

            row[
                f"cluster_{cluster_id}_enrichment"
            ] = enrichment


        dominant = max(
            CLUSTERS,
            key=lambda c:
                row[
                    f"cluster_{c}_percentage"
                ]
        )


        row[
            "dominant_cluster"
        ] = dominant

        row[
            "dominant_purity"
        ] = row[
            f"cluster_{dominant}_percentage"
        ]


        unit_rows.append(
            row
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
        "guinevere_usgs_unit_validation.csv",
        index=False
    )


    # ========================================================
    # AGGREGATE BY BROAD GEOLOGICAL CATEGORY
    # ========================================================

    print("\n")
    print("=" * 110)
    print("AGGREGATION BY BROAD GEOLOGICAL CATEGORY")
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


        row = {
            "category": category,
            "polygons": len(group),
            "valid_pixels": total_pixels
        }


        for cluster_id in CLUSTERS:

            cluster_pixels = (
                group[
                    f"cluster_{cluster_id}_pixels"
                ].sum()
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


            row[
                f"cluster_{cluster_id}_percentage"
            ] = percentage

            row[
                f"cluster_{cluster_id}_enrichment"
            ] = enrichment


        dominant = max(
            CLUSTERS,
            key=lambda c:
                row[
                    f"cluster_{c}_percentage"
                ]
        )


        row[
            "dominant_cluster"
        ] = dominant

        row[
            "dominant_purity"
        ] = row[
            f"cluster_{dominant}_percentage"
        ]


        category_rows.append(
            row
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
        "guinevere_geological_category_validation.csv",
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("GUINEVERE USGS POLYGON VALIDATION COMPLETE")
print("=" * 110)

print("\nFiles created:")

print(
    "1. guinevere_usgs_polygon_validation.csv"
)

print(
    "2. guinevere_usgs_unit_validation.csv"
)

print(
    "3. guinevere_geological_category_validation.csv"
)

print("\nThe key result is the broad CATEGORY table.")

print(
    "\nWe will compare:"
)

print(
    "Tessera/Upland"
)

print(
    "Plains"
)

print(
    "Volcanic Flow"
)

print(
    "Volcanic Edifice"
)

print(
    "Impact Crater"
)

print(
    "\nagainst our four ML clusters."
)