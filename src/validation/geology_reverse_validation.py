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

CLUSTER_FILE = str(K4_RASTER)

GEOLOGY_FILE = str(OVDA_GEOLOGY_FILE)

VENUS_RADIUS = 6_051_000.0

MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0

NODATA = 255

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


print("=" * 70)
print("REVERSE GEOLOGICAL VALIDATION")
print("=" * 70)


with rasterio.open(CLUSTER_FILE) as src:

    raster_transform = src.transform

    raster_width = src.width

    raster_height = src.height

    raster_crs = src.crs

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

    window = from_bounds(

        min_x,
        min_y,
        max_x,
        max_y,

        transform=raster_transform

    ).round_offsets().round_lengths()

    cluster_data = src.read(
        1,
        window=window
    )


print("\nOvda cluster array:")
print(cluster_data.shape)

valid = (
    cluster_data != NODATA
)

valid_values = cluster_data[
    valid
]



cluster_totals = {}


print("\n" + "=" * 70)
print("OVDA CLUSTER TOTALS")
print("=" * 70)


for cluster in range(4):

    count = int(
        np.sum(
            valid_values == cluster
        )
    )

    cluster_totals[cluster] = count

    print(
        f"Cluster {cluster}: "
        f"{count:,} pixels"
    )


baseline = {}


for cluster in range(4):

    baseline[cluster] = (

        cluster_totals[cluster]

        /

        len(valid_values)

        *

        100

    )


print("\nOvda baseline:")

for cluster in range(4):

    print(
        f"Cluster {cluster}: "
        f"{baseline[cluster]:.2f}%"
    )



with rasterio.open(CLUSTER_FILE) as src:

    window_transform = rasterio.windows.transform(
        window,
        src.transform
    )



def analyze_polygon(
    name,
    geometry,
    feature_type
):

    print("\n" + "-" * 70)

    print(
        f"{feature_type}: {name}"
    )

    print("-" * 70)



    polygon_raster = rasterize(

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

        (polygon_raster == 1)

        &

        valid

    )


    values = cluster_data[
        mask
    ]


    total_feature_pixels = len(values)


    print(
        "Feature pixels:",
        f"{total_feature_pixels:,}"
    )


    if total_feature_pixels == 0:

        print(
            "No valid overlapping pixels."
        )

        return None


    record = {

        "feature_type": feature_type,

        "geological_feature": name,

        "feature_pixels": total_feature_pixels

    }



    for cluster in range(4):

        intersection = int(
            np.sum(
                values == cluster
            )
        )



        purity = (

            intersection

            /

            total_feature_pixels

            *

            100

        )



        coverage = (

            intersection

            /

            cluster_totals[cluster]

            *

            100

        )



        enrichment = (

            purity

            /

            baseline[cluster]

        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        record[
            f"cluster_{cluster}_pixels"
        ] = intersection


        record[
            f"cluster_{cluster}_purity_percent"
        ] = purity


        record[
            f"cluster_{cluster}_coverage_percent"
        ] = coverage


        record[
            f"cluster_{cluster}_enrichment"
        ] = enrichment


        print(

            f"\nCluster {cluster}"

        )

        print(

            f"  Intersection: "
            f"{intersection:,}"

        )

        print(

            f"  Purity: "
            f"{purity:.2f}%"

        )

        print(

            f"  Coverage: "
            f"{coverage:.4f}%"

        )

        print(

            f"  Enrichment: "
            f"{enrichment:.2f}x"

        )


    return record


print("\n" + "=" * 70)

print(
    "TECTONOMORPHIC UNITS"
)

print("=" * 70)


tectono = gpd.read_file(

    GEOLOGY_FILE,

    layer="Tectonomorphic_map"

)


tectono["geometry"] = (

    tectono["geometry"]
    .apply(transform_geometry)

)


tectono = tectono[
    tectono.geometry.intersects(
        ovda_box
    )
].copy()


tectono_results = []


for _, feature in tectono.iterrows():

    result = analyze_polygon(

        name=str(
            feature["Name"]
        ),

        geometry=feature.geometry,

        feature_type="Tectonomorphic"

    )


    if result is not None:

        tectono_results.append(
            result
        )



print("\n" + "=" * 70)

print(
    "INTRA-TESSERA PLAINS"
)

print("=" * 70)


tessera = gpd.read_file(

    GEOLOGY_FILE,

    layer="IntraTesseraPlains_map"

)


tessera["geometry"] = (

    tessera["geometry"]
    .apply(transform_geometry)

)


tessera_results = []


for index, feature in tessera.iterrows():

    result = analyze_polygon(

        name=f"Intra-tessera plains #{index + 1}",

        geometry=feature.geometry,

        feature_type="Intra-tessera plains"

    )


    if result is not None:

        tessera_results.append(
            result
        )


print("\n" + "=" * 70)

print(
    "ALL INTRA-TESSERA PLAINS COMBINED"
)

print("=" * 70)


combined_tessera = (

    tessera.geometry
    .union_all()

)


combined_tessera_result = analyze_polygon(

    name="All intra-tessera plains combined",

    geometry=combined_tessera,

    feature_type="Combined feature"

)



print("\n" + "=" * 70)

print(
    "BRUSHED UNIT"
)

print("=" * 70)


brushed = gpd.read_file(

    GEOLOGY_FILE,

    layer="BrushedUnit_map"

)


brushed["geometry"] = (

    brushed["geometry"]
    .apply(transform_geometry)

)


brushed_results = []


for _, feature in brushed.iterrows():

    feature_id = feature["id"]


    result = analyze_polygon(

        name=f"Brushed unit ID {feature_id}",

        geometry=feature.geometry,

        feature_type="Brushed unit"

    )


    if result is not None:

        result[
            "source_id"
        ] = feature_id


        brushed_results.append(
            result
        )


all_results = (

    tectono_results

    +

    tessera_results

    +

    brushed_results

)


results_df = pd.DataFrame(
    all_results
)



results_df.to_csv(

    "geology_reverse_validation.csv",

    index=False

)


if combined_tessera_result is not None:

    pd.DataFrame(
        [combined_tessera_result]
    ).to_csv(

        "intra_tessera_combined_validation.csv",

        index=False

    )



print("\n" + "=" * 70)

print(
    "STRONGEST GEOLOGICAL COVERAGE"
)

print("=" * 70)


if len(results_df) > 0:

    for cluster in range(4):

        column = (
            f"cluster_{cluster}_coverage_percent"
        )


        best_index = (
            results_df[column]
            .idxmax()
        )


        best_feature = results_df.loc[
            best_index,
            "geological_feature"
        ]


        best_value = results_df.loc[
            best_index,
            column
        ]


        print(

            f"\nCluster {cluster}:"

        )

        print(

            f"Highest mapped-feature coverage: "
            f"{best_feature}"

        )

        print(

            f"Coverage: "
            f"{best_value:.4f}%"

        )


print("\n" + "=" * 70)

print(
    "REVERSE GEOLOGICAL VALIDATION COMPLETE"
)

print("=" * 70)

print(
    "\nSaved:"
)

print(
    "geology_reverse_validation.csv"
)

print(
    "intra_tessera_combined_validation.csv"
)
