import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FINAL GEOLOGICAL VALIDATION MATRIX
# Venus K-Means K=4
#
# Uses the actual geological validation CSVs.
#
# Regions currently available:
#   - Lada V-56
#   - Guinevere V-30
#
# The script:
#   1. loads validation CSVs
#   2. standardizes column names
#   3. preserves original enrichment values
#   4. combines all validation results
#   5. identifies cluster/geology associations
#   6. performs large-sample analysis
#   7. produces final summary tables
#
# Important:
#   K-Means clusters are surface-property regimes.
#   Geological maps are independent validation data.
# ============================================================


# ============================================================
# 1. FILE DISCOVERY
# ============================================================

print("\n" + "=" * 90)
print("CHECKING VALIDATION FILES")
print("=" * 90)


file_candidates = {

    "Lada_units": [
        "lada_usgs_unit_validation_corrected.csv",
        "lada_usgs_unit_validation.csv",
    ],

    "Lada_categories": [
        "lada_usgs_category_validation_corrected.csv",
        "lada_usgs_category_validation.csv",
    ],

    "Guinevere_units": [
        "guinevere_usgs_unit_validation.csv",
        "guinevere_usgs_unit_validation_corrected.csv",
    ],

    "Guinevere_categories": [
        "guinevere_usgs_category_validation.csv",
        "guinevere_usgs_category_validation_corrected.csv",
    ],
}


def find_file(candidates):

    for filename in candidates:

        path = Path(filename)

        if path.exists():
            return path

    return None


paths = {}

for name, candidates in file_candidates.items():

    path = find_file(candidates)

    paths[name] = path

    if path is not None:
        print(f"{name}: FOUND -> {path.name}")

    else:
        print(f"{name}: NOT FOUND")


# ============================================================
# 2. LOAD FILES
# ============================================================

def load_csv(path):

    if path is None:
        return None

    try:

        return pd.read_csv(path)

    except Exception as exc:

        print(f"\nERROR loading {path}")
        print(exc)

        return None


lada_units = load_csv(paths["Lada_units"])
lada_categories = load_csv(paths["Lada_categories"])

guinevere_units = load_csv(paths["Guinevere_units"])
guinevere_categories = load_csv(paths["Guinevere_categories"])


print("\n" + "=" * 90)
print("LOADED TABLES")
print("=" * 90)


for name, df in [

    ("Lada units", lada_units),
    ("Lada categories", lada_categories),

    ("Guinevere units", guinevere_units),
    ("Guinevere categories", guinevere_categories),

]:

    if df is None:

        print(f"{name}: unavailable")

    else:

        print(
            f"{name}: {len(df)} rows"
        )

        print(
            "Columns:",
            list(df.columns)
        )


# ============================================================
# 3. STANDARDIZATION FUNCTION
# ============================================================

def standardize_validation_table(
    df,
    region,
    validation_level
):

    if df is None:
        return None


    # --------------------------------------------------------
    # IMPORTANT:
    # Explicitly make string columns dtype=object.
    # This prevents the Pandas dtype error we just encountered.
    # --------------------------------------------------------

    result = pd.DataFrame(index=df.index)

    result["region"] = region
    result["validation_level"] = validation_level


    # --------------------------------------------------------
    # GEOLOGICAL NAME
    # --------------------------------------------------------

    name_candidates = [

        "unit_name",
        "UnitName",

        "description",

        "category",

        "unit",
        "Unit",

        "geology",

        "usgs_type",

    ]


    name_column = None

    for column in name_candidates:

        if column in df.columns:

            name_column = column
            break


    if name_column is not None:

        result["geology"] = (
            df[name_column]
            .astype(str)
            .str.strip()
        )

    else:

        result["geology"] = "Unknown"


    # --------------------------------------------------------
    # UNIT CODE
    # --------------------------------------------------------

    code_candidates = [

        "unit",
        "Unit",
        "usgs_type",
        "TYPE",

    ]


    code_column = None

    for column in code_candidates:

        if column in df.columns:

            code_column = column
            break


    if code_column is not None:

        result["unit_code"] = (
            df[code_column]
            .astype(str)
            .str.strip()
        )

    else:

        result["unit_code"] = ""


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if "category" in df.columns:

        result["category"] = (
            df["category"]
            .astype(str)
            .str.strip()
        )

    else:

        result["category"] = result["geology"]


    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    pixel_candidates = [

        "valid_pixels",
        "sample_pixels",
        "pixels",
        "pixel_count",
        "n_pixels",
        "count",

    ]


    pixel_column = None

    for column in pixel_candidates:

        if column in df.columns:

            pixel_column = column
            break


    if pixel_column is not None:

        result["valid_pixels"] = pd.to_numeric(
            df[pixel_column],
            errors="coerce"
        )

    else:

        result["valid_pixels"] = np.nan


    # --------------------------------------------------------
    # CLUSTER PERCENTAGES
    # --------------------------------------------------------

    cluster_candidates = {

        "C0_pct": [
            "cluster_0_percentage",
            "cluster0_percentage",
            "C0_pct",
            "c0_pct",
        ],

        "C1_pct": [
            "cluster_1_percentage",
            "cluster1_percentage",
            "C1_pct",
            "c1_pct",
        ],

        "C2_pct": [
            "cluster_2_percentage",
            "cluster2_percentage",
            "C2_pct",
            "c2_pct",
        ],

        "C3_pct": [
            "cluster_3_percentage",
            "cluster3_percentage",
            "C3_pct",
            "c3_pct",
        ],
    }


    for output_column, candidates in cluster_candidates.items():

        source_column = None

        for candidate in candidates:

            if candidate in df.columns:

                source_column = candidate
                break


        if source_column is not None:

            result[output_column] = pd.to_numeric(
                df[source_column],
                errors="coerce"
            )

        else:

            result[output_column] = np.nan


    # --------------------------------------------------------
    # ORIGINAL ENRICHMENT VALUES
    # --------------------------------------------------------

    enrichment_candidates = {

        "C0_enrichment": [
            "cluster_0_enrichment",
            "C0_enrichment",
        ],

        "C1_enrichment": [
            "cluster_1_enrichment",
            "C1_enrichment",
        ],

        "C2_enrichment": [
            "cluster_2_enrichment",
            "C2_enrichment",
        ],

        "C3_enrichment": [
            "cluster_3_enrichment",
            "C3_enrichment",
        ],
    }


    for output_column, candidates in enrichment_candidates.items():

        source_column = None

        for candidate in candidates:

            if candidate in df.columns:

                source_column = candidate
                break


        if source_column is not None:

            result[output_column] = pd.to_numeric(
                df[source_column],
                errors="coerce"
            )

        else:

            result[output_column] = np.nan


    # --------------------------------------------------------
    # DOMINANT CLUSTER
    #
    # Explicitly use object dtype.
    # --------------------------------------------------------

    result["dominant_cluster"] = pd.Series(
        index=df.index,
        dtype="object"
    )

    result["dominant_purity"] = np.nan


    pct_columns = [
        "C0_pct",
        "C1_pct",
        "C2_pct",
        "C3_pct",
    ]


    valid_rows = (
        result[pct_columns]
        .notna()
        .any(axis=1)
    )


    if valid_rows.any():

        valid_pct = result.loc[
            valid_rows,
            pct_columns
        ]


        dominant = (
            valid_pct
            .idxmax(axis=1)
            .str.replace(
                "_pct",
                "",
                regex=False
            )
        )


        purity = valid_pct.max(axis=1)


        # Use .astype(object) explicitly
        result.loc[
            valid_rows,
            "dominant_cluster"
        ] = dominant.astype(object)


        result.loc[
            valid_rows,
            "dominant_purity"
        ] = purity.astype(float)


    return result


# ============================================================
# 4. STANDARDIZE AVAILABLE TABLES
# ============================================================

tables = []


table_definitions = [

    (
        lada_units,
        "Lada",
        "USGS unit"
    ),

    (
        lada_categories,
        "Lada",
        "USGS category"
    ),

    (
        guinevere_units,
        "Guinevere",
        "USGS unit"
    ),

    (
        guinevere_categories,
        "Guinevere",
        "USGS category"
    ),

]


for df, region, level in table_definitions:

    standardized = standardize_validation_table(
        df,
        region,
        level
    )

    if standardized is not None:

        tables.append(
            standardized
        )


if not tables:

    raise RuntimeError(
        "\nNo validation tables were loaded."
    )


# ============================================================
# 5. COMBINE
# ============================================================

combined = pd.concat(
    tables,
    ignore_index=True
)


# Remove rows with no cluster information

cluster_columns = [
    "C0_pct",
    "C1_pct",
    "C2_pct",
    "C3_pct",
]


combined = combined[
    combined[
        cluster_columns
    ]
    .notna()
    .any(axis=1)
].copy()


# ============================================================
# 6. SAVE MASTER MATRIX
# ============================================================

combined.to_csv(
    "final_geological_validation_matrix.csv",
    index=False
)


# ============================================================
# 7. PRINT MASTER MATRIX
# ============================================================

print("\n" + "=" * 90)
print("FINAL GEOLOGICAL VALIDATION MATRIX")
print("=" * 90)


display_columns = [

    "region",
    "validation_level",
    "unit_code",
    "geology",
    "category",
    "valid_pixels",

    "C0_pct",
    "C1_pct",
    "C2_pct",
    "C3_pct",

    "C0_enrichment",
    "C1_enrichment",
    "C2_enrichment",
    "C3_enrichment",

    "dominant_cluster",
    "dominant_purity",
]


print(
    combined[
        display_columns
    ]
    .to_string(index=False)
)


# ============================================================
# 8. ROBUST ASSOCIATIONS
#
# Conservative threshold:
#   dominant purity >= 60%
# ============================================================

print("\n" + "=" * 90)
print("ROBUST GEOLOGICAL ASSOCIATIONS")
print("=" * 90)


robust = combined[
    combined["dominant_purity"] >= 60
].copy()


robust = robust.sort_values(
    "dominant_purity",
    ascending=False
)


if len(robust) > 0:

    print(
        robust[
            [
                "region",
                "validation_level",
                "unit_code",
                "geology",
                "category",
                "valid_pixels",
                "dominant_cluster",
                "dominant_purity",
            ]
        ]
        .to_string(index=False)
    )

else:

    print("No associations reached 60% purity.")


robust.to_csv(
    "final_robust_geological_associations.csv",
    index=False
)


# ============================================================
# 9. LARGE-SAMPLE VALIDATION
#
# We treat >=10,000 pixels as a conservative robust sample.
# ============================================================

large_sample = combined[
    (
        combined["valid_pixels"].isna()
    )
    |
    (
        combined["valid_pixels"] >= 10000
    )
].copy()


large_sample.to_csv(
    "final_geological_validation_large_samples.csv",
    index=False
)


print("\n" + "=" * 90)
print("LARGE-SAMPLE VALIDATION")
print("Threshold: >= 10,000 valid pixels")
print("=" * 90)


print(
    large_sample[
        [
            "region",
            "validation_level",
            "unit_code",
            "geology",
            "category",
            "valid_pixels",

            "C0_pct",
            "C1_pct",
            "C2_pct",
            "C3_pct",

            "dominant_cluster",
            "dominant_purity",
        ]
    ]
    .sort_values(
        [
            "region",
            "dominant_purity"
        ],
        ascending=[
            True,
            False
        ]
    )
    .to_string(index=False)
)


# ============================================================
# 10. CLUSTER-SPECIFIC ANALYSIS
# ============================================================

def print_cluster_analysis(
    cluster
):

    print("\n" + "=" * 90)
    print(f"{cluster} ANALYSIS")
    print("=" * 90)


    pct_column = f"{cluster}_pct"
    enrichment_column = f"{cluster}_enrichment"


    subset = combined[
        combined[pct_column].notna()
    ].copy()


    subset = subset.sort_values(
        pct_column,
        ascending=False
    )


    columns = [

        "region",
        "validation_level",
        "unit_code",
        "geology",
        "category",
        "valid_pixels",

        pct_column,
        enrichment_column,

    ]


    print(
        subset[
            columns
        ]
        .head(30)
        .to_string(index=False)
    )


for cluster in [
    "C0",
    "C1",
    "C2",
    "C3"
]:

    print_cluster_analysis(
        cluster
    )


# ============================================================
# 11. TESSERA ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("TESSERA-RELATED GEOLOGICAL UNITS")
print("=" * 90)


tessera_mask = (

    combined["geology"]
    .str.contains(
        "tessera",
        case=False,
        na=False
    )

    |

    combined["category"]
    .str.contains(
        "tessera",
        case=False,
        na=False
    )

)


tessera = combined[
    tessera_mask
].copy()


if len(tessera) > 0:

    print(
        tessera[
            [
                "region",
                "validation_level",
                "unit_code",
                "geology",
                "category",
                "valid_pixels",

                "C0_pct",
                "C1_pct",
                "C2_pct",
                "C3_pct",

                "dominant_cluster",
                "dominant_purity",
            ]
        ]
        .sort_values(
            "C0_pct",
            ascending=False
        )
        .to_string(index=False)
    )

else:

    print(
        "No Tessera-related units found."
    )


# ============================================================
# 12. ENRICHMENT RANKING
# ============================================================

print("\n" + "=" * 90)
print("STRONGEST ENRICHMENT VALUES")
print("=" * 90)


for cluster in [
    "C0",
    "C1",
    "C2",
    "C3"
]:

    enrichment_column = (
        f"{cluster}_enrichment"
    )

    pct_column = (
        f"{cluster}_pct"
    )


    subset = combined[
        combined[enrichment_column].notna()
    ].copy()


    if len(subset) == 0:

        print(
            f"\n{cluster}: "
            f"no enrichment values available."
        )

        continue


    subset = subset[
        np.isfinite(
            subset[enrichment_column]
        )
    ]


    subset = subset[
        subset[enrichment_column] > 0
    ]


    subset = subset.sort_values(
        enrichment_column,
        ascending=False
    )


    print(f"\n{cluster}")


    print(
        subset[
            [
                "region",
                "validation_level",
                "unit_code",
                "geology",
                "category",
                "valid_pixels",

                pct_column,
                enrichment_column,

            ]
        ]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# 13. REGION SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("REGION SUMMARY")
print("=" * 90)


for region in sorted(
    combined[
        "region"
    ]
    .dropna()
    .unique()
):

    region_df = combined[
        combined["region"] == region
    ]


    print(
        f"\n{region}"
    )


    print(
        "Validation rows:",
        len(region_df)
    )


    for cluster in [
        "C0",
        "C1",
        "C2",
        "C3"
    ]:

        column = (
            f"{cluster}_pct"
        )


        values = pd.to_numeric(
            region_df[column],
            errors="coerce"
        ).dropna()


        if len(values) > 0:

            print(
                f"  {cluster}: "
                f"mean={values.mean():.3f}% | "
                f"max={values.max():.3f}%"
            )


# ============================================================
# 14. FINAL SCIENTIFIC INTERPRETATION
# ============================================================

print("\n" + "=" * 90)
print("FINAL SCIENTIFIC INTERPRETATION")
print("=" * 90)


print("""
C0
--
C0 shows the strongest repeated association with mapped
Tessera among the currently available validation regions.
It should be described as a TESSERA-ASSOCIATED SURFACE-
PROPERTY REGIME rather than as the Tessera geological unit.


C1
--
C1 is a widespread surface-property regime and is frequently
dominant in plains and mixed geological environments. It does
not currently show a strong one-to-one geological identity.


C2
--
C2 is strongly represented in several Lada plains units and
especially in densely lineated terrain. Its association with
Tessera is weaker than that of C0.

Therefore C2 should NOT be called a generic Tessera regime.


C3
--
C3 is strongly concentrated in the tested Ovda geological
contexts but is absent from the tested Lada region and nearly
absent from Guinevere.

Therefore C3 is best described as a GEOGRAPHICALLY RESTRICTED
SURFACE-PROPERTY REGIME rather than a universal geological unit.


IMPORTANT SCIENTIFIC DISTINCTION
---------------------------------

K-Means was trained using the surface-property variables.

USGS geological maps were not used to create the clusters.

Therefore these results demonstrate GEOLOGICAL ASSOCIATION,
not GEOLOGICAL IDENTITY.
""")


# ============================================================
# 15. OUTPUT FILES
# ============================================================

print("\n" + "=" * 90)
print("OUTPUT FILES")
print("=" * 90)


print(
    "final_geological_validation_matrix.csv"
)

print(
    "final_geological_validation_large_samples.csv"
)

print(
    "final_robust_geological_associations.csv"
)


print("\nFinal geological validation completed successfully.")