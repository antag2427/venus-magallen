import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CROSS-REGION GEOLOGICAL VALIDATION
# Venus K-Means K=4
#
# Purpose:
#   Combine the independent geological validation results
#   from Ovda, Guinevere, and Lada.
#
# Important:
#   These are geological validation regions, NOT training labels.
# ============================================================


# ------------------------------------------------------------
# 1. KNOWN VALIDATION RESULTS
# ------------------------------------------------------------
#
# Values below come from our completed regional analyses.
# Percentages represent the proportion of each region/unit
# assigned to each K-Means cluster.
#
# We deliberately describe clusters as SURFACE-PROPERTY
# REGIMES, not geological units.
# ------------------------------------------------------------


records = [

    # ========================================================
    # OVDA REGIO
    # ========================================================

    {
        "region": "Ovda",
        "geology": "Regional baseline",
        "category": "All mapped/valid surface",
        "C0_pct": 55.09,
        "C1_pct": 20.56,
        "C2_pct": 1.57,
        "C3_pct": 22.77,
        "sample_pixels": np.nan
    },

    {
        "region": "Ovda",
        "geology": "Tessera / tectonomorphic units",
        "category": "Tectonic_highland",
        "C0_pct": np.nan,
        "C1_pct": np.nan,
        "C2_pct": np.nan,
        "C3_pct": 90.0,
        "sample_pixels": np.nan
    },

    {
        "region": "Ovda",
        "geology": "Intra-tessera plains",
        "category": "Intra_tessera_plains",
        "C0_pct": np.nan,
        "C1_pct": np.nan,
        "C2_pct": 3.03,
        "C3_pct": 82.81,
        "sample_pixels": np.nan
    },

    {
        "region": "Ovda",
        "geology": "Brushed unit ID 2",
        "category": "Brushed_unit",
        "C0_pct": np.nan,
        "C1_pct": np.nan,
        "C2_pct": np.nan,
        "C3_pct": 100.0,
        "sample_pixels": np.nan
    },


    # ========================================================
    # GUINEVERE V-30
    # ========================================================

    {
        "region": "Guinevere",
        "geology": "Tessera",
        "category": "Tessera",
        "C0_pct": 61.34,
        "C1_pct": 38.22,
        "C2_pct": 0.443,
        "C3_pct": 0.0,
        "sample_pixels": np.nan
    },

    {
        "region": "Guinevere",
        "geology": "Plains",
        "category": "Plains",
        "C0_pct": 29.21,
        "C1_pct": 70.54,
        "C2_pct": 0.246,
        "C3_pct": 0.0,
        "sample_pixels": np.nan
    },

    {
        "region": "Guinevere",
        "geology": "Tessera / upland",
        "category": "Tessera_Upland",
        "C0_pct": 49.53,
        "C1_pct": 50.22,
        "C2_pct": 0.25,
        "C3_pct": 0.0,
        "sample_pixels": np.nan
    },

    {
        "region": "Guinevere",
        "geology": "Volcanic flow",
        "category": "Volcanic_Flow",
        "C0_pct": 47.20,
        "C1_pct": 52.74,
        "C2_pct": 0.061,
        "C3_pct": 0.005,
        "sample_pixels": np.nan
    },


    # ========================================================
    # LADA V-56
    # ========================================================

    {
        "region": "Lada",
        "geology": "Regional baseline",
        "category": "All valid surface",
        "C0_pct": 25.54,
        "C1_pct": 34.74,
        "C2_pct": 39.72,
        "C3_pct": 0.00,
        "sample_pixels": 2085131
    },

    {
        "region": "Lada",
        "geology": "Tessera",
        "category": "Tessera",
        "C0_pct": 62.924,
        "C1_pct": 23.909,
        "C2_pct": 13.167,
        "C3_pct": 0.000,
        "sample_pixels": 310140
    },

    {
        "region": "Lada",
        "geology": "Tessera-like terrain",
        "category": "Tessera_like",
        "C0_pct": 51.544,
        "C1_pct": 26.628,
        "C2_pct": 21.827,
        "C3_pct": 0.000,
        "sample_pixels": 10102
    },

    {
        "region": "Lada",
        "geology": "Densely lineated terrain",
        "category": "Densely_lineated_terrain",
        "C0_pct": 9.106,
        "C1_pct": 19.126,
        "C2_pct": 71.768,
        "C3_pct": 0.000,
        "sample_pixels": 51527
    },

    {
        "region": "Lada",
        "geology": "Intra-tessera basin",
        "category": "Intra_tessera_basin",
        "C0_pct": 17.394,
        "C1_pct": 49.123,
        "C2_pct": 33.483,
        "C3_pct": 0.000,
        "sample_pixels": 139840
    },

    {
        "region": "Lada",
        "geology": "Plains",
        "category": "Plains",
        "C0_pct": 18.789,
        "C1_pct": 36.072,
        "C2_pct": 45.139,
        "C3_pct": 0.000,
        "sample_pixels": 1510832
    },

    {
        "region": "Lada",
        "geology": "Shield plains corona",
        "category": "Shield_plains_corona",
        "C0_pct": 32.814,
        "C1_pct": 36.527,
        "C2_pct": 30.659,
        "C3_pct": 0.000,
        "sample_pixels": 55184
    },
]


df = pd.DataFrame(records)


# ============================================================
# 2. SAVE MASTER VALIDATION TABLE
# ============================================================

df.to_csv(
    "cross_region_validation_master.csv",
    index=False
)

print("\n" + "=" * 80)
print("CROSS-REGION VALIDATION MASTER TABLE")
print("=" * 80)

print(df.to_string(index=False))


# ============================================================
# 3. DOMINANT CLUSTER
# ============================================================

cluster_columns = [
    "C0_pct",
    "C1_pct",
    "C2_pct",
    "C3_pct"
]

df["dominant_cluster"] = (
    df[cluster_columns]
    .idxmax(axis=1)
    .str.replace("_pct", "", regex=False)
)

df["dominant_purity"] = df[cluster_columns].max(axis=1)


# ============================================================
# 4. STRONGEST ASSOCIATIONS FOR EACH CLUSTER
# ============================================================

print("\n" + "=" * 80)
print("STRONGEST GEOLOGICAL ASSOCIATIONS")
print("=" * 80)

for cluster in ["C0", "C1", "C2", "C3"]:

    column = f"{cluster}_pct"

    subset = (
        df[
            ~df["category"].isin(
                ["All mapped/valid surface", "All valid surface"]
            )
        ]
        .sort_values(column, ascending=False)
    )

    print(f"\n{cluster}")
    print("-" * 40)

    print(
        subset[
            [
                "region",
                "geology",
                column,
                "dominant_cluster",
                "dominant_purity"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 5. REGION × CLUSTER SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("REGION-LEVEL SUMMARY")
print("=" * 80)

region_summary = []

for region, group in df.groupby("region"):

    non_baseline = group[
        ~group["category"].isin(
            ["All mapped/valid surface", "All valid surface"]
        )
    ]

    row = {
        "region": region,
        "mean_C0": non_baseline["C0_pct"].mean(),
        "mean_C1": non_baseline["C1_pct"].mean(),
        "mean_C2": non_baseline["C2_pct"].mean(),
        "mean_C3": non_baseline["C3_pct"].mean(),
    }

    region_summary.append(row)

region_summary = pd.DataFrame(region_summary)

print(region_summary.to_string(index=False))

region_summary.to_csv(
    "cross_region_region_summary.csv",
    index=False
)


# ============================================================
# 6. GEOLOGICAL CATEGORY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("GEOLOGICAL CATEGORY SUMMARY")
print("=" * 80)

category_summary = (
    df[
        ~df["category"].isin(
            [
                "All mapped/valid surface",
                "All valid surface"
            ]
        )
    ]
    .groupby("category", as_index=False)[
        ["C0_pct", "C1_pct", "C2_pct", "C3_pct"]
    ]
    .mean()
)

print(
    category_summary.to_string(index=False)
)

category_summary.to_csv(
    "cross_region_geological_category_summary.csv",
    index=False
)


# ============================================================
# 7. C2 ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("C2 CROSS-REGION ANALYSIS")
print("=" * 80)

c2 = (
    df[
        ~df["category"].isin(
            [
                "All mapped/valid surface",
                "All valid surface"
            ]
        )
    ]
    .sort_values("C2_pct", ascending=False)
)

print(
    c2[
        [
            "region",
            "geology",
            "C2_pct"
        ]
    ]
    .to_string(index=False)
)

print(
    "\nInterpretation:"
    "\nC2 is repeatedly enriched in some plains and densely lineated"
    "\nterrain, while mapped Tessera itself is not consistently C2-dominant."
)


# ============================================================
# 8. C0 ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("C0 CROSS-REGION ANALYSIS")
print("=" * 80)

c0 = (
    df[
        ~df["category"].isin(
            [
                "All mapped/valid surface",
                "All valid surface"
            ]
        )
    ]
    .sort_values("C0_pct", ascending=False)
)

print(
    c0[
        [
            "region",
            "geology",
            "C0_pct"
        ]
    ]
    .to_string(index=False)
)

print(
    "\nInterpretation:"
    "\nC0 is strongly represented in mapped Tessera in both"
    "\nGuinevere and Lada, supporting a tessera/highland-associated"
    "\nsurface-property interpretation."
)


# ============================================================
# 9. C3 ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("C3 CROSS-REGION ANALYSIS")
print("=" * 80)

c3 = (
    df[
        ~df["category"].isin(
            [
                "All mapped/valid surface",
                "All valid surface"
            ]
        )
    ]
    .sort_values("C3_pct", ascending=False)
)

print(
    c3[
        [
            "region",
            "geology",
            "C3_pct"
        ]
    ]
    .to_string(index=False)
)

print(
    "\nInterpretation:"
    "\nC3 is strongly expressed in specific Ovda geological contexts"
    "\nbut absent from the tested Lada and largely absent from"
    "\nGuinevere. This suggests a geographically restricted"
    "\nsurface-property regime rather than a universal geological class."
)


# ============================================================
# 10. FINAL WORKING INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("CURRENT SCIENTIFIC INTERPRETATION")
print("=" * 80)

print("""
C0:
A broad surface-property regime strongly associated with
mapped tessera/highland-related terrain.

C1:
A widespread background surface-property regime that dominates
large areas and several plains-rich environments.

C2:
A high-reflectivity surface-property regime associated with
specific plains and densely lineated terrain, with strong
regional and geographic dependence. It is NOT a generic
tessera detector.

C3:
A rare, spatially coherent and multivariate-distinctive
surface-property regime strongly expressed in Ovda but absent
from the tested Lada region and mostly absent from Guinevere.
This suggests geographic restriction rather than a universal
geological unit.
""")


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)

print("cross_region_validation_master.csv")
print("cross_region_region_summary.csv")
print("cross_region_geological_category_summary.csv")

print("\nCross-region validation completed.")