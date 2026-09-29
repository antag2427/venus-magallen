import numpy as np
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER


# ============================================================
# INPUT
# ============================================================

CLUSTER_FILE = str(K4_RASTER)

VENUS_RADIUS = 6_051_000.0


# ============================================================
# OVDA REGIO BOUNDARY
# ============================================================

MIN_LON = 90.0
MAX_LON = 120.0

MIN_LAT = -25.0
MAX_LAT = 0.0


# ============================================================
# DIVIDING LINE
#
# This is a simple first-order division for the experiment.
#
# It is NOT being claimed as the actual geological boundary.
#
# We are simply asking whether the northern and southern
# portions have different cluster compositions.
# ============================================================

DIVIDE_LAT = -12.5


# ============================================================
# CONVERT LATITUDE/LONGITUDE TO PROJECTED COORDINATES
# ============================================================

min_x = (
    MIN_LON
    * np.pi
    / 180.0
    * VENUS_RADIUS
)

max_x = (
    MAX_LON
    * np.pi
    / 180.0
    * VENUS_RADIUS
)

min_y = (
    MIN_LAT
    * np.pi
    / 180.0
    * VENUS_RADIUS
)

max_y = (
    MAX_LAT
    * np.pi
    / 180.0
    * VENUS_RADIUS
)


divide_y = (
    DIVIDE_LAT
    * np.pi
    / 180.0
    * VENUS_RADIUS
)


# ============================================================
# 1. OPEN RASTER
# ============================================================

print("=" * 60)
print("OVDA REGIO CLUSTER STATISTICS")
print("=" * 60)


with rasterio.open(CLUSTER_FILE) as src:

    # --------------------------------------------------------
    # Convert geographic boundaries to pixel coordinates
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
    # Convert dividing latitude to raster row
    # --------------------------------------------------------

    divide_row, _ = src.index(
        0,
        divide_y
    )


    # --------------------------------------------------------
    # Make sure boundaries are valid
    # --------------------------------------------------------

    row_top = max(
        0,
        row_top
    )

    row_bottom = min(
        src.height,
        row_bottom
    )

    col_left = max(
        0,
        col_left
    )

    col_right = min(
        src.width,
        col_right
    )


    divide_row = max(
        row_top,
        min(
            row_bottom,
            divide_row
        )
    )


    print("\nPixel boundaries:")

    print(
        "Top row:",
        row_top
    )

    print(
        "Divide row:",
        divide_row
    )

    print(
        "Bottom row:",
        row_bottom
    )

    print(
        "Left column:",
        col_left
    )

    print(
        "Right column:",
        col_right
    )


    # ========================================================
    # 2. READ NORTHERN PART
    # ========================================================

    northern_window = rasterio.windows.Window(

        col_left,

        row_top,

        col_right - col_left,

        divide_row - row_top

    )


    northern = src.read(
        1,
        window=northern_window
    )


    # ========================================================
    # 3. READ SOUTHERN PART
    # ========================================================

    southern_window = rasterio.windows.Window(

        col_left,

        divide_row,

        col_right - col_left,

        row_bottom - divide_row

    )


    southern = src.read(
        1,
        window=southern_window
    )


# ============================================================
# 4. FUNCTION TO CALCULATE CLUSTER DISTRIBUTION
# ============================================================

def calculate_distribution(data):

    valid = data[
        data != 255
    ]

    total = len(valid)

    print(
        "\nValid pixels:",
        f"{total:,}"
    )


    for cluster in range(4):

        count = np.sum(
            valid == cluster
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
# 5. NORTHERN REGION
# ============================================================

print("\n" + "=" * 60)

print(
    "NORTHERN OVDA REGION"
)

print(
    "0Â° to -12.5Â° latitude"
)

print("=" * 60)


calculate_distribution(
    northern
)


# ============================================================
# 6. SOUTHERN REGION
# ============================================================

print("\n" + "=" * 60)

print(
    "SOUTHERN OVDA REGION"
)

print(
    "-12.5Â° to -25Â° latitude"
)

print("=" * 60)


calculate_distribution(
    southern
)


# ============================================================
# 7. COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "OVDA CLUSTER ANALYSIS COMPLETE"
)

print("=" * 60)
