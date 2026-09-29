import numpy as np
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER, K4_SMOOTHED_RASTER

from rasterio.windows import Window
from scipy.ndimage import convolve


# ============================================================
# FILES
# ============================================================

INPUT_FILE = str(K4_RASTER)

OUTPUT_FILE = str(K4_SMOOTHED_RASTER)


# ============================================================
# SETTINGS
# ============================================================

NODATA = 255

BLOCK_SIZE = 512

# Neighborhood:
#
# 3 x 3 majority filter
#
# This means each pixel looks at itself and its 8 neighbors.
#
KERNEL_SIZE = 3

# Number of clusters
N_CLUSTERS = 4


# ============================================================
# 1. OPEN INPUT RASTER
# ============================================================

print("=" * 70)
print("CHUNKED K=4 SPATIAL SMOOTHING")
print("=" * 70)


with rasterio.open(INPUT_FILE) as src:

    print("\nInput raster:")
    print("Width:", src.width)
    print("Height:", src.height)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("NoData:", src.nodata)

    width = src.width
    height = src.height

    profile = src.profile.copy()


    # ========================================================
    # 2. OUTPUT PROFILE
    # ========================================================

    profile.update(

        dtype="uint8",

        count=1,

        nodata=NODATA,

        compress="lzw",

        predictor=2

    )


    # ========================================================
    # 3. CREATE OUTPUT
    # ========================================================

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:


        # ====================================================
        # 4. PROCESS BLOCKS
        # ====================================================

        total_pixels = (
            width * height
        )

        processed_pixels = 0

        last_reported = 0


        for row_start in range(
            0,
            height,
            BLOCK_SIZE
        ):

            for col_start in range(
                0,
                width,
                BLOCK_SIZE
            ):


                # ------------------------------------------------
                # Actual output block dimensions
                # ------------------------------------------------

                block_height = min(
                    BLOCK_SIZE,
                    height - row_start
                )

                block_width = min(
                    BLOCK_SIZE,
                    width - col_start
                )


                # ------------------------------------------------
                # 1-pixel halo around block.
                #
                # We need this because a 3x3 neighborhood needs
                # one neighboring pixel around every edge.
                # ------------------------------------------------

                halo = 1


                read_row_start = max(
                    0,
                    row_start - halo
                )

                read_col_start = max(
                    0,
                    col_start - halo
                )

                read_row_end = min(
                    height,
                    row_start
                    + block_height
                    + halo
                )

                read_col_end = min(
                    width,
                    col_start
                    + block_width
                    + halo
                )


                read_height = (
                    read_row_end
                    -
                    read_row_start
                )

                read_width = (
                    read_col_end
                    -
                    read_col_start
                )


                read_window = Window(
                    read_col_start,
                    read_row_start,
                    read_width,
                    read_height
                )


                # ------------------------------------------------
                # READ BLOCK WITH HALO
                # ------------------------------------------------

                block = src.read(
                    1,
                    window=read_window
                )


                # ------------------------------------------------
                # VALID MASK
                # ------------------------------------------------

                valid = (
                    block != NODATA
                )


                # ------------------------------------------------
                # CREATE EMPTY OUTPUT
                # ------------------------------------------------

                smoothed = np.full(
                    block.shape,
                    NODATA,
                    dtype=np.uint8
                )


                # ------------------------------------------------
                # MAJORITY VOTE
                #
                # We calculate how many neighboring pixels belong
                # to each cluster.
                #
                # This avoids generic_filter and works efficiently
                # on each manageable block.
                # ------------------------------------------------

                counts = np.zeros(
                    (
                        N_CLUSTERS,
                        block.shape[0],
                        block.shape[1]
                    ),
                    dtype=np.uint8
                )


                for cluster in range(
                    N_CLUSTERS
                ):

                    cluster_mask = (
                        block == cluster
                    )


                    # Convert boolean â†’ uint8.
                    cluster_mask = (
                        cluster_mask
                        .astype(np.uint8)
                    )


                    # Count cluster occurrences in 3x3 window.
                    cluster_count = convolve(

                        cluster_mask,

                        np.ones(
                            (
                                KERNEL_SIZE,
                                KERNEL_SIZE
                            ),
                            dtype=np.uint8
                        ),

                        mode="constant",

                        cval=0

                    )


                    counts[
                        cluster
                    ] = cluster_count


                # ------------------------------------------------
                # FIND MAJORITY CLUSTER
                # ------------------------------------------------

                local_majority = np.argmax(
                    counts,
                    axis=0
                ).astype(
                    np.uint8
                )


                # ------------------------------------------------
                # HOW MANY VALID PIXELS?
                # ------------------------------------------------

                valid_count = convolve(

                    valid.astype(
                        np.uint8
                    ),

                    np.ones(
                        (
                            KERNEL_SIZE,
                            KERNEL_SIZE
                        ),
                        dtype=np.uint8
                    ),

                    mode="constant",

                    cval=0

                )


                # ------------------------------------------------
                # Only change pixels that have a real local
                # neighborhood.
                #
                # If the center pixel itself is NoData, keep
                # it NoData.
                # ------------------------------------------------

                smoothed[
                    valid
                ] = local_majority[
                    valid
                ]


                # ------------------------------------------------
                # EXTRACT THE ORIGINAL OUTPUT REGION FROM THE
                # HALO BLOCK
                # ------------------------------------------------

                output_row_offset = (
                    row_start
                    -
                    read_row_start
                )

                output_col_offset = (
                    col_start
                    -
                    read_col_start
                )


                output_block = smoothed[

                    output_row_offset:
                    output_row_offset
                    + block_height,

                    output_col_offset:
                    output_col_offset
                    + block_width

                ]


                # ------------------------------------------------
                # WRITE OUTPUT
                # ------------------------------------------------

                output_window = Window(

                    col_start,

                    row_start,

                    block_width,

                    block_height

                )


                dst.write(

                    output_block,

                    1,

                    window=output_window

                )


                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                processed_pixels += (

                    block_height
                    *
                    block_width

                )


                progress = (

                    processed_pixels
                    /
                    total_pixels
                    *
                    100

                )


                report = (
                    int(progress // 10)
                    * 10
                )


                if (
                    report >= 10
                    and report > last_reported
                ):

                    print(
                        f"{report}% complete"
                    )

                    last_reported = report


# ============================================================
# 5. COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "SPATIAL SMOOTHING COMPLETE"
)

print("=" * 70)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Filter:",
    "3 x 3 majority"
)

print(
    "Input was processed in",
    f"{BLOCK_SIZE} x {BLOCK_SIZE}",
    "tiles."
)
