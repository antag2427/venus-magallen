import numpy as np
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER
from scipy.ndimage import label




CLUSTER_FILE = str(K4_RASTER)

NODATA = 255



print("=" * 60)
print("K=4 SPATIAL CLUSTER ANALYSIS")
print("=" * 60)


with rasterio.open(CLUSTER_FILE) as src:

    cluster_map = src.read(1)

    height = src.height
    width = src.width


print(
    "Raster size:",
    height,
    width
)




clusters = [
    0,
    1,
    2,
    3
]



structure = np.ones(
    (3, 3),
    dtype=np.uint8
)



for cluster in clusters:

    print("\n" + "=" * 60)

    print(
        f"CLUSTER {cluster}"
    )

    print("=" * 60)


 

    mask = (
        cluster_map == cluster
    )


    pixel_count = int(
        mask.sum()
    )


    print(
        "Total pixels:",
        f"{pixel_count:,}"
    )


    if pixel_count == 0:

        print(
            "No pixels found."
        )

        continue


    
    labeled_array, number_of_regions = label(
        mask,
        structure=structure
    )


    print(
        "Number of connected regions:",
        f"{number_of_regions:,}"
    )



    region_sizes = np.bincount(
        labeled_array.ravel()
    )[1:]


    largest_region = int(
        region_sizes.max()
    )

    smallest_region = int(
        region_sizes.min()
    )

    mean_region = (
        region_sizes.mean() 
    )

    median_region = np.median(
        region_sizes
    )



    largest_fraction = (
        largest_region
        /
        pixel_count
        *
        100
    )


    print(
        "Largest connected region:",
        f"{largest_region:,}",
        "pixels"
    )


    print(
        "Smallest connected region:",
        f"{smallest_region:,}",
        "pixels"
    )


    print(
        "Mean region size:",
        f"{mean_region:.2f}",
        "pixels"
    )


    print(
        "Median region size:",
        f"{median_region:.2f}",
        "pixels"
    )


    print(
        "Percentage of cluster in largest region:",
        f"{largest_fraction:.2f}%"
    )



print("\n" + "=" * 60)

print("K=4 SPATIAL ANALYSIS COMPLETE")

print("=" * 60)
