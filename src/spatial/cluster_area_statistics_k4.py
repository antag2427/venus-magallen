import numpy as np
import rasterio
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import K4_RASTER


CLUSTER_FILE = str(K4_RASTER)

NODATA = 255


with rasterio.open(CLUSTER_FILE) as src:

    data = src.read(1)

    total_pixels = data.size

    valid = data != NODATA

    valid_data = data[valid]

    valid_pixels = len(valid_data)


print("=" * 60)
print("K=4 FULL VENUS CLUSTER STATISTICS")
print("=" * 60)

print(
    "Total raster pixels:",
    f"{total_pixels:,}"
)

print(
    "Valid pixels:",
    f"{valid_pixels:,}"
)

print(
    "NoData pixels:",
    f"{total_pixels - valid_pixels:,}"
)

print(
    "Valid percentage:",
    f"{valid_pixels / total_pixels * 100:.2f}%"
)


print("\nCluster distribution:")
print("-" * 40)


clusters = np.unique(
    valid_data
)


for cluster in clusters:

    count = np.sum(
        valid_data == cluster
    )

    percentage = (
        count
        /
        valid_pixels
        *
        100
    )

    print(
        f"Cluster {cluster}: "
        f"{count:,} pixels "
        f"({percentage:.2f}%)"
    )
