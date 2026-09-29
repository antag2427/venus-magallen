import numpy as np
import matplotlib.pyplot as plt
import rasterio

from src.paths import K4_RASTER


CLUSTER_FILE = K4_RASTER

NODATA = 255


with rasterio.open(CLUSTER_FILE) as src:

    print("=" * 60)
    print("K=4 FULL VENUS CLUSTER MAP")
    print("=" * 60)

    print("Width:", src.width)
    print("Height:", src.height)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("NoData:", src.nodata)


    scale = 4

    out_height = src.height // scale

    out_width = src.width // scale

    cluster_map = src.read(
        1,
        out_shape=(
            out_height,
            out_width
        ),
        resampling=rasterio.enums.Resampling.nearest
    )



cluster_map = np.ma.masked_equal(
    cluster_map,
    NODATA
)



plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    cluster_map,
    cmap="tab10",
    interpolation="nearest"
)

plt.colorbar(
    label="Cluster ID"
)

plt.title(
    "K=4 Full Venus Terrain Clustering"
)

plt.xlabel(
    "Radar Grid X"
)

plt.ylabel(
    "Radar Grid Y"
)

plt.tight_layout()

plt.show()