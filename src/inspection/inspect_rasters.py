import rasterio

from src.paths import (
    RADAR_FILE,
    HEIGHT_FILE,
    EMISSIVITY_FILE,
    SLOPE_FILE,
    REFLECTIVITY_FILE,
)


files = [
    RADAR_FILE,
    HEIGHT_FILE,
    EMISSIVITY_FILE,
    SLOPE_FILE,
    REFLECTIVITY_FILE,
]


for filepath in files:

    print("\n" + "=" * 60)
    print(filepath.name)
    print("=" * 60)

    with rasterio.open(filepath) as src:

        data = src.read(1)

        print("Shape:")
        print(src.height, src.width)

        print("\nCRS:")
        print(src.crs)

        print("\nTransform:")
        print(src.transform)

        print("\nResolution:")
        print(src.res)

        print("\nDtype:")
        print(src.dtypes)

        print("\nRasterio NoData:")
        print(src.nodata)

        print("\nMinimum:")
        print(data.min())

        print("\nMaximum:")
        print(data.max())

        print("\nZero pixels:")
        print((data == 0).sum())

        print(
            "Zero percentage:",
            (data == 0).mean() * 100,
            "%",
        )
