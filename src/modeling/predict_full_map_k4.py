import numpy as np
import pandas as pd
import rasterio

from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window



RADAR_FILE = str(RADAR_FILE)

HEIGHT_FILE = str(HEIGHT_FILE)

EMISSIVITY_FILE = str(EMISSIVITY_FILE)

SLOPE_FILE = str(SLOPE_FILE)

REFLECTIVITY_FILE = str(REFLECTIVITY_FILE)

CENTROIDS_FILE = str(K4_CENTROIDS_CSV)

SCALER_FILE = str(K4_SCALER_CSV)

OUTPUT_FILE = str(K4_RASTER)



BLOCK_SIZE = 512

OUTPUT_NODATA = 255



print("=" * 60)
print("LOADING K=4 MODEL")
print("=" * 60)

centroid_df = pd.read_csv(
    CENTROIDS_FILE,
    index_col=0
)


features = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity"
]


centroids = centroid_df[
    features
].to_numpy(
    dtype=np.float64
)


print("\nK=4 centroids:")
print(centroid_df)



scaler_df = pd.read_csv(
    SCALER_FILE
)


means = scaler_df[
    "mean"
].to_numpy(
    dtype=np.float64
)

scales = scaler_df[
    "scale"
].to_numpy(
    dtype=np.float64
)



centroids_scaled = (
    centroids - means
) / scales


print("\nScaler parameters loaded.")


print("\n" + "=" * 60)
print("OPENING RADAR REFERENCE")
print("=" * 60)

radar_src = rasterio.open(
    RADAR_FILE
)


radar_crs = radar_src.crs

radar_transform = radar_src.transform

radar_width = radar_src.width

radar_height = radar_src.height


print("Width:", radar_width)

print("Height:", radar_height)

print("CRS:", radar_crs)

print("Resolution:", radar_src.res)

print("NoData:", radar_src.nodata)




print("\n" + "=" * 60)
print("CREATING ALIGNED RASTER VIEWS")
print("=" * 60)


height_src = rasterio.open(
    HEIGHT_FILE
)

emissivity_src = rasterio.open(
    EMISSIVITY_FILE
)

slope_src = rasterio.open(
    SLOPE_FILE
)

reflectivity_src = rasterio.open(
    REFLECTIVITY_FILE
)


height_vrt = WarpedVRT(

    height_src,

    crs=radar_crs,

    transform=radar_transform,

    width=radar_width,

    height=radar_height,

    resampling=Resampling.bilinear,

    src_nodata=height_src.nodata,

    nodata=-32768
)


emissivity_vrt = WarpedVRT(

    emissivity_src,

    crs=radar_crs,

    transform=radar_transform,

    width=radar_width,

    height=radar_height,

    resampling=Resampling.bilinear,

    src_nodata=emissivity_src.nodata,

    nodata=-32768
)


slope_vrt = WarpedVRT(

    slope_src,

    crs=radar_crs,

    transform=radar_transform,

    width=radar_width,

    height=radar_height,

    resampling=Resampling.bilinear,

    src_nodata=slope_src.nodata,

    nodata=-32768
)


reflectivity_vrt = WarpedVRT(

    reflectivity_src,

    crs=radar_crs,

    transform=radar_transform,

    width=radar_width,

    height=radar_height,

    resampling=Resampling.bilinear,

    src_nodata=reflectivity_src.nodata,

    nodata=-32768
)




output_profile = radar_src.profile.copy()


output_profile.update(

    dtype="uint8",

    count=1,

    nodata=OUTPUT_NODATA,

    compress="lzw",

    predictor=2

)


output = rasterio.open(

    OUTPUT_FILE,

    "w",

    **output_profile
)



print("\n" + "=" * 60)
print("PROCESSING FULL VENUS RASTER â€” K=4")
print("=" * 60)

print(
    "Raster size:",
    radar_height,
    "x",
    radar_width
)

print(
    "Block size:",
    BLOCK_SIZE,
    "x",
    BLOCK_SIZE
)


total_pixels = (
    radar_height
    *
    radar_width
)

processed_pixels = 0

last_reported_percent = 0



for row_start in range(
    0,
    radar_height,
    BLOCK_SIZE
):

    for col_start in range(
        0,
        radar_width,
        BLOCK_SIZE
    ):

        block_height = min(
            BLOCK_SIZE,
            radar_height - row_start
        )

        block_width = min(
            BLOCK_SIZE,
            radar_width - col_start
        )


        window = Window(
            col_start,
            row_start,
            block_width,
            block_height
        )



        radar_block = radar_src.read(
            1,
            window=window
        ).astype(np.float64)


        

        height_block = height_vrt.read(
            1,
            window=window
        ).astype(np.float64)


        emissivity_block = emissivity_vrt.read(
            1,
            window=window
        ).astype(np.float64)


        slope_block = slope_vrt.read(
            1,
            window=window
        ).astype(np.float64)


        reflectivity_block = reflectivity_vrt.read(
            1,
            window=window
        ).astype(np.float64)



        valid = (

            (radar_block != 0)

            &

            (height_block != -32768)

            &

            (emissivity_block != -32768)

            &

            (slope_block != -32768)

            &

            (reflectivity_block != -32768)

        )



        cluster_block = np.full(

            (
                block_height,
                block_width
            ),

            OUTPUT_NODATA,

            dtype=np.uint8

        )


   

        if np.any(valid):

            X = np.column_stack((

                radar_block[valid],

                height_block[valid],

                emissivity_block[valid],

                slope_block[valid],

                reflectivity_block[valid]

            ))



            X_scaled = (
                X - means
            ) / scales



            distances = np.sum(

                (

                    X_scaled[:, np.newaxis, :]

                    -

                    centroids_scaled[np.newaxis, :, :]

                ) ** 2,

                axis=2

            )


            predictions = np.argmin(
                distances,
                axis=1
            )



            cluster_block[valid] = (
                predictions.astype(np.uint8)
            )


        output.write(
            cluster_block,
            1,
            window=window
        )


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


        current_report = int(
            progress // 10
        ) * 10


        if (
            current_report > last_reported_percent
            and current_report >= 10
        ):

            print(
                f"{current_report}% complete"
            )

            last_reported_percent = (
                current_report
            )




height_vrt.close()

emissivity_vrt.close()

slope_vrt.close()

reflectivity_vrt.close()

height_src.close()

emissivity_src.close()

slope_src.close()

reflectivity_src.close()

radar_src.close()




print("\n" + "=" * 60)

print("K=4 FULL VENUS CLASSIFICATION COMPLETE")

print("=" * 60)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Raster size:",
    radar_height,
    "x",
    radar_width
)

print(
    "Output NoData:",
    OUTPUT_NODATA
)
