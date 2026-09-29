import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import (  # noqa: E402
    RADAR_FILE,
    HEIGHT_FILE,
    EMISSIVITY_FILE,
    SLOPE_FILE,
    REFLECTIVITY_FILE,
    TRAINING_SAMPLE,
)

FEATURES = ["radar", "height", "emissivity", "slope", "reflectivity"]
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
OTHER_NODATA = -32768


def _check_inputs() -> None:
    files = [RADAR_FILE, HEIGHT_FILE, EMISSIVITY_FILE, SLOPE_FILE, REFLECTIVITY_FILE]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raster inputs:\n" + "\n".join(str(path) for path in missing)
        )


def main() -> None:
    _check_inputs()
    TRAINING_SAMPLE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VENUS TRAINING-SAMPLE PREPARATION")
    print("=" * 80)
    print("Reference grid: radar")
    print(f"Requested radar-valid sample: {SAMPLE_SIZE:,}")
    print(f"Random seed: {RANDOM_SEED}")

    with rasterio.open(RADAR_FILE) as radar_src, \
         rasterio.open(HEIGHT_FILE) as height_src, \
         rasterio.open(EMISSIVITY_FILE) as emissivity_src, \
         rasterio.open(SLOPE_FILE) as slope_src, \
         rasterio.open(REFLECTIVITY_FILE) as reflectivity_src:

        width = radar_src.width
        height = radar_src.height
        total_pixels = width * height

        # The established workflow samples from radar-valid pixels first,
        # then removes samples whose aligned companion rasters are NoData.
        radar = radar_src.read(1)
        radar_valid = radar != 0
        valid_flat = np.flatnonzero(radar_valid.ravel())

        if len(valid_flat) == 0:
            raise ValueError("Radar contains no valid pixels (radar != 0).")

        sample_n = min(SAMPLE_SIZE, len(valid_flat))
        rng = np.random.default_rng(RANDOM_SEED)
        selected_flat = rng.choice(valid_flat, size=sample_n, replace=False)
        selected_flat.sort()

        rows = selected_flat // width
        cols = selected_flat % width

        # Read aligned companion rasters in manageable blocks. We only keep
        # the selected rows/columns, so memory use stays bounded.
        values = np.empty((sample_n, 5), dtype=np.float64)
        values[:, 0] = radar.ravel()[selected_flat]

        vrts = [
            WarpedVRT(
                src,
                crs=radar_src.crs,
                transform=radar_src.transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                nodata=OTHER_NODATA,
            )
            for src in (height_src, emissivity_src, slope_src, reflectivity_src)
        ]

        try:
            # Group points by 512-row blocks to avoid scattered full-raster reads.
            block_size = 512
            for row_start in range(0, height, block_size):
                mask = (rows >= row_start) & (rows < min(row_start + block_size, height))
                if not np.any(mask):
                    continue
                point_rows = rows[mask]
                point_cols = cols[mask]
                order = np.flatnonzero(mask)
                for feature_idx, vrt in enumerate(vrts, start=1):
                    row_min = int(point_rows.min())
                    row_max = int(point_rows.max()) + 1
                    col_min = int(point_cols.min())
                    col_max = int(point_cols.max()) + 1
                    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
                    block = vrt.read(1, window=window).astype(np.float64)
                    values[order, feature_idx] = block[
                        point_rows - row_min,
                        point_cols - col_min,
                    ]
        finally:
            for vrt in vrts:
                vrt.close()

        valid = np.all(np.isfinite(values), axis=1)
        valid &= np.all(values[:, 1:] != OTHER_NODATA, axis=1)
        values = values[valid]
        sample_rows = rows[valid]
        sample_cols = cols[valid]

    df = pd.DataFrame(values, columns=FEATURES)
    df["row"] = sample_rows.astype(np.int64)
    df["col"] = sample_cols.astype(np.int64)
    df.to_csv(TRAINING_SAMPLE, index=False)

    print(f"Raster size: {height:,} x {width:,} ({total_pixels:,} pixels)")
    print(f"Radar-valid pixels available: {len(valid_flat):,}")
    print(f"Sampled: {sample_n:,}")
    print(f"Final valid training samples: {len(df):,}")
    print(f"Saved: {TRAINING_SAMPLE}")


if __name__ == "__main__":
    main()
