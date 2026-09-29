from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from src.paths import (
    EMISSIVITY_FILE,
    HEIGHT_FILE,
    RADAR_FILE,
    REFLECTIVITY_FILE,
    SLOPE_FILE,
)

from .config import ALL_FEATURES, BASE_FEATURES


BASE_PATHS = {
    "radar": RADAR_FILE,
    "height": HEIGHT_FILE,
    "emissivity": EMISSIVITY_FILE,
    "slope": SLOPE_FILE,
    "reflectivity": REFLECTIVITY_FILE,
}


def _safe_nodata_mask(array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    return mask


def _fill_invalid_with_local_mean(data: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    kernel = 2 * radius + 1
    weight = valid.astype(np.float64)
    safe = np.where(valid, data, 0.0)
    numerator = ndimage.uniform_filter(safe, size=kernel, mode="nearest")
    denominator = ndimage.uniform_filter(weight, size=kernel, mode="nearest")
    local_mean = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    fallback = np.nanmedian(data[valid]) if np.any(valid) else 0.0
    filled = np.where(valid, data, local_mean)
    filled = np.where(np.isfinite(filled), filled, fallback)
    return filled


def _local_std(data: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    kernel = 2 * radius + 1
    weight = valid.astype(np.float64)
    safe = np.where(valid, data, 0.0).astype(np.float64)
    mean_num = ndimage.uniform_filter(safe, size=kernel, mode="nearest")
    mean_den = ndimage.uniform_filter(weight, size=kernel, mode="nearest")
    mean = np.divide(mean_num, mean_den, out=np.zeros_like(mean_num), where=mean_den > 0)
    sq_num = ndimage.uniform_filter(safe * safe, size=kernel, mode="nearest")
    second = np.divide(sq_num, mean_den, out=np.zeros_like(sq_num), where=mean_den > 0)
    variance = np.maximum(second - mean * mean, 0.0)
    return np.sqrt(variance)


def _local_range(data: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    high = np.where(valid, data, -np.inf)
    low = np.where(valid, data, np.inf)
    high = ndimage.maximum_filter(high, size=size, mode="nearest")
    low = ndimage.minimum_filter(low, size=size, mode="nearest")
    out = high - low
    out[~np.isfinite(out)] = 0.0
    return out


def derive_local_features(
    arrays: Dict[str, np.ndarray],
    valid_masks: Dict[str, np.ndarray],
    radius: int,
    pixel_size_x: float,
    pixel_size_y: float,
) -> Dict[str, np.ndarray]:
    """Compute local derivatives while respecting per-raster validity masks."""
    out: Dict[str, np.ndarray] = {}
    for name in BASE_FEATURES:
        data = np.asarray(arrays[name], dtype=np.float64)
        valid = np.asarray(valid_masks[name], dtype=bool)
        out[f"{name}_local_std"] = _local_std(data, valid, radius)

    for name in ("radar", "height", "emissivity", "reflectivity"):
        data = np.asarray(arrays[name], dtype=np.float64)
        valid = np.asarray(valid_masks[name], dtype=bool)
        out[f"{name}_local_range"] = _local_range(data, valid, radius)

    height = _fill_invalid_with_local_mean(
        np.asarray(arrays["height"], dtype=np.float64),
        np.asarray(valid_masks["height"], dtype=bool),
        radius,
    )
    radar = _fill_invalid_with_local_mean(
        np.asarray(arrays["radar"], dtype=np.float64),
        np.asarray(valid_masks["radar"], dtype=bool),
        radius,
    )

    gy_h, gx_h = np.gradient(height, pixel_size_y, pixel_size_x)
    gy_r, gx_r = np.gradient(radar, pixel_size_y, pixel_size_x)
    out["height_gradient"] = np.hypot(gx_h, gy_h)
    out["radar_gradient"] = np.hypot(gx_r, gy_r)

    # A normalized discrete Laplacian is used as a compact curvature proxy.
    lap = ndimage.laplace(height, mode="nearest")
    spacing2 = max((abs(pixel_size_x) + abs(pixel_size_y)) * 0.5, 1.0) ** 2
    out["height_curvature"] = lap / spacing2

    missing = set(ALL_FEATURES) - set(BASE_FEATURES) - set(out)
    if missing:
        raise RuntimeError(f"Feature generation incomplete: {sorted(missing)}")
    return out


def _read_aligned_window(
    stack: dict[str, WarpedVRT | rasterio.io.DatasetReader],
    window: Window,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    arrays: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for name, src in stack.items():
        arr = src.read(1, window=window).astype(np.float64, copy=False)
        valid = _safe_nodata_mask(arr, src.nodata)
        arrays[name] = arr
        masks[name] = valid
    return arrays, masks


def aligned_rasters():
    """Return (ExitStack, radar_reference, aligned_sources)."""
    stack = ExitStack()
    radar = stack.enter_context(rasterio.open(RADAR_FILE))
    sources = {"radar": radar}
    for name in ("height", "emissivity", "slope", "reflectivity"):
        src = stack.enter_context(rasterio.open(BASE_PATHS[name]))
        vrt = stack.enter_context(
            WarpedVRT(
                src,
                crs=radar.crs,
                transform=radar.transform,
                width=radar.width,
                height=radar.height,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                nodata=-32768.0,
            )
        )
        sources[name] = vrt
    return stack, radar, sources


def iter_blocks(width: int, height: int, block_size: int) -> Iterator[tuple[int, int, Window]]:
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            h = min(block_size, height - row)
            w = min(block_size, width - col)
            yield row, col, Window(col, row, w, h)


def _valid_count_pass(radar: rasterio.io.DatasetReader, sources: dict, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    keys = []
    for row, col, window in iter_blocks(radar.width, radar.height, block_size):
        arrays, masks = _read_aligned_window(sources, window)
        valid = np.logical_and.reduce([masks[name] for name in BASE_FEATURES])
        counts.append(int(valid.sum()))
        keys.append((row, col, window))
    return np.asarray(counts, dtype=np.int64), np.asarray(keys, dtype=object)


def _allocate_stratified_sample(counts: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    total = int(counts.sum())
    if total < sample_size:
        raise ValueError(f"Only {total:,} valid aligned pixels are available; requested {sample_size:,}.")
    expected = counts.astype(np.float64) * sample_size / total
    allocation = np.floor(expected).astype(np.int64)
    remainder = sample_size - int(allocation.sum())
    if remainder:
        probs = expected - allocation
        order = np.argsort(probs)[::-1]
        allocation[order[:remainder]] += 1
    return allocation


def build_advanced_training_sample(
    output_csv: str | Path,
    sample_size: int = 100_000,
    random_state: int = 42,
    block_size: int = 512,
    local_radius: int = 2,
) -> dict[str, int | float | str]:
    """Build a reproducible enriched training table from aligned Venus rasters.

    Sampling is stratified across raster blocks proportional to their valid-pixel count,
    then local feature windows are evaluated only for sampled blocks.
    """
    import pandas as pd

    rng = np.random.default_rng(random_state)
    ctx, radar, sources = aligned_rasters()
    with ctx:
        counts, keys = _valid_count_pass(radar, sources, block_size)
        total_valid = int(counts.sum())
        allocation = _allocate_stratified_sample(counts, sample_size, rng)
        records: list[dict[str, float | int]] = []
        radius = int(local_radius)
        px_x, px_y = abs(radar.res[0]), abs(radar.res[1])

        for block_idx, (_, _, base_window) in enumerate(keys):
            take = int(allocation[block_idx])
            if take == 0:
                continue
            halo_window = Window(
                max(0, int(base_window.col_off) - radius),
                max(0, int(base_window.row_off) - radius),
                min(radar.width - max(0, int(base_window.col_off) - radius), int(base_window.width) + 2 * radius),
                min(radar.height - max(0, int(base_window.row_off) - radius), int(base_window.height) + 2 * radius),
            )
            arrays, masks = _read_aligned_window(sources, halo_window)
            valid = np.logical_and.reduce([masks[name] for name in BASE_FEATURES])
            center_row0 = int(base_window.row_off) - int(halo_window.row_off)
            center_col0 = int(base_window.col_off) - int(halo_window.col_off)
            center = valid[center_row0:center_row0 + int(base_window.height), center_col0:center_col0 + int(base_window.width)]
            valid_positions = np.argwhere(center)
            if take > len(valid_positions):
                raise RuntimeError("Block allocation exceeded available valid pixels.")
            chosen_idx = rng.choice(len(valid_positions), size=take, replace=False)
            chosen = valid_positions[chosen_idx]

            derived = derive_local_features(arrays, masks, radius, px_x, px_y)
            for r, c in chosen:
                rr = center_row0 + int(r)
                cc = center_col0 + int(c)
                global_row = int(base_window.row_off) + int(r)
                global_col = int(base_window.col_off) + int(c)
                record: dict[str, float | int] = {"row": global_row, "col": global_col}
                for name in BASE_FEATURES:
                    record[name] = float(arrays[name][rr, cc])
                for name in derived:
                    record[name] = float(derived[name][rr, cc])
                records.append(record)

    df = pd.DataFrame.from_records(records)
    if len(df) != sample_size:
        raise RuntimeError(f"Expected {sample_size:,} samples but built {len(df):,}.")
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return {
        "samples": int(len(df)),
        "total_valid_aligned_pixels": total_valid,
        "feature_count": len([c for c in df.columns if c not in {"row", "col"}]),
        "output": str(output),
    }
