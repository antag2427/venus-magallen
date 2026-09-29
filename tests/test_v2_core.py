from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label

from src.v2.consensus import align_labels
from src.v2.raster_features import derive_local_features
from src.v2.config import ALL_FEATURES
from src.v2.uncertainty import fit_distance_uncertainty


def test_label_alignment_recovers_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    candidate = np.array([2, 2, 0, 0, 3, 3, 1, 1])
    aligned, mapping = align_labels(reference, candidate, 4)
    assert np.array_equal(aligned, reference)
    assert mapping == {2: 0, 0: 1, 3: 2, 1: 3}


def test_derived_feature_shapes():
    rng = np.random.default_rng(42)
    arrays = {name: rng.normal(size=(11, 13)) for name in ["radar", "height", "emissivity", "slope", "reflectivity"]}
    masks = {name: np.ones((11, 13), dtype=bool) for name in arrays}
    features = derive_local_features(arrays, masks, radius=2, pixel_size_x=2025, pixel_size_y=2025)
    assert set(ALL_FEATURES) == set(arrays) | set(features)
    assert all(v.shape == (11, 13) for v in features.values())


def test_uncertainty_output(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(300, 5))
    data[:, 0] += np.repeat([-3, 0, 3], 100)
    df = pd.DataFrame(data, columns=["radar", "height", "emissivity", "slope", "reflectivity"])
    for feature in ALL_FEATURES[5:]:
        df[feature] = rng.normal(size=len(df))
    inp = tmp_path / "sample.csv"
    out = tmp_path / "uncertainty.csv"
    df.to_csv(inp, index=False)
    result = fit_distance_uncertainty(inp, out, k=4, random_state=42)
    assert out.exists()
    assert 0 <= result["mean_confidence"] <= 1
