from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .config import ALL_FEATURES


def kmeans_stability(
    sample_csv: str | Path,
    output_csv: str | Path,
    k: int = 4,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 10, 20, 42, 100),
    n_init: int = 20,
) -> dict[str, float | int | str]:
    df = pd.read_csv(sample_csv)
    features = [c for c in ALL_FEATURES if c in df.columns]
    X = StandardScaler().fit_transform(df[features].to_numpy(dtype=np.float64))
    labels = {}
    for seed in seeds:
        labels[seed] = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=500).fit_predict(X)
    rows=[]
    for i, a in enumerate(seeds):
        for b in seeds[i+1:]:
            rows.append({"seed_a": a, "seed_b": b, "ARI": adjusted_rand_score(labels[a], labels[b])})
    result=pd.DataFrame(rows)
    out=Path(output_csv); out.parent.mkdir(parents=True, exist_ok=True); result.to_csv(out,index=False)
    return {"seeds":len(seeds),"pairs":len(result),"min_ARI":float(result.ARI.min()),"mean_ARI":float(result.ARI.mean()),"median_ARI":float(result.ARI.median()),"max_ARI":float(result.ARI.max()),"output":str(out)}
