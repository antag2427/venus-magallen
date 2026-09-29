from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

BASE_FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity",
]

DERIVED_FEATURES = [
    "radar_local_std",
    "height_local_std",
    "emissivity_local_std",
    "slope_local_std",
    "reflectivity_local_std",
    "radar_local_range",
    "height_local_range",
    "emissivity_local_range",
    "reflectivity_local_range",
    "height_curvature",
    "radar_gradient",
    "height_gradient",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


@dataclass(frozen=True)
class V2Config:
    sample_size: int = 100_000
    sample_min_candidate_factor: float = 1.10
    sample_batch_factor: float = 1.5
    random_state: int = 42
    block_size: int = 512
    local_radius: int = 2
    k: int = 4
    consensus_subset: int = 20_000
    anomaly_subset: int = 50_000
    anomaly_contamination: float = 0.01
    isolation_estimators: int = 300
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 500
    gmm_n_init: int = 3
    gmm_max_iter: int = 300
    birch_threshold: float = 0.5
    minibatch_batch_size: int = 2048
    object_downsample_max_dim: int = 2048
    output_prefix: str = "v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "V2Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
