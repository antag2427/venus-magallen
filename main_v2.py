from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.paths import K4_RASTER
from src.v2.anomaly import run_anomaly_detection
from src.v2.config import V2Config
from src.v2.consensus import run_consensus
from src.v2.discovery import kmeans_stability
from src.v2.feature_diagnostics import run_feature_diagnostics
from src.v2.geology_association import discover_and_run_associations
from src.v2.full_map import build_full_v2_maps
from src.v2.report import write_json, write_stage_report
from src.v2.raster_features import build_advanced_training_sample
from src.v2.spatial_objects import extract_coarse_objects
from src.v2.uncertainty import fit_distance_uncertainty
from src.v2.visualization import (
    save_anomaly_distribution,
    save_confidence_histogram,
    save_consensus_agreement,
    save_top_anomalies,
)


PROJECT_ROOT = Path(__file__).resolve().parent
V2_PROCESSED = PROJECT_ROOT / "data" / "processed" / "v2"
V2_CLUSTERING = PROJECT_ROOT / "outputs" / "v2" / "clustering"
V2_ANALYSIS = PROJECT_ROOT / "outputs" / "v2" / "analysis"
V2_SPATIAL = PROJECT_ROOT / "outputs" / "v2" / "spatial"
V2_FIGURES = PROJECT_ROOT / "figures" / "v2"
CONFIG_PATH = PROJECT_ROOT / "configs" / "v2_default.json"


def load_config() -> V2Config:
    if CONFIG_PATH.exists():
        return V2Config.from_json(CONFIG_PATH)
    return V2Config()


def stage1(cfg: V2Config) -> None:
    for d in (V2_PROCESSED,):
        d.mkdir(parents=True, exist_ok=True)
    result = build_advanced_training_sample(
        V2_PROCESSED / "venus_advanced_training_sample.csv",
        sample_size=cfg.sample_size,
        random_state=cfg.random_state,
        block_size=cfg.block_size,
        local_radius=cfg.local_radius,
    )
    diagnostics = run_feature_diagnostics(
        V2_PROCESSED / "venus_advanced_training_sample.csv",
        V2_PROCESSED / "diagnostics",
    )
    payload = {"feature_build": result, "diagnostics": diagnostics}
    write_json(V2_PROCESSED / "stage1_summary.json", payload)
    write_stage_report(V2_PROCESSED, "Stage 1 — Planetary Feature Expansion", [("Feature build", result), ("Feature diagnostics", diagnostics)], "stage1_report.md")
    print(json.dumps(payload, indent=2))


def stage2(cfg: V2Config) -> None:
    sample = V2_PROCESSED / "venus_advanced_training_sample.csv"
    if not sample.exists():
        raise FileNotFoundError("Run `python main_v2.py features` first.")
    result = run_consensus(sample, V2_CLUSTERING, k=cfg.k, subset_size=cfg.consensus_subset, random_state=cfg.random_state, n_init=cfg.kmeans_n_init)
    stability = kmeans_stability(sample, V2_CLUSTERING / "v2_kmeans_stability_pairs.csv", k=cfg.k, n_init=cfg.kmeans_n_init)
    write_json(V2_CLUSTERING / "stage2_summary.json", {"consensus": result, "stability": stability})
    write_stage_report(V2_CLUSTERING, "Stage 2 — Multi-Model Surface Regime Discovery", [("Consensus", result), ("K-Means stability", stability)], "stage2_report.md")
    print(json.dumps({"consensus": result, "stability": stability}, indent=2))


def stage3(cfg: V2Config) -> None:
    sample = V2_PROCESSED / "venus_advanced_training_sample.csv"
    if not sample.exists():
        raise FileNotFoundError("Run `python main_v2.py features` first.")
    V2_ANALYSIS.mkdir(parents=True, exist_ok=True)
    uncertainty = fit_distance_uncertainty(sample, V2_ANALYSIS / "v2_cluster_uncertainty.csv", k=cfg.k, random_state=cfg.random_state, n_init=cfg.kmeans_n_init)
    anomaly = run_anomaly_detection(sample, V2_ANALYSIS / "v2_anomaly_candidates.csv", k=cfg.k, random_state=cfg.random_state, contamination=cfg.anomaly_contamination, n_estimators=cfg.isolation_estimators, fit_subset=cfg.anomaly_subset)
    save_confidence_histogram(V2_ANALYSIS / "v2_cluster_uncertainty.csv", V2_FIGURES / "v2_confidence_distribution.png")
    save_anomaly_distribution(V2_ANALYSIS / "v2_anomaly_candidates.csv", V2_FIGURES / "v2_anomaly_distribution.png")
    save_consensus_agreement(V2_CLUSTERING / "v2_consensus_labels.csv", V2_FIGURES / "v2_consensus_agreement.png")
    save_top_anomalies(V2_ANALYSIS / "v2_anomaly_candidates.csv", V2_FIGURES / "v2_top_anomalies.png")

    association_results = discover_and_run_associations(
        PROJECT_ROOT / "outputs" / "validation",
        V2_ANALYSIS / "geology_associations",
    )

    object_result = None
    if Path(K4_RASTER).exists():
        object_result = extract_coarse_objects(K4_RASTER, V2_SPATIAL / "v2_coarse_object_atlas.csv", max_dim=cfg.object_downsample_max_dim)

    full_map = build_full_v2_maps(
        sample,
        PROJECT_ROOT / "outputs" / "v2" / "rasters",
        k=cfg.k,
        random_state=cfg.random_state,
        block_size=cfg.block_size,
        local_radius=cfg.local_radius,
        isolation_contamination=cfg.anomaly_contamination,
        isolation_estimators=cfg.isolation_estimators,
        isolation_fit_subset=cfg.anomaly_subset,
    )

    payload = {"uncertainty": uncertainty, "anomaly": anomaly, "geology_associations": association_results, "coarse_objects": object_result, "full_map": full_map}
    write_json(V2_ANALYSIS / "stage3_summary.json", payload)
    write_stage_report(V2_ANALYSIS, "Stage 3 — Uncertainty, Anomaly, and Spatial Candidate Discovery", [("Uncertainty", uncertainty), ("Anomaly detection", anomaly), ("Geology association tables", {"compatible_tables": len(association_results)}), ("Coarse spatial objects", object_result or {"status":"skipped", "reason":"V1 K4 raster not present"}), ("Full-map V2 products", full_map)], "stage3_report.md")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Venus / Magellan V2 expansion pipeline")
    parser.add_argument("stage", choices=["features", "discover", "interpret", "all"])
    args = parser.parse_args()
    cfg = load_config()
    if args.stage in ("features", "all"):
        stage1(cfg)
    if args.stage in ("discover", "all"):
        stage2(cfg)
    if args.stage in ("interpret", "all"):
        stage3(cfg)


if __name__ == "__main__":
    main()
