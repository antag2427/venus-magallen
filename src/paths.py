from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ------------------------------------------------------------------
# Raw Venus / Magellan rasters
# ------------------------------------------------------------------

RADAR_FILE = RAW_DATA_DIR / "venus.tif"
HEIGHT_FILE = RAW_DATA_DIR / "venus_height.tif"
EMISSIVITY_FILE = RAW_DATA_DIR / "venus_emissivity.tif"
SLOPE_FILE = RAW_DATA_DIR / "venus_slope.tif"
REFLECTIVITY_FILE = RAW_DATA_DIR / "venus_reflectivity.tif"

# ------------------------------------------------------------------
# External geological datasets
# ------------------------------------------------------------------

OVDA_GEOLOGY_FILE = EXTERNAL_DATA_DIR / "OvdaRegioMaps.gpkg"

GUINEVERE_GDB = (
    EXTERNAL_DATA_DIR
    / "sim3539_Venus.gdb"
    / "sim3539_Venus.gdb"
)

LADA_GDB = (
    EXTERNAL_DATA_DIR
    / "SIM_3249_GIS"
    / "SIM_3249_GIS"
    / "ESRI_FileGeodatabase"
    / "V56_Geology.gdb"
)

# ------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLUSTERING_OUTPUT_DIR = OUTPUTS_DIR / "clustering"
RASTER_OUTPUT_DIR = OUTPUTS_DIR / "rasters"
SPATIAL_OUTPUT_DIR = OUTPUTS_DIR / "spatial"
VALIDATION_OUTPUT_DIR = OUTPUTS_DIR / "validation"

FIGURES_DIR = PROJECT_ROOT / "figures"
VALIDATION_FIGURES_DIR = FIGURES_DIR / "validation"

# ------------------------------------------------------------------
# Core generated artifacts
# ------------------------------------------------------------------

TRAINING_SAMPLE = PROCESSED_DATA_DIR / "venus_training_sample.csv"

K4_SAMPLE_CSV = (
    CLUSTERING_OUTPUT_DIR
    / "venus_clustered_k4.csv"
)

K4_CENTROIDS_CSV = (
    CLUSTERING_OUTPUT_DIR
    / "cluster_centroids_k4.csv"
)

K4_SCALER_CSV = (
    CLUSTERING_OUTPUT_DIR
    / "scaler_parameters_k4.csv"
)

K4_RASTER = (
    RASTER_OUTPUT_DIR
    / "venus_clusters_k4_full.tif"
)

K4_SMOOTHED_RASTER = (
    RASTER_OUTPUT_DIR
    / "venus_clusters_k4_smoothed.tif"
)

# ------------------------------------------------------------------
# Diagnostics / experiment outputs
# ------------------------------------------------------------------

K_EVALUATION = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_k_evaluation_corrected.csv"
)

STABILITY_RUNS = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_runs.csv"
)

STABILITY_REFERENCE_ARI = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_reference_ari.csv"
)

STABILITY_PAIRWISE_ARI = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_pairwise_ari.csv"
)

STABILITY_CENTROIDS = (
    CLUSTERING_OUTPUT_DIR
    / "kmeans_stability_centroids.csv"
)

FEATURES = [
    "radar",
    "height",
    "emissivity",
    "slope",
    "reflectivity",
]

K = 4
RANDOM_SEED = 42
N_INIT = 10
MAX_ITER = 300


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        OUTPUTS_DIR,
        CLUSTERING_OUTPUT_DIR,
        RASTER_OUTPUT_DIR,
        SPATIAL_OUTPUT_DIR,
        VALIDATION_OUTPUT_DIR,
        FIGURES_DIR,
        VALIDATION_FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

