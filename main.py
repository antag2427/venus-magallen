import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paths import (  # noqa: E402
    RADAR_FILE,
    HEIGHT_FILE,
    EMISSIVITY_FILE,
    SLOPE_FILE,
    REFLECTIVITY_FILE,
    TRAINING_SAMPLE,
    K4_SAMPLE_CSV,
    K4_CENTROIDS_CSV,
    K4_SCALER_CSV,
    K4_RASTER,
    ensure_project_dirs,
)

CORE_STAGES = {
    "prepare": SRC / "preprocessing" / "build_training_sample.py",
    "kmeans": SRC / "modeling" / "kmeans_model_k4.py",
    "evaluate-k": SRC / "modeling" / "evaluate_k.py",
    "stability": SRC / "modeling" / "kmeans_stability.py",
}


def check_inputs() -> None:
    print("=" * 80)
    print("VENUS / MAGELLAN PROJECT INPUT CHECK")
    print("=" * 80)
    for path in (
        RADAR_FILE,
        HEIGHT_FILE,
        EMISSIVITY_FILE,
        SLOPE_FILE,
        REFLECTIVITY_FILE,
    ):
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{status:>7}: {path}")


def run_stage(name: str) -> None:
    script = CORE_STAGES[name]
    print("\n" + "=" * 80)
    print(f"RUNNING STAGE: {name}")
    print("=" * 80)
    subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )


def show_status() -> None:
    print("\nCurrent generated core artifacts:")
    for path in (TRAINING_SAMPLE, K4_SAMPLE_CSV, K4_CENTROIDS_CSV, K4_SCALER_CSV, K4_RASTER):
        print(f"{'FOUND' if path.exists() else '-----'}  {path}")


def main() -> None:
    ensure_project_dirs()
    parser = argparse.ArgumentParser(
        description="Launcher for the Venus / Magellan geospatial ML project."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=["check", "prepare", "kmeans", "evaluate-k", "stability", "core"],
        default="check",
        help="Pipeline stage to run.",
    )
    args = parser.parse_args()

    if args.stage == "check":
        check_inputs()
        show_status()
        print("\nCore stages: prepare -> kmeans -> evaluate-k -> stability")
        return

    check_inputs()
    missing = [
        p for p in (
            RADAR_FILE,
            HEIGHT_FILE,
            EMISSIVITY_FILE,
            SLOPE_FILE,
            REFLECTIVITY_FILE,
        ) if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Copy the five Venus TIFFs into data/raw before running the pipeline."
        )

    if args.stage == "core":
        for stage in ("prepare", "kmeans", "evaluate-k", "stability"):
            run_stage(stage)
    else:
        run_stage(args.stage)

    show_status()


if __name__ == "__main__":
    main()
