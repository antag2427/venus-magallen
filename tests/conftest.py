from pathlib import Path
import sys
import types

import rasterio  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]

# The real repository already has src.paths. The upgrade bundle intentionally does
# not replace it, so tests inject a minimal adapter when run standalone.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import src.paths  # noqa: F401
except ModuleNotFoundError:
    mod = types.ModuleType("src.paths")
    data_dir = ROOT / "tests" / "_synthetic_data"
    raw = data_dir / "raw"
    mod.RADAR_FILE = raw / "radar.tif"
    mod.HEIGHT_FILE = raw / "height.tif"
    mod.EMISSIVITY_FILE = raw / "emissivity.tif"
    mod.SLOPE_FILE = raw / "slope.tif"
    mod.REFLECTIVITY_FILE = raw / "reflectivity.tif"
    mod.K4_RASTER = data_dir / "k4.tif"
    mod.GUINEVERE_GDB = data_dir / "guinevere.gdb"
    mod.LADA_GDB = data_dir / "lada.gdb"
    mod.OVDA_GEOLOGY_FILE = data_dir / "ovda.gpkg"
    sys.modules["src.paths"] = mod
