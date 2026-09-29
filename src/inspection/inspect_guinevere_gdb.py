import os
import geopandas as gpd

from src.paths import GUINEVERE_GDB


# ============================================================
# ACTUAL PATH TO YOUR USGS GEODATABASE
# ============================================================

GDB_PATH = GUINEVERE_GDB


print("=" * 100)
print("USGS GUINEVERE V-30 GEODATABASE INSPECTION")
print("=" * 100)


# ============================================================
# CHECK PATH
# ============================================================

print("\nChecking geodatabase...")

if not os.path.exists(GDB_PATH):

    print("\nERROR: Geodatabase not found.")

    print("\nPython is looking here:")
    print(os.path.abspath(GDB_PATH))

    input("\nPress Enter to close...")
    raise SystemExit


print("\nGeodatabase FOUND:")
print(os.path.abspath(GDB_PATH))


# ============================================================
# CHECK GEOPANDAS
# ============================================================

print("\nGeoPandas version:")
print(gpd.__version__)


# ============================================================
# LIST LAYERS
# ============================================================

print("\n")
print("=" * 100)
print("AVAILABLE LAYERS")
print("=" * 100)


try:

    layers = gpd.list_layers(GDB_PATH)

    print(
        f"\nNumber of layers: {len(layers)}\n"
    )

    print(
        layers.to_string(index=False)
    )

except Exception as e:

    print("\nERROR WHILE READING GEODATABASE:")

    print(e)

    input("\nPress Enter to close...")
    raise SystemExit


# ============================================================
# INSPECT EACH LAYER
# ============================================================

for layer_name in layers["name"]:

    print("\n")
    print("=" * 100)
    print(f"LAYER: {layer_name}")
    print("=" * 100)

    try:

        gdf = gpd.read_file(
            GDB_PATH,
            layer=layer_name
        )

        print(
            f"\nNumber of features: "
            f"{len(gdf):,}"
        )

        print(
            f"\nCRS:\n{gdf.crs}"
        )

        print(
            "\nColumns:"
        )

        for column in gdf.columns:

            print(
                f"  {column}"
            )

        print(
            "\nFirst 5 records:"
        )

        print(
            gdf.head()
            .to_string(index=False)
        )

    except Exception as e:

        print(
            f"\nERROR reading layer "
            f"'{layer_name}':"
        )

        print(e)


print("\n")
print("=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)

input("\nPress Enter to close...")