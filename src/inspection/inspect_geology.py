import geopandas as gpd
GEOLOGY_FILE = "OvdaRegioMaps.gpkg"

layers = [
    "Tectonomorphic_map",
    "BrushedUnit_map",
    "LargeCraters_map",
    "IntraTesseraPlains_map"
]

for layer in layers:

    print("\n" + "=" * 70)
    print(layer)
    print("=" * 70)

    gdf = gpd.read_file(
        GEOLOGY_FILE,
        layer=layer
    )

    print("\nNumber of features:")
    print(len(gdf))

    print("\nCRS:")
    print(gdf.crs)

    print("\nColumns:")
    print(gdf.columns.tolist())

    print("\nFirst five records:")
    print(gdf.head())

    print("\nGeometry types:")
    print(gdf.geometry.geom_type.value_counts())

    print("\nNon-geometry attributes:")

    for column in gdf.columns:

        if column == "geometry":
            continue

        print("\n---", column, "---")

        print(
            gdf[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
        )