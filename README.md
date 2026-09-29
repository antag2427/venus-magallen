# Venus / Magellan Geospatial ML

Geospatial machine-learning analysis of Venus using five Magellan-derived surface features:
- Radar
- Height
- Emissivity
- Slope
- Reflectivity

## Method

1. Align the five rasters to the radar reference grid.
2. Mask invalid pixels.
3. Sample 100,000 valid pixels using a fixed seed.
4. Standardize the five features.
5. Apply K-Means clustering.
6. Evaluate K from 2 to 10.
7. Analyze K-Means stability.
8. Predict the K=4 model across the full raster.
9. Perform spatial analysis and 3x3 majority smoothing.
10. Validate cluster distributions against independent Venus geological maps.

## Main Model

K-Means: K=4, n_init=10, max_iter=300, random_state=42.

The clusters represent statistical surface-property regimes, not direct geological classes.

## Geological Validation

Independent datasets include Ovda Regio mapping, Guinevere USGS V-30, and Lada USGS V-56.

Additional experiments include PCA, GMM, DBSCAN, hierarchical clustering, spatial analysis, and stability analysis.

## Run

pip install -r requirements.txt
python main.py check
python main.py prepare
python main.py core
