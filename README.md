# SwissSnow (QGIS plugin)

Seasonal avalanche-susceptibility mapping for the Swiss Alps, as a QGIS
Processing plugin. It combines morphological terrain factors (slope, aspect,
profile curvature) from the 2 m swissALTI3D DTM with a snow-load factor from the
Open-Meteo historical archive, with optional elevation snow-lapse and
wind-loading corrections.

This is a standalone tool, not affiliated with or validated by SLF or swisstopo.
Susceptibility is not hazard: do not use it for operational safety decisions.

## Install

1. Download `swisssnow.zip` (or zip the `swisssnow/` folder in this repo).
2. In QGIS: Plugins > Manage and Install Plugins > Install from ZIP.
3. Enable SwissSnow (it is flagged experimental, so tick "Show also
   experimental plugins" if needed).

The `swisssnow/` folder is self-contained: the computational engine
(`swisssnow_core`) is bundled inside it, so no extra Python packages beyond what
QGIS already ships (numpy, rasterio, geopandas, requests, GDAL) are required.

## Tools (Processing toolbox > SwissSnow)

### Avalanche susceptibility
Pick an area of interest (a polygon layer or a map extent), choose how the snow
load is measured, and get a 4-class susceptibility raster, a continuous score
raster, and zone polygons, all styled automatically.

- Snow basis: "Climatology" (default, averages the last N winters over a fixed
  winter window) or "Single winter" (specific dates) to study or compare a
  particular season.
- Advanced: factor weights, snow-lapse gradient, snow-day threshold, tile cap,
  DTM tile cache folder, and the climatology settings.

Needs an internet connection (terrain tiles + snow API). Downloaded DTM tiles
are cached under `~/.swisssnow/tiles` and reused across runs.

### Validate against incidents
Checks how well a susceptibility map discriminates real avalanche events.
Provide your own incident points or let the plugin download the official SLF
avalanche-accident dataset (Switzerland, since 1970) and clip it to the map
area. Reports the frequency ratio per class (FR > 1 means accidents concentrate
there) plus a capture/lift figure.

## Data sources

- swissALTI3D digital terrain model, swisstopo (STAC API).
- Historical weather archive, Open-Meteo.
- Avalanche-accident dataset, WSL Institute for Snow and Avalanche Research SLF
  (via EnviDat).

## License

Released under the MIT License. See `LICENSE`.
