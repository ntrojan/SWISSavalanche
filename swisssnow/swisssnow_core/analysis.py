"""
SwissSnow · orchestration layer.

`run_analysis` chains the four pipeline stages (DTM → morphology → snow →
composite) into a single call that any front-end (QGIS plugin, web app, CLI)
can drive. Front-ends only need to:

    1. provide an area of interest (a vector file path or a GeoDataFrame),
    2. provide a season window and a few parameters,
    3. optionally pass a `progress` callback to report advancement.

The function is deliberately side-effect-light: it writes its outputs under
``<base>/data/processed`` (the file contract the rest of the project already
uses) and returns an :class:`AnalysisResult` describing what was produced.
"""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import geopandas as gpd

from . import dtm, morpho, snow, composite

# A progress callback: progress(fraction in [0,1], human-readable message).
ProgressCb = Callable[[float, str], None]

AOI = Union[str, Path, "gpd.GeoDataFrame"]


@dataclass
class AnalysisResult:
    """Paths and summary produced by a full analysis run."""
    base: Path
    dtm_path: Path
    score_path: Path
    class_path: Path
    zones_path: Path
    stats: dict
    snow_stats_path: Path
    snow_factor: float
    rasters: dict = field(default_factory=dict)   # slope/aspect/curvature paths


def _noop(_frac: float, _msg: str) -> None:
    pass


def _validate_season(start: str, end: str) -> None:
    """Fail early with a clear message on bad season dates."""
    try:
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Dates must be YYYY-MM-DD (got start={start!r}, end={end!r}).")
    if d1 <= d0:
        raise ValueError(f"Season end ({end}) must be after start ({start}).")
    if d1 > date.today():
        raise ValueError(
            f"Season end ({end}) is in the future. The Open-Meteo archive only "
            "covers past dates (up to a few days ago).")


def _resolve_aoi(aoi: AOI, base: Path) -> tuple[Path, float, float]:
    """
    Normalise the AOI to a vector file path and return its WGS84 centroid
    (lat, lon) for the snow query.
    """
    if isinstance(aoi, gpd.GeoDataFrame):
        gdf = aoi
        aoi_path = base / "data" / "aoi.gpkg"
        aoi_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(aoi_path, driver="GPKG")
    else:
        aoi_path = Path(aoi)
        gdf = gpd.read_file(aoi_path)

    if gdf.crs is None:
        raise ValueError("AOI has no CRS; cannot locate it for the snow query.")
    # Bounding-box midpoint in WGS84 - version-proof across geopandas releases.
    minx, miny, maxx, maxy = gdf.to_crs("EPSG:4326").total_bounds
    return aoi_path, float((miny + maxy) / 2), float((minx + maxx) / 2)


def run_analysis(
    aoi: AOI,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    snow_mode: str = "climatology",
    n_winters: int = 10,
    winter_start_md: str = "12-01",
    winter_end_md: str = "03-31",
    base: Union[str, Path] = ".",
    target_res: float = 10.0,
    weights: Optional[dict] = None,
    snow_lapse: bool = True,
    lapse_per_100m: float = 0.06,
    max_gain_elev_m: float = 3000.0,
    wind_weight: float = 0.0,
    snow_threshold_cm: float = 5.0,
    rain_threshold_mm: float = 1.0,
    monthly: bool = False,
    keep_tiles: bool = False,
    tile_cache_dir: Optional[Union[str, Path]] = None,
    max_tiles: int = 400,
    morpho_engine: str = "auto",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    progress: Optional[ProgressCb] = None,
) -> AnalysisResult:
    """
    Run the full SwissSnow analysis for one AOI and season.

    Parameters
    ----------
    aoi : path | GeoDataFrame
        Area of interest (any CRS; reprojected internally to EPSG:2056).
    start, end : str
        Season window as ISO dates, e.g. "2023-12-01", "2024-03-31".
    base : path
        Project root; outputs go under ``<base>/data/processed``.
    target_res : float
        Output resolution in metres (10 m is a good default for regional runs).
    weights : dict, optional
        Factor weights. Defaults to composite.DEFAULT_WEIGHTS, with the wind
        weight overridden by ``wind_weight``. Re-normalised to sum to 1.
    snow_lapse : bool
        Apply the elevation snow-lapse gradient (recommended for high-relief AOIs).
    wind_weight : float
        Weight of the optional wind-loading factor (0 disables it).
    lat, lon : float, optional
        Override the snow-query point; defaults to the AOI centroid.
    progress : callable, optional
        progress(fraction, message) for UI feedback.

    Returns
    -------
    AnalysisResult
    """
    p = progress or _noop
    base = Path(base)
    if snow_mode == "season":
        _validate_season(start, end)
    elif snow_mode != "climatology":
        raise ValueError(f"Unknown snow_mode: {snow_mode!r}")

    # ── Weights ─────────────────────────────────────────────────────
    w = dict(composite.DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    w["wind"] = wind_weight
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Factor weights sum to zero.")
    w = {k: v / total for k, v in w.items()}

    # ── 0. Resolve AOI + snow point ─────────────────────────────────
    p(0.02, "Preparing area of interest…")
    aoi_path, c_lat, c_lon = _resolve_aoi(aoi, base)
    if lat is None:
        lat = c_lat
    if lon is None:
        lon = c_lon

    # ── 1. DTM ──────────────────────────────────────────────────────
    p(0.05, "Downloading and clipping terrain (swissALTI3D)…")
    if tile_cache_dir is None:
        tile_cache_dir = Path.home() / ".swisssnow" / "tiles"
    dtm_path = dtm.run_pipeline(
        base=base, aoi_path=aoi_path, target_res=target_res,
        keep_tiles=keep_tiles, tiles_dir=Path(tile_cache_dir),
        max_tiles=max_tiles, verbose=False)

    # ── 2. Morphology ───────────────────────────────────────────────
    p(0.45, "Computing slope / aspect / curvature…")
    morpho.run_pipeline(base=base, engine=morpho_engine, verbose=False)
    mpaths = morpho.build_paths(base)

    # ── 3. Snow ─────────────────────────────────────────────────────
    monthly_data = None
    if snow_mode == "climatology":
        p(0.6, f"Fetching snow climatology ({n_winters} winters, Open-Meteo)…")
        stats, _winters = snow.fetch_snow_climatology(
            lat, lon, n_winters, winter_start_md, winter_end_md,
            snow_threshold_cm, rain_threshold_mm)
    else:
        p(0.6, "Fetching snow data (Open-Meteo)…")
        stats = snow.fetch_snow_season(lat, lon, start, end,
                                       snow_threshold_cm, rain_threshold_mm)
        if monthly:
            monthly_data = snow.fetch_snow_monthly(
                lat, lon, start, end, snow_threshold_cm, rain_threshold_mm)
    factor = snow.snow_factor_scalar(stats)
    snow_json = base / "data" / "processed" / "snow_stats.json"
    snow.save_snow_stats(stats, factor, snow_json, monthly=monthly_data)

    # ── 4. Composite ────────────────────────────────────────────────
    p(0.8, "Combining factors and classifying…")
    result_stats = composite.run_pipeline(
        base=base, snow_json=snow_json, weights=w,
        snow_lapse=snow_lapse, lapse_per_100m=lapse_per_100m,
        max_gain_elev_m=max_gain_elev_m, verbose=False,
    )
    cpaths = composite.build_paths(base)

    p(1.0, "Done.")
    return AnalysisResult(
        base=base,
        dtm_path=Path(dtm_path),
        score_path=cpaths["susc_score"],
        class_path=cpaths["susc_class"],
        zones_path=cpaths["zones_gpkg"],
        stats=result_stats,
        snow_stats_path=snow_json,
        snow_factor=factor.value,
        rasters={"slope": mpaths["slope"], "aspect": mpaths["aspect"],
                 "curvature": mpaths["curvature"]},
    )
