"""
========================================================================
  SWISSSNOW · 03_snow.py
  Snow factor module
  ----------------------------------------------------------------------
  Queries the Open-Meteo Historical Weather API for snow variables
  at one or more locations, then computes a normalised snow_factor
  scalar (or raster-ready array) used as input to 04_composite.py.

  Usage (standalone):
      python 03_snow.py --lat 46.5 --lon 8.0 --start 2023-12-01 --end 2024-03-31

  Usage (imported):
      from pipeline.snow import fetch_snow_season, snow_factor_scalar
========================================================================
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ── Optional: openmeteo-requests + requests-cache for caching ──────
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    _USE_SDK = True
except ImportError:
    _USE_SDK = False


# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables requested from the API
SNOW_VARIABLES = [
    "snowfall",                    # cm/h  → accumulated over period → hs proxy
    "snow_depth",                  # m     → daily mean
    "et0_fao_evapotranspiration",  # not used but kept for future SWE estimation
]

# Historical archive also exposes snow_depth directly (m).
# For SWE we use the approximation: SWE ≈ snow_depth × density
# Swiss mean seasonal density ≈ 300 kg/m³ → factor 0.30
SNOW_DENSITY_FACTOR = 0.30   # [ ]  → SWE = snow_depth_m × 0.30 × 1000 (mm)

# Aspect scoring: N=0°, clockwise. Shadow face (N, NE, E) scores highest.
# Bins (center_deg): 0, 45, 90, 135, 180, 225, 270, 315
ASPECT_SHADOW_SCORE = {
    0:   1.0,   # N    - maximum accumulation
    45:  0.95,  # NE
    90:  0.80,  # E
    135: 0.50,  # SE
    180: 0.20,  # S    - high sublimation, lower persistence
    225: 0.10,  # SW
    270: 0.25,  # W
    315: 0.60,  # NW
}


# ════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ════════════════════════════════════════════════════════════════════

@dataclass
class SnowStats:
    """Raw seasonal statistics for a single grid point."""
    lat: float
    lon: float
    season_start: str
    season_end: str
    hs_mean_cm: float          # mean snow depth over the season (cm)
    hs_max_cm: float           # maximum daily snow depth (cm)
    hswe_mean_mm: float        # mean SWE estimate (mm)
    hswe_max_mm: float         # max SWE estimate (mm)
    n_snow_days: int           # days with snow_depth > threshold
    total_snowfall_cm: float   # cumulative snowfall over the period
    n_rain_on_snow_days: int = 0   # days with rain on an existing snowpack (instability proxy)
    elevation_m: Optional[float] = None


@dataclass
class SnowFactor:
    """Normalised [0, 1] snow factor ready for the composite model."""
    hs_norm: float
    hswe_norm: float
    days_norm: float
    value: float               # weighted combination
    weights: dict              # weights used


# ════════════════════════════════════════════════════════════════════
#  API LAYER
# ════════════════════════════════════════════════════════════════════

def _build_session():
    """Return a requests session - cached if the SDK is available."""
    if _USE_SDK:
        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        return retry(cache_session, retries=5, backoff_factor=0.2)
    return requests.Session()


def fetch_snow_season(
    lat: float,
    lon: float,
    start: str,
    end: str,
    snow_depth_threshold_cm: float = 5.0,
    rain_threshold_mm: float = 1.0,
) -> SnowStats:
    """
    Query the Open-Meteo Historical API and return a SnowStats object.

    Parameters
    ----------
    lat, lon : float
        WGS84 coordinates of the point (e.g. centroid of the AOI).
    start, end : str
        ISO dates, e.g. "2023-12-01" and "2024-03-31".
    snow_depth_threshold_cm : float
        Minimum snow depth (cm) to count a day as a snow day.

    Returns
    -------
    SnowStats
    """
    session = _build_session()

    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily": ["snowfall_sum", "snow_depth_mean", "rain_sum"],
        "timezone": "Europe/Zurich",
    }

    resp = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Open-Meteo API error: {data.get('reason', data['error'])}")

    daily = data.get("daily", {})
    # Fail loudly instead of silently producing a zero snow factor if the API
    # schema changes (e.g. a variable is renamed in a future model version).
    for var in ("snowfall_sum", "snow_depth_mean"):
        if not daily.get(var):
            raise RuntimeError(
                f"Open-Meteo returned no '{var}' data for {start}..{end} "
                f"at ({lat}, {lon}). Check the variable name in SNOW_VARIABLES / params."
            )

    snowfall  = np.array(daily.get("snowfall_sum",      []), dtype=float)   # cm/day
    snow_dep  = np.array(daily.get("snow_depth_mean",   []), dtype=float)   # m/day
    rain      = np.array(daily.get("rain_sum",          []), dtype=float)   # mm/day (liquid)

    # Convert depth m → cm
    snow_dep_cm = snow_dep * 100.0

    # Estimate SWE: SWE(mm) = depth(m) × density_factor × 1000
    hswe_mm = snow_dep * SNOW_DENSITY_FACTOR * 1000.0

    # Replace NaN with 0 for accumulation metrics
    snowfall    = np.nan_to_num(snowfall,    nan=0.0)
    snow_dep_cm = np.nan_to_num(snow_dep_cm, nan=0.0)
    hswe_mm     = np.nan_to_num(hswe_mm,     nan=0.0)
    rain        = np.nan_to_num(rain,        nan=0.0)

    n_snow_days = int(np.sum(snow_dep_cm >= snow_depth_threshold_cm))
    # Rain-on-snow: liquid rain falling on an existing snowpack - a strong
    # short-term destabiliser (lubricates weak layers, adds load). Time-varying
    # instability is otherwise outside the static model, so we surface it as a
    # season-level count rather than folding it into the susceptibility score.
    if rain.size == snow_dep_cm.size:
        n_ros = int(np.sum((rain >= rain_threshold_mm) &
                           (snow_dep_cm >= snow_depth_threshold_cm)))
    else:
        n_ros = 0

    return SnowStats(
        lat=lat,
        lon=lon,
        season_start=start,
        season_end=end,
        hs_mean_cm=float(np.mean(snow_dep_cm)),
        hs_max_cm=float(np.max(snow_dep_cm)) if len(snow_dep_cm) else 0.0,
        hswe_mean_mm=float(np.mean(hswe_mm)),
        hswe_max_mm=float(np.max(hswe_mm)) if len(hswe_mm) else 0.0,
        n_snow_days=n_snow_days,
        total_snowfall_cm=float(np.sum(snowfall)),
        n_rain_on_snow_days=n_ros,
        elevation_m=data.get("elevation"),
    )


def fetch_snow_grid(
    lats: list[float],
    lons: list[float],
    start: str,
    end: str,
    snow_depth_threshold_cm: float = 5.0,
) -> list[SnowStats]:
    """
    Fetch snow stats for a list of grid points (e.g. raster centroids).
    Open-Meteo supports up to ~100 points per call with comma-separated params;
    here we iterate per-point to stay within free-tier limits.
    """
    results = []
    total = len(lats)
    for i, (lat, lon) in enumerate(zip(lats, lons), 1):
        print(f"  [{i}/{total}] lat={lat:.4f} lon={lon:.4f}", end="\r", flush=True)
        stats = fetch_snow_season(lat, lon, start, end, snow_depth_threshold_cm)
        results.append(stats)
    print()
    return results


def _month_windows(start: str, end: str) -> list[tuple[str, str, str]]:
    """
    Split [start, end] into calendar-month windows.
    Returns a list of (label "YYYY-MM", window_start, window_end) tuples,
    each clipped to the overall [start, end] range.
    """
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end,   "%Y-%m-%d").date()
    windows = []
    cur = d0.replace(day=1)
    while cur <= d1:
        # First day of the next month
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        w_start = max(cur, d0)
        w_end   = min(nxt - timedelta(days=1), d1)
        windows.append((cur.strftime("%Y-%m"),
                        w_start.isoformat(), w_end.isoformat()))
        cur = nxt
    return windows


def fetch_snow_monthly(
    lat: float,
    lon: float,
    start: str,
    end: str,
    snow_depth_threshold_cm: float = 5.0,
    rain_threshold_mm: float = 1.0,
    **factor_kwargs,
) -> list[dict]:
    """
    Compute the snow factor for each calendar month in [start, end].

    Makes the otherwise season-static model time-resolved: returns one entry
    per month so the dashboard can show how the snow factor (and rain-on-snow
    events) evolve through the season.

    Returns
    -------
    list of dicts: {"month", "stats": SnowStats(asdict), "factor": SnowFactor(asdict)}
    """
    out = []
    for label, w_start, w_end in _month_windows(start, end):
        stats  = fetch_snow_season(lat, lon, w_start, w_end,
                                   snow_depth_threshold_cm, rain_threshold_mm)
        factor = snow_factor_scalar(stats, **factor_kwargs)
        out.append({"month": label,
                    "stats": asdict(stats), "factor": asdict(factor)})
    return out


def winter_windows(n_years: int = 10,
                   winter_start_md: str = "12-01",
                   winter_end_md: str = "03-31",
                   today: Optional[date] = None) -> list:
    """
    Generate the last `n_years` winter windows as (start_iso, end_iso) tuples,
    most recent first.

    A winter is labelled by its start year Y: it runs from Y-<winter_start_md>
    to (Y+spans)-<winter_end_md>, where spans=1 when the window crosses the new
    year (e.g. 12-01 → 03-31). Only complete winters (end date ≤ today) are used.
    """
    today = today or date.today()
    sm, sd = (int(x) for x in winter_start_md.split("-"))
    em, ed = (int(x) for x in winter_end_md.split("-"))
    spans = 1 if (sm, sd) > (em, ed) else 0   # crosses the new year?

    y = today.year
    while date(y + spans, em, ed) > today:
        y -= 1

    out = []
    for k in range(n_years):
        ys = y - k
        out.append((date(ys, sm, sd).isoformat(),
                    date(ys + spans, em, ed).isoformat()))
    return out


def fetch_snow_climatology(
    lat: float,
    lon: float,
    n_years: int = 10,
    winter_start_md: str = "12-01",
    winter_end_md: str = "03-31",
    snow_depth_threshold_cm: float = 5.0,
    rain_threshold_mm: float = 1.0,
    progress=None,
) -> tuple:
    """
    Average the winter snow statistics over the last `n_years` winters.

    This gives the *typical* (climatological) snow load of the area, the right
    basis for a susceptibility map - a single winter can be anomalous.
    Returns (averaged SnowStats, list of per-winter SnowStats).
    """
    windows = winter_windows(n_years, winter_start_md, winter_end_md)
    per_winter = []
    for i, (s, e) in enumerate(windows, 1):
        if progress:
            progress(i / len(windows), f"Snow climatology: winter {s[:4]}…")
        try:
            per_winter.append(
                fetch_snow_season(lat, lon, s, e,
                                  snow_depth_threshold_cm, rain_threshold_mm))
        except Exception:
            continue   # skip a winter with missing/failed data
    if not per_winter:
        raise RuntimeError(
            "Could not fetch snow data for any winter in the climatology window.")

    def avg(attr):
        return float(np.mean([getattr(s, attr) for s in per_winter]))

    averaged = SnowStats(
        lat=lat, lon=lon,
        season_start=windows[-1][0], season_end=windows[0][1],
        hs_mean_cm=avg("hs_mean_cm"), hs_max_cm=avg("hs_max_cm"),
        hswe_mean_mm=avg("hswe_mean_mm"), hswe_max_mm=avg("hswe_max_mm"),
        n_snow_days=int(round(avg("n_snow_days"))),
        total_snowfall_cm=avg("total_snowfall_cm"),
        n_rain_on_snow_days=int(round(avg("n_rain_on_snow_days"))),
        elevation_m=per_winter[0].elevation_m,
    )
    return averaged, per_winter


# ════════════════════════════════════════════════════════════════════
#  NORMALISATION + SNOW FACTOR
# ════════════════════════════════════════════════════════════════════

def _normalize(value: float, ref_max: float, clip: bool = True) -> float:
    """Normalise a single value to [0, 1] given a reference maximum."""
    if ref_max <= 0:
        return 0.0
    n = value / ref_max
    return float(np.clip(n, 0.0, 1.0)) if clip else float(n)


def snow_factor_scalar(
    stats: SnowStats,
    ref_hs_max_cm: float = 200.0,
    ref_hswe_max_mm: float = 600.0,
    ref_n_days_max: float = 120.0,
    w_hs: float = 0.40,
    w_hswe: float = 0.40,
    w_days: float = 0.20,
) -> SnowFactor:
    """
    Compute the normalised snow factor for a single point.

    Reference maxima represent the 'very high snow season' baseline
    for the Swiss Alps. Adjust them to your study region if needed.

    Parameters
    ----------
    stats : SnowStats
        Output of fetch_snow_season().
    ref_hs_max_cm : float
        Reference maximum mean snow depth (cm). Default: 200 cm.
    ref_hswe_max_mm : float
        Reference maximum mean SWE (mm). Default: 600 mm.
    ref_n_days_max : float
        Reference maximum snow-day count. Default: 120 days.
    w_hs, w_hswe, w_days : float
        Weights summing to 1.0.

    Returns
    -------
    SnowFactor
    """
    assert abs(w_hs + w_hswe + w_days - 1.0) < 1e-6, "Weights must sum to 1."

    hs_n   = _normalize(stats.hs_mean_cm,    ref_hs_max_cm)
    hswe_n = _normalize(stats.hswe_mean_mm,  ref_hswe_max_mm)
    days_n = _normalize(stats.n_snow_days,   ref_n_days_max)

    value = w_hs * hs_n + w_hswe * hswe_n + w_days * days_n

    return SnowFactor(
        hs_norm=hs_n,
        hswe_norm=hswe_n,
        days_norm=days_n,
        value=float(np.clip(value, 0.0, 1.0)),
        weights={"w_hs": w_hs, "w_hswe": w_hswe, "w_days": w_days},
    )


def snow_factor_array(
    stats_list: list[SnowStats],
    shape: tuple[int, int],
    transform,
    **kwargs,
) -> np.ndarray:
    """
    Build a 2-D snow_factor raster from a list of SnowStats objects.

    This is a simple IDW interpolation from point factors onto the DTM grid.
    For a single-point AOI centroid, the array is filled with a constant.

    Parameters
    ----------
    stats_list : list[SnowStats]
        One or more points with their snow stats.
    shape : (rows, cols)
        Target raster shape, matching the DTM clip.
    transform : affine.Affine
        Affine transform of the target raster.
    **kwargs
        Passed to snow_factor_scalar().

    Returns
    -------
    np.ndarray  shape=(rows, cols), dtype=float32, values in [0, 1]
    """
    factors = [snow_factor_scalar(s, **kwargs) for s in stats_list]
    factor_values = np.array([f.value for f in factors], dtype=np.float64)

    if len(factors) == 1:
        # Constant fill - single centroid query
        return np.full(shape, factor_values[0], dtype=np.float32)

    # IDW interpolation (p=2) onto the grid
    from affine import Affine  # local import - rasterio ships it
    rows, cols = shape
    xs = np.array([s.lon for s in stats_list])
    ys = np.array([s.lat for s in stats_list])

    # Grid lon/lat
    col_idx = np.arange(cols)
    row_idx = np.arange(rows)
    col_grid, row_grid = np.meshgrid(col_idx, row_idx)
    grid_x, grid_y = transform * (col_grid + 0.5, row_grid + 0.5)

    # Distances from each grid cell to each sample point
    out = np.zeros(shape, dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)

    for i, (px, py, fv) in enumerate(zip(xs, ys, factor_values)):
        dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
        dist2 = np.where(dist2 == 0, 1e-12, dist2)
        w = 1.0 / dist2
        out += w * fv
        weight_sum += w

    return (out / np.where(weight_sum == 0, 1e-12, weight_sum)).astype(np.float32)


def snow_factor_lapse_array(
    base_stats: SnowStats,
    dtm_arr: np.ndarray,
    ref_elev_m: Optional[float] = None,
    lapse_per_100m: float = 0.06,
    max_gain_elev_m: float = 3000.0,
    n_bins: int = 64,
    **norm_kwargs,
) -> np.ndarray:
    """
    Build an elevation-dependent snow_factor raster from a single point query.

    Open-Meteo's historical archive returns snow depth from a coarse ERA5 grid
    cell and does NOT downscale it by elevation (verified empirically: the
    ``elevation`` parameter only adjusts temperature). To remove the
    single-point bias in AOIs with significant relief, we apply a snow lapse
    gradient: snow depth / SWE / snow-days are scaled linearly with elevation
    relative to the query point, then re-normalised against the fixed reference
    maxima. The gradient is capped above ``max_gain_elev_m`` because above the
    accumulation zone wind scour and steep terrain stop the increase.

    Parameters
    ----------
    base_stats : SnowStats
        Point query (typically the AOI centroid) from fetch_snow_season().
    dtm_arr : np.ndarray
        Clipped DTM (metres), NaN outside the AOI. Defines the target grid.
    ref_elev_m : float, optional
        Anchor elevation where the gradient is neutral (gain = 1). Defaults to
        the elevation Open-Meteo snapped the query to (base_stats.elevation_m).
    lapse_per_100m : float
        Fractional change in snow depth per 100 m. Default 0.06 (+6 %/100 m),
        a typical Swiss-Alps accumulation-zone gradient.
    max_gain_elev_m : float
        Elevation above which the gain stops increasing.
    n_bins : int
        Number of elevation bins used to build the elevation→factor lookup.
    **norm_kwargs
        Passed to snow_factor_scalar() (reference maxima, weights).

    Returns
    -------
    np.ndarray, same shape as dtm_arr, float32, values in [0, 1], NaN outside AOI.
    """
    if ref_elev_m is None:
        ref_elev_m = base_stats.elevation_m
    if ref_elev_m is None:
        raise ValueError("ref_elev_m is None and base_stats has no elevation_m.")

    finite = np.isfinite(dtm_arr)
    if not finite.any():
        raise ValueError("DTM has no valid pixels.")

    z_min = float(np.nanmin(dtm_arr))
    z_max = float(np.nanmax(dtm_arr))
    # Degenerate flat AOI: a single bin would break np.interp.
    bin_elevs = (np.array([z_min], dtype=float) if z_max == z_min
                 else np.linspace(z_min, z_max, n_bins))

    bin_factors = np.empty_like(bin_elevs, dtype=float)
    for i, z in enumerate(bin_elevs):
        eff_z = min(z, max_gain_elev_m)
        gain = max(0.0, 1.0 + lapse_per_100m * (eff_z - ref_elev_m) / 100.0)
        scaled = SnowStats(
            lat=base_stats.lat, lon=base_stats.lon,
            season_start=base_stats.season_start, season_end=base_stats.season_end,
            hs_mean_cm=base_stats.hs_mean_cm * gain,
            hs_max_cm=base_stats.hs_max_cm * gain,
            hswe_mean_mm=base_stats.hswe_mean_mm * gain,
            hswe_max_mm=base_stats.hswe_max_mm * gain,
            n_snow_days=base_stats.n_snow_days * gain,
            total_snowfall_cm=base_stats.total_snowfall_cm * gain,
            elevation_m=z,
        )
        bin_factors[i] = snow_factor_scalar(scaled, **norm_kwargs).value

    out = np.full(dtm_arr.shape, np.nan, dtype=np.float32)
    if bin_elevs.size == 1:
        out[finite] = bin_factors[0]
    else:
        out[finite] = np.interp(dtm_arr[finite], bin_elevs, bin_factors)
    return out


# ════════════════════════════════════════════════════════════════════
#  ASPECT WEIGHTING  (used by 04_composite.py)
# ════════════════════════════════════════════════════════════════════

def aspect_shadow_weight(aspect_deg_arr: np.ndarray) -> np.ndarray:
    """
    Convert an aspect raster (degrees, 0=N clockwise) to a shadow-face
    weight in [0, 1]. Uses bilinear interpolation between the 8 cardinal
    scores defined in ASPECT_SHADOW_SCORE.

    Parameters
    ----------
    aspect_deg_arr : np.ndarray
        Aspect in degrees [0, 360). Shape arbitrary.

    Returns
    -------
    np.ndarray, same shape, float32, values in [0, 1].
    """
    arr = np.asarray(aspect_deg_arr, dtype=np.float64) % 360.0
    centers = sorted(ASPECT_SHADOW_SCORE.keys())   # [0, 45, 90, ..., 315]
    scores  = [ASPECT_SHADOW_SCORE[c] for c in centers]

    # Extend for wrap-around: append 360 = same as 0
    centers_ext = centers + [360]
    scores_ext  = scores  + [scores[0]]

    out = np.interp(arr, centers_ext, scores_ext)
    return out.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  PERSISTENCE BONUS  (optional refinement)
# ════════════════════════════════════════════════════════════════════

def persistence_bonus(
    stats: SnowStats,
    consecutive_days_threshold: int = 30,
) -> float:
    """
    Returns a small bonus [0, 0.15] if the season has a long continuous
    snow cover (proxy: n_snow_days >= threshold).

    A long persistent snowpack is more likely to develop weak depth-hoar
    layers that trigger slab avalanches. Not used in the base composite
    but available as an optional additive correction.
    """
    ratio = stats.n_snow_days / max(consecutive_days_threshold, 1)
    return float(np.clip(ratio * 0.15, 0.0, 0.15))


# ════════════════════════════════════════════════════════════════════
#  EXPORT HELPER
# ════════════════════════════════════════════════════════════════════

def save_snow_stats(stats: SnowStats, factor: SnowFactor, out_path: Path,
                    monthly: Optional[list] = None) -> None:
    """Serialise SnowStats + SnowFactor to JSON for downstream modules."""
    payload = {
        "stats": asdict(stats),
        "factor": asdict(factor),
    }
    if monthly is not None:
        payload["monthly"] = monthly
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  → saved {out_path}")


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="SwissSnow · fetch snow data and compute snow_factor"
    )
    parser.add_argument("--lat",   type=float, required=True,  help="Latitude (WGS84)")
    parser.add_argument("--lon",   type=float, required=True,  help="Longitude (WGS84)")
    parser.add_argument("--start", type=str,   required=True,  help="Season start, e.g. 2023-12-01")
    parser.add_argument("--end",   type=str,   required=True,  help="Season end,   e.g. 2024-03-31")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Snow-day threshold in cm (default: 5)")
    parser.add_argument("--rain-threshold", type=float, default=1.0,
                        help="Rain-on-snow threshold in mm/day (default: 1)")
    parser.add_argument("--monthly", action="store_true",
                        help="Also compute a per-month breakdown of the snow factor")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output JSON path")
    args = parser.parse_args()

    print(f"\nFetching snow data for ({args.lat}, {args.lon})")
    print(f"Season: {args.start} → {args.end}\n")

    stats  = fetch_snow_season(args.lat, args.lon, args.start, args.end,
                               args.threshold, args.rain_threshold)
    factor = snow_factor_scalar(stats)

    # Pretty print
    print("── Snow statistics ─────────────────────────────")
    print(f"  Mean snow depth:    {stats.hs_mean_cm:6.1f} cm")
    print(f"  Max  snow depth:    {stats.hs_max_cm:6.1f} cm")
    print(f"  Mean SWE:           {stats.hswe_mean_mm:6.1f} mm")
    print(f"  Max  SWE:           {stats.hswe_max_mm:6.1f} mm")
    print(f"  Snow days (>{args.threshold} cm): {stats.n_snow_days:4d} days")
    print(f"  Total snowfall:     {stats.total_snowfall_cm:6.1f} cm")
    print(f"  Rain-on-snow days:  {stats.n_rain_on_snow_days:4d} days  (instability proxy)")
    if stats.elevation_m:
        print(f"  Station elevation:  {stats.elevation_m:6.0f} m a.s.l.")
    print()
    print("── Snow factor ─────────────────────────────────")
    print(f"  hs_norm:    {factor.hs_norm:.3f}  (w={factor.weights['w_hs']})")
    print(f"  hswe_norm:  {factor.hswe_norm:.3f}  (w={factor.weights['w_hswe']})")
    print(f"  days_norm:  {factor.days_norm:.3f}  (w={factor.weights['w_days']})")
    print(f"  ─────────────────────")
    print(f"  snow_factor = {factor.value:.4f}")
    print()

    monthly = None
    if args.monthly:
        print("── Monthly breakdown ───────────────────────────")
        monthly = fetch_snow_monthly(args.lat, args.lon, args.start, args.end,
                                     args.threshold, args.rain_threshold)
        for m in monthly:
            print(f"  {m['month']}:  snow_factor={m['factor']['value']:.3f}   "
                  f"hs_mean={m['stats']['hs_mean_cm']:5.1f}cm   "
                  f"ros={m['stats']['n_rain_on_snow_days']}d")
        print()

    if args.out:
        save_snow_stats(stats, factor, Path(args.out), monthly=monthly)

    return stats, factor


if __name__ == "__main__":
    _cli()
