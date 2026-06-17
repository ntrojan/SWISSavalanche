"""
========================================================================
  SWISSSNOW · 04_composite.py
  Composite susceptibility model
  ----------------------------------------------------------------------
  Reads pre-computed morphological rasters (slope, aspect, curvature)
  produced by 02_morpho.py and the snow factor from 03_snow.py, then:

    1. Normalises every factor to [0, 1]
    2. Combines them into a weighted susceptibility score
    3. Classifies the score into 4 ordinal classes at the quartiles
    4. Exports:
         • data/processed/rasters/susceptibility_class.tif   (raster, Int16)
         • data/processed/vectors/susceptibility_zones.gpkg  (polygons)
         • data/processed/stats.json                         (statistics)

  Usage:
      python 04_composite.py \
          --snow-json data/processed/snow_stats.json \
          --start 2023-12-01 --end 2024-03-31

  All raster paths are resolved relative to the project BASE directory.
  Override with --base /path/to/project if needed.
========================================================================
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes as raster_shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import shape

warnings.filterwarnings("ignore")

# Sibling snow module (proper package-relative import).
from .snow import (                                      # noqa: E402
    aspect_shadow_weight,
    snow_factor_lapse_array,
    SnowStats,
)


# ════════════════════════════════════════════════════════════════════
#  PATHS  (relative to project BASE)
# ════════════════════════════════════════════════════════════════════

def build_paths(base: Path) -> dict:
    r = base / "data" / "processed" / "rasters"
    v = base / "data" / "processed" / "vectors"
    return {
        # inputs
        "dtm":     r / "dtm_clip.tif",
        "slope":   r / "slope.tif",
        "aspect":  r / "aspect.tif",
        "curv":    r / "curvature.tif",
        # outputs
        "susc_score": r / "susceptibility_score.tif",
        "susc_class": r / "susceptibility_class.tif",
        "zones_gpkg": v / "susceptibility_zones.gpkg",
        "stats_json": base / "data" / "processed" / "stats.json",
    }


# ════════════════════════════════════════════════════════════════════
#  WEIGHTS  (must sum to 1.0)
# ════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "slope":  0.35,   # primary mechanical trigger
    "snow":   0.25,   # dynamic seasonal load
    "aspect": 0.20,   # shadow-face accumulation proxy
    "curv":   0.20,   # profile curvature - convex = unstable
    "wind":   0.00,   # wind-loading proxy - off by default (opt-in via CLI)
}

# Slope: only the 25-55° band is truly critical for avalanche release.
# Below 25° insufficient gravitational stress; above 55° snow slides off
# before accumulating. We score inside the band and taper the flanks.
SLOPE_BAND_LOW  = 25.0   # degrees
SLOPE_BAND_HIGH = 55.0   # degrees


# ════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ════════════════════════════════════════════════════════════════════

def read_raster(path: Path) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    """Read a single-band raster. Returns (array float32, profile)."""
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as src:
        arr  = src.read(1).astype(np.float32)
        prof = src.profile.copy()
        nd   = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr, prof


def write_raster(arr: np.ndarray, profile: dict, path: Path, dtype="float32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    prof.update(dtype=dtype, count=1, nodata=-9999 if dtype != "float32" else np.nan)
    out  = arr.astype(dtype)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
    print(f"  → {path}")


def load_snow_factor(json_path: Path) -> float:
    """Load snow_factor.value from the JSON saved by 03_snow.py."""
    if not json_path.exists():
        raise FileNotFoundError(f"Snow JSON not found: {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    value = data["factor"]["value"]
    print(f"  Snow factor loaded: {value:.4f}")
    return float(value)


def load_snow_stats(json_path: Path) -> "SnowStats":
    """Reconstruct the full SnowStats object from the JSON saved by 03_snow.py."""
    if not json_path.exists():
        raise FileNotFoundError(f"Snow JSON not found: {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    return SnowStats(**data["stats"])


# ════════════════════════════════════════════════════════════════════
#  NORMALISATION
# ════════════════════════════════════════════════════════════════════

def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1], ignoring NaN."""
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)


def normalise_slope(slope_arr: np.ndarray) -> np.ndarray:
    """
    Avalanche-specific slope normalisation.

    Score function:
      • < SLOPE_BAND_LOW  → linear ramp 0 → 0.3  (some risk, sub-critical)
      • BAND_LOW - BAND_HIGH → triangle peak at 38° (empirical optimum for slab)
      • > SLOPE_BAND_HIGH → linear decline 0.5 → 0 (snow slides off before loading)

    The 38° peak is the modal angle for slab avalanche release in alpine
    literature (McClung & Schaerer 2006).
    """
    arr = np.asarray(slope_arr, dtype=np.float64)
    out = np.zeros_like(arr)

    PEAK = 38.0   # degrees - maximum susceptibility

    # Zone 1: below critical band
    m1 = arr < SLOPE_BAND_LOW
    out[m1] = (arr[m1] / SLOPE_BAND_LOW) * 0.30

    # Zone 2: critical band - triangle centred on PEAK
    m2 = (arr >= SLOPE_BAND_LOW) & (arr <= SLOPE_BAND_HIGH)
    left  = (arr[m2] - SLOPE_BAND_LOW) / (PEAK - SLOPE_BAND_LOW)
    right = (SLOPE_BAND_HIGH - arr[m2]) / (SLOPE_BAND_HIGH - PEAK)
    out[m2] = np.where(arr[m2] <= PEAK, left, right)
    out[m2] = np.clip(out[m2], 0.0, 1.0)

    # Zone 3: above critical band - quick decay
    m3 = arr > SLOPE_BAND_HIGH
    excess = arr[m3] - SLOPE_BAND_HIGH
    out[m3] = np.clip(0.50 - excess / 30.0, 0.0, 0.50)

    return out.astype(np.float32)


def normalise_curvature(curv_arr: np.ndarray) -> np.ndarray:
    """
    Profile curvature normalisation.

    Convex slopes (positive profile curvature) are mechanically unstable
    and score high. Concave slopes (negative) can be release zones too
    but also trap redistributed snow - score moderately.
    Score: sigmoid-like mapping; strong convexity → 1.0, flat → 0.5,
    strong concavity → 0.3.
    """
    arr = np.asarray(curv_arr, dtype=np.float64)
    # Standardise
    sd  = np.nanstd(arr)
    if sd == 0:
        return np.full_like(arr, 0.5, dtype=np.float32)
    z = arr / sd   # z-score relative to local terrain variability

    # Sigmoid: centre at 0, output in [0.3, 1.0]
    sig = 1.0 / (1.0 + np.exp(-z))         # classic sigmoid, [0, 1]
    out = 0.3 + 0.7 * sig                   # rescale to [0.3, 1.0]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  COMPOSITE SCORE
# ════════════════════════════════════════════════════════════════════

def compute_wind_load(
    slope_norm: np.ndarray,
    aspect_norm: np.ndarray,
    curv_arr: np.ndarray,
    curv_norm: np.ndarray,
) -> np.ndarray:
    """
    Wind-loading proxy in [0, 1].

    Wind scours windward faces and deposits snow on convex, lee-side terrain
    that also lies in the avalanche slope band. We do not model wind direction
    explicitly; instead we combine three proxies multiplicatively so the index
    is high only where all three coincide:

        wind_load = aspect_norm · convexity · slope_in_band

    - ``aspect_norm`` - shadow/lee-face weighting (N/NE highest), a proxy for
      preferential deposition away from the prevailing W/SW winds.
    - ``convexity``   - curv_norm restricted to convex pixels (profile
      curvature > 0); concave pixels contribute 0.
    - ``slope_in_band`` - the triangular slope score (peak 38°), so loading
      only matters on release-prone slopes.

    Returns
    -------
    np.ndarray, float32, values in [0, 1].
    """
    convexity = np.where(curv_arr > 0, curv_norm, 0.0)
    wind_load = aspect_norm * convexity * slope_norm
    return np.clip(wind_load, 0.0, 1.0).astype(np.float32)


def compute_score(
    slope_norm: np.ndarray,
    aspect_norm: np.ndarray,
    curv_norm: np.ndarray,
    snow_factor: float,
    weights: dict = DEFAULT_WEIGHTS,
    wind_load: np.ndarray = None,
) -> np.ndarray:
    """
    Weighted linear combination of the normalised factors.

    snow_factor is a scalar (single seasonal value for the whole AOI).
    If you computed a snow_factor raster via snow_factor_lapse_array(), pass
    the 2-D array directly - the arithmetic broadcasts the same way.

    wind_load is an optional extra factor; its weight defaults to 0, so the
    base four-factor model is recovered when it is omitted.

    Returns
    -------
    np.ndarray, float32, values in [0, 1].
    """
    w = weights
    # snow_factor may be a scalar (uniform) or a 2-D array (elevation-lapsed);
    # the weighted sum broadcasts either way.
    score = (
        w["slope"]  * slope_norm  +
        w["snow"]   * snow_factor +
        w["aspect"] * aspect_norm +
        w["curv"]   * curv_norm
    )
    if w.get("wind", 0.0) and wind_load is not None:
        score = score + w["wind"] * wind_load
    return np.clip(score, 0.0, 1.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ════════════════════════════════════════════════════════════════════

CLASS_LABELS = {1: "Low", 2: "Moderate", 3: "High", 4: "Very high"}
CLASS_COLORS = {1: "#1a9850", 2: "#a6d96a", 3: "#fdae61", 4: "#d73027"}


def classify(score: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Split the score into 4 ordinal classes at the 25th, 50th and 75th
    percentile of valid pixels - same approach used in SwissMorph.

    Returns
    -------
    classes : np.ndarray, int16, 1-4 (NaN pixels → -9999)
    thresholds : dict  {q25, q50, q75}
    """
    valid = score[np.isfinite(score)].flatten()
    q25, q50, q75 = np.nanpercentile(valid, [25, 50, 75])

    classes = np.full(score.shape, -9999, dtype=np.int16)
    fin = np.isfinite(score)
    classes[fin & (score <= q25)] = 1
    classes[fin & (score >  q25) & (score <= q50)] = 2
    classes[fin & (score >  q50) & (score <= q75)] = 3
    classes[fin & (score >  q75)] = 4

    thresholds = {"q25": float(q25), "q50": float(q50), "q75": float(q75)}
    return classes, thresholds


# ════════════════════════════════════════════════════════════════════
#  VECTORISATION
# ════════════════════════════════════════════════════════════════════

def vectorise(
    classes: np.ndarray,
    profile: dict,
    out_path: Path,
) -> gpd.GeoDataFrame:
    """
    Convert the classified raster to a GeoPackage of polygons.

    Each polygon carries:
      • class_id    (int)
      • label       (str)  Low / Moderate / High / Very high
      • color       (str)  hex color for styling
      • area_km2    (float)
    """
    transform = profile["transform"]
    crs       = profile["crs"]

    # Mask out nodata
    mask = (classes > 0).astype(np.uint8)

    records = []
    for geom_json, val in raster_shapes(
        classes.astype(np.int32), mask=mask, transform=transform
    ):
        v = int(val)
        if v not in CLASS_LABELS:
            continue
        geom = shape(geom_json)
        records.append({
            "class_id": v,
            "label":    CLASS_LABELS[v],
            "color":    CLASS_COLORS[v],
            "geometry": geom,
        })

    if not records:
        print("  ⚠ No valid polygons to vectorise.")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Dissolve by class to merge adjacent same-class pixels.
    # (as_index=False is unavailable in old geopandas → reset_index instead.)
    gdf = gdf.dissolve(by="class_id").reset_index()
    gdf["label"] = gdf["class_id"].map(CLASS_LABELS)
    gdf["color"] = gdf["class_id"].map(CLASS_COLORS)

    # Area in km²
    if gdf.crs and gdf.crs.is_projected:
        gdf["area_km2"] = (gdf.geometry.area / 1e6).round(4)
    else:
        # Approximate via re-projection to EPSG:2056
        gdf_proj = gdf.to_crs("EPSG:2056")
        gdf["area_km2"] = (gdf_proj.geometry.area / 1e6).round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="susceptibility_zones")
    print(f"  → {out_path}  ({len(gdf)} polygons)")
    return gdf


# ════════════════════════════════════════════════════════════════════
#  STATISTICS
# ════════════════════════════════════════════════════════════════════

def compute_statistics(
    score: np.ndarray,
    classes: np.ndarray,
    thresholds: dict,
    gdf: gpd.GeoDataFrame,
    weights: dict,
    snow_factor: float,
    profile: dict,
) -> dict:
    """Assemble a stats dictionary for stats.json."""
    valid_score = score[np.isfinite(score)].flatten()
    n_total     = int(np.sum(classes > 0))

    # Pixel area in m²
    t   = profile["transform"]
    pxa = abs(t.a) * abs(t.e)   # m² if projected

    class_stats = {}
    for cid, label in CLASS_LABELS.items():
        n_px = int(np.sum(classes == cid))
        pct  = round(n_px / n_total * 100, 2) if n_total else 0.0
        area_km2 = (
            float(gdf.loc[gdf["class_id"] == cid, "area_km2"].values[0])
            if not gdf.empty and (gdf["class_id"] == cid).any() else 0.0
        )
        class_stats[label] = {
            "class_id": cid,
            "n_pixels": n_px,
            "pct": pct,
            "area_km2": area_km2,
        }

    return {
        "model": {
            "weights": weights,
            "snow_factor": round(snow_factor, 4),
            "thresholds": thresholds,
            "slope_band_deg": [SLOPE_BAND_LOW, SLOPE_BAND_HIGH],
        },
        "score": {
            "min":    round(float(np.nanmin(valid_score)), 4),
            "max":    round(float(np.nanmax(valid_score)), 4),
            "mean":   round(float(np.nanmean(valid_score)), 4),
            "median": round(float(np.nanmedian(valid_score)), 4),
            "std":    round(float(np.nanstd(valid_score)), 4),
        },
        "classes": class_stats,
        "raster": {
            "crs":    str(profile.get("crs", "unknown")),
            "width":  profile.get("width"),
            "height": profile.get("height"),
            "pixel_area_m2": round(pxa, 2),
        },
    }


def save_statistics(stats: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  → {out_path}")


# ════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def run_pipeline(
    base: Path,
    snow_json: Path,
    weights: dict = DEFAULT_WEIGHTS,
    snow_lapse: bool = False,
    lapse_per_100m: float = 0.06,
    max_gain_elev_m: float = 3000.0,
    verbose: bool = True,
) -> dict:
    """
    Full composite pipeline. Returns the stats dictionary.

    Parameters
    ----------
    base : Path
        Project root directory.
    snow_json : Path
        JSON produced by 03_snow.py  (save_snow_stats).
    weights : dict
        Factor weights summing to 1.0.
    verbose : bool
        Print progress.
    """
    def log(msg):
        if verbose:
            print(msg)

    paths = build_paths(base)

    # ── 1. Load rasters ─────────────────────────────────────────────
    log("\n[1/5] Loading rasters...")
    slope_arr, profile = read_raster(paths["slope"])
    aspect_arr, _      = read_raster(paths["aspect"])
    curv_arr,   _      = read_raster(paths["curv"])
    log(f"  Grid: {slope_arr.shape[1]}×{slope_arr.shape[0]} px")

    # ── 2. Load snow factor (scalar, or elevation-lapsed raster) ────
    log("[2/5] Loading snow factor...")
    if snow_lapse:
        dtm_arr, _ = read_raster(paths["dtm"])
        base_stats = load_snow_stats(snow_json)
        snow_factor = snow_factor_lapse_array(
            base_stats, dtm_arr,
            lapse_per_100m=lapse_per_100m,
            max_gain_elev_m=max_gain_elev_m,
        )
        log(f"  Snow lapse ON (+{lapse_per_100m*100:.0f}%/100m, cap {max_gain_elev_m:.0f}m): "
            f"factor {np.nanmin(snow_factor):.3f}-{np.nanmax(snow_factor):.3f} "
            f"(mean {np.nanmean(snow_factor):.3f})")
    else:
        snow_factor = load_snow_factor(snow_json)

    # ── 3. Normalise ────────────────────────────────────────────────
    log("[3/5] Normalising factors...")
    slope_norm  = normalise_slope(slope_arr)
    aspect_norm = aspect_shadow_weight(aspect_arr)
    curv_norm   = normalise_curvature(curv_arr)

    log(f"  slope  norm  → mean={np.nanmean(slope_norm):.3f}  max={np.nanmax(slope_norm):.3f}")
    log(f"  aspect norm  → mean={np.nanmean(aspect_norm):.3f}  max={np.nanmax(aspect_norm):.3f}")
    log(f"  curv   norm  → mean={np.nanmean(curv_norm):.3f}  max={np.nanmax(curv_norm):.3f}")
    snow_report = (float(np.nanmean(snow_factor))
                   if isinstance(snow_factor, np.ndarray) else float(snow_factor))
    log(f"  snow   {'mean' if isinstance(snow_factor, np.ndarray) else 'scalar'}= {snow_report:.4f}")

    # ── 4. Composite score ──────────────────────────────────────────
    log("[4/5] Computing composite score...")
    wind_load = None
    if weights.get("wind", 0.0):
        wind_load = compute_wind_load(slope_norm, aspect_norm, curv_arr, curv_norm)
        log(f"  wind load    → mean={np.nanmean(wind_load):.3f}  max={np.nanmax(wind_load):.3f}  (w={weights['wind']:.2f})")
    score   = compute_score(slope_norm, aspect_norm, curv_norm, snow_factor, weights, wind_load)
    classes, thresholds = classify(score)

    log(f"  Score range: {np.nanmin(score):.4f} - {np.nanmax(score):.4f}")
    log(f"  Thresholds (q25/q50/q75): "
        f"{thresholds['q25']:.4f} / {thresholds['q50']:.4f} / {thresholds['q75']:.4f}")

    # Write intermediate score raster
    write_raster(score, profile, paths["susc_score"], dtype="float32")

    # Write classified raster (nodata = -9999, dtype int16)
    prof_int = profile.copy()
    prof_int.update(dtype="int16", nodata=-9999)
    write_raster(classes.astype(np.float32), prof_int, paths["susc_class"], dtype="int16")

    # ── 5. Vectorise + stats ────────────────────────────────────────
    log("[5/5] Vectorising and computing statistics...")
    gdf   = vectorise(classes, profile, paths["zones_gpkg"])
    stats = compute_statistics(
        score, classes, thresholds, gdf, weights, snow_report, profile
    )
    save_statistics(stats, paths["stats_json"])

    # Summary
    log("\n── Susceptibility summary ──────────────────────────────")
    for label, s in stats["classes"].items():
        bar = "█" * int(s["pct"] / 5)
        log(f"  {label:<12} {s['pct']:5.1f}%  {bar}  ({s['area_km2']:.2f} km²)")
    log("")

    return stats


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="SwissSnow · composite susceptibility model"
    )
    parser.add_argument(
        "--base", type=str, default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--snow-json", type=str,
        default="data/processed/snow_stats.json",
        help="Path to snow_stats.json produced by 03_snow.py"
    )
    parser.add_argument(
        "--w-slope",  type=float, default=DEFAULT_WEIGHTS["slope"],
        help=f"Weight for slope  (default: {DEFAULT_WEIGHTS['slope']})"
    )
    parser.add_argument(
        "--w-snow",   type=float, default=DEFAULT_WEIGHTS["snow"],
        help=f"Weight for snow   (default: {DEFAULT_WEIGHTS['snow']})"
    )
    parser.add_argument(
        "--w-aspect", type=float, default=DEFAULT_WEIGHTS["aspect"],
        help=f"Weight for aspect (default: {DEFAULT_WEIGHTS['aspect']})"
    )
    parser.add_argument(
        "--w-curv",   type=float, default=DEFAULT_WEIGHTS["curv"],
        help=f"Weight for curv   (default: {DEFAULT_WEIGHTS['curv']})"
    )
    parser.add_argument(
        "--w-wind",   type=float, default=DEFAULT_WEIGHTS["wind"],
        help=f"Weight for wind-loading proxy (default: {DEFAULT_WEIGHTS['wind']}, off)"
    )
    parser.add_argument(
        "--snow-lapse", action="store_true",
        help="Apply an elevation snow-lapse gradient instead of a uniform scalar"
    )
    parser.add_argument(
        "--lapse-per-100m", type=float, default=0.06,
        help="Fractional snow-depth change per 100 m of elevation (default: 0.06)"
    )
    parser.add_argument(
        "--max-gain-elev", type=float, default=3000.0,
        help="Elevation (m) above which the snow gain stops increasing (default: 3000)"
    )
    args = parser.parse_args()

    weights = {
        "slope":  args.w_slope,
        "snow":   args.w_snow,
        "aspect": args.w_aspect,
        "curv":   args.w_curv,
        "wind":   args.w_wind,
    }
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        print(f"⚠ Weights sum to {total:.4f}, not 1.0 - normalising automatically.")
        weights = {k: v / total for k, v in weights.items()}

    run_pipeline(
        base=Path(args.base),
        snow_json=Path(args.snow_json),
        weights=weights,
        snow_lapse=args.snow_lapse,
        lapse_per_100m=args.lapse_per_100m,
        max_gain_elev_m=args.max_gain_elev,
    )


if __name__ == "__main__":
    _cli()
