#!/usr/bin/env python3
"""
SwissAvalanche · 05_validate.py

Validate the susceptibility map against observed avalanche incidents.

Susceptibility is not hazard: this module does not turn the model into a hazard
map, but it quantifies how well the susceptibility classes discriminate real
events. Given a point layer of avalanche incidents, it samples the
susceptibility class/score at each incident and reports:

  - incident count and share per susceptibility class
  - the area share per class (from the classified raster)
  - the frequency ratio  FR = (% incidents in class) / (% area in class)
        FR > 1  → class is over-represented among incidents (good discrimination)
        FR ≈ 1  → no better than random
  - the capture rate of the High + Very High classes
  - a success-rate / AUC-style score: fraction of incidents falling in the
    top fraction of the score, compared to the area that fraction covers.

Incident data source
--------------------
SLF publishes avalanche-accident data for Switzerland (EnviDat / opendata.swiss).
Download it manually and pass any GeoPandas-readable point file (GPKG, SHP,
GeoJSON) or a CSV with longitude/latitude columns via --incidents. The layer is
reprojected to the raster CRS (EPSG:2056) automatically.

Usage
-----
    python pipeline/05_validate.py --incidents data/raw/slf_incidents.gpkg
    python pipeline/05_validate.py --incidents data/raw/accidents.csv \
        --lon-col lon --lat-col lat --out data/processed/validation.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd

warnings.filterwarnings("ignore")

TARGET_CRS = "EPSG:2056"
CLASS_LABELS = {1: "Low", 2: "Moderate", 3: "High", 4: "Very high"}

# SLF avalanche-accident dataset on EnviDat (≈4700 accidents since 1970/71,
# with WGS84 start-zone coordinates). Public, no key. CSV has 3 banner lines
# before the real header.
SLF_CSV_URL = (
    "https://www.envidat.ch/dataset/aa035efb-630a-4b7f-a406-f7a579a74de9/"
    "resource/944beac1-11d1-4c84-9bd9-683d12a3c581/download/"
    "version2_avalanche_accidents_all_switzerland_since_1970.csv"
)
SLF_LAT_COL = "start.zone.coordinates.latitude"
SLF_LON_COL = "start.zone.coordinates.longitude"


def fetch_slf_incidents(
    bbox_wgs84: tuple = None,
    start: str = None,
    end: str = None,
    cache_dir: Path = None,
    refresh: bool = False,
    target_crs: str = TARGET_CRS,
) -> "gpd.GeoDataFrame":
    """
    Download the SLF avalanche-accident dataset and return it as points.

    Parameters
    ----------
    bbox_wgs84 : (minx, miny, maxx, maxy), optional
        Keep only accidents inside this WGS84 bounding box (e.g. the AOI).
    start, end : str, optional
        Keep only accidents with date in [start, end] (ISO YYYY-MM-DD).
    cache_dir : Path, optional
        Where to cache the CSV (default ~/.swissavalanche). Re-used unless `refresh`.
    refresh : bool
        Force a fresh download even if a cached copy exists.

    Returns
    -------
    GeoDataFrame in `target_crs` with columns avalanche_id, date, canton,
    municipality, elevation, n_dead, n_caught, activity (when present).
    """
    import pandas as pd
    import requests

    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".swissavalanche"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "slf_avalanche_accidents.csv"

    if refresh or not csv_path.exists():
        resp = requests.get(SLF_CSV_URL, timeout=60)
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)

    # The 4th line is the header, but the publisher wrapped the whole line in
    # quotes (so a naive read sees one column). Recover the column names from
    # it, then read the data rows (line 5+) with those names.
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        header_line = f.readlines()[3]
    columns = header_line.strip().strip('"').replace('""', '').split(",")
    df = pd.read_csv(csv_path, skiprows=4, header=None, names=columns)

    df = df[df[SLF_LAT_COL].notna() & df[SLF_LON_COL].notna()].copy()
    df[SLF_LAT_COL] = pd.to_numeric(df[SLF_LAT_COL], errors="coerce")
    df[SLF_LON_COL] = pd.to_numeric(df[SLF_LON_COL], errors="coerce")
    df = df.dropna(subset=[SLF_LAT_COL, SLF_LON_COL])

    if start or end:
        d = pd.to_datetime(df.get("date"), errors="coerce")
        if start:
            df = df[d >= pd.to_datetime(start)]
            d = d[d >= pd.to_datetime(start)]
        if end:
            df = df[d <= pd.to_datetime(end)]

    keep = {
        "avalanche.id": "avalanche_id", "date": "date", "canton": "canton",
        "municipality": "municipality", "start.zone.elevation": "elevation",
        "number.dead": "n_dead", "number.caught": "n_caught", "activity": "activity",
    }
    attrs = {new: df[old] for old, new in keep.items() if old in df.columns}
    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=gpd.points_from_xy(df[SLF_LON_COL], df[SLF_LAT_COL]),
        crs="EPSG:4326",
    )
    if bbox_wgs84 is not None:
        minx, miny, maxx, maxy = bbox_wgs84
        gdf = gdf.cx[minx:maxx, miny:maxy]
    return gdf.to_crs(target_crs)


# ════════════════════════════════════════════════════════════════════
#  PATHS
# ════════════════════════════════════════════════════════════════════

def build_paths(base: Path) -> dict:
    r = base / "data" / "processed" / "rasters"
    return {
        "susc_class": r / "susceptibility_class.tif",
        "susc_score": r / "susceptibility_score.tif",
        "out_points": base / "data" / "processed" / "vectors" / "incidents_classified.gpkg",
        "out_json":   base / "data" / "processed" / "validation.json",
    }


# ════════════════════════════════════════════════════════════════════
#  INCIDENT LOADING
# ════════════════════════════════════════════════════════════════════

def load_incidents(path: Path, lon_col: str = "lon", lat_col: str = "lat",
                   target_crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    """
    Load avalanche incidents as a point GeoDataFrame in the target CRS.

    Accepts any vector format readable by GeoPandas, or a CSV with longitude /
    latitude columns (assumed WGS84). Raises if the file has no usable geometry.
    """
    if not path.exists():
        raise FileNotFoundError(f"Incidents file not found: {path}")

    if path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(
                f"CSV must contain '{lon_col}' and '{lat_col}' columns "
                f"(found: {list(df.columns)}). Use --lon-col / --lat-col.")
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326")
    else:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path} has no CRS; cannot reproject reliably.")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf.to_crs(target_crs)


def sample_raster_at_points(raster_path: Path, gdf: gpd.GeoDataFrame,
                            nodata_vals=(-9999,)) -> np.ndarray:
    """
    Sample raster band 1 at each point. Returns a float array (NaN where the
    point falls outside the raster or on nodata).
    """
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        nd = src.nodata
    if nd is not None:
        nodata_vals = tuple(nodata_vals) + (nd,)
    for ndv in nodata_vals:
        vals[vals == ndv] = np.nan

    # Mark out-of-bounds points as NaN too
    xs = np.array([geom.x for geom in gdf.geometry])
    ys = np.array([geom.y for geom in gdf.geometry])
    inside = (xs >= bounds.left) & (xs <= bounds.right) & \
             (ys >= bounds.bottom) & (ys <= bounds.top)
    vals[~inside] = np.nan
    return vals


# ════════════════════════════════════════════════════════════════════
#  VALIDATION METRICS
# ════════════════════════════════════════════════════════════════════

def class_area_fractions(class_path: Path) -> dict:
    """Return {class_id: area_fraction} from the classified raster."""
    with rasterio.open(class_path) as src:
        arr = src.read(1)
        nd = src.nodata
    valid = arr[(arr > 0) & (arr != nd)] if nd is not None else arr[arr > 0]
    total = valid.size
    return {cid: float(np.sum(valid == cid) / total) if total else 0.0
            for cid in CLASS_LABELS}


def success_rate(score_path: Path, incident_scores: np.ndarray,
                 fractions=(0.1, 0.2, 0.3, 0.5)) -> list[dict]:
    """
    Success-rate curve: for each top-`f` fraction of the score (the most
    susceptible f of the area), report what share of incidents it captures.
    A skilful model captures far more than `f` of incidents in the top `f`.
    """
    with rasterio.open(score_path) as src:
        arr = src.read(1).astype(float)
        nd = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    valid = arr[np.isfinite(arr)]
    inc = incident_scores[np.isfinite(incident_scores)]
    out = []
    for f in fractions:
        thr = np.quantile(valid, 1.0 - f)        # score cutoff for top-f area
        captured = float(np.mean(inc >= thr)) if inc.size else 0.0
        out.append({"area_fraction": f, "score_threshold": round(float(thr), 4),
                    "incident_capture": round(captured, 4),
                    "lift": round(captured / f, 2) if f else 0.0})
    return out


def validate(class_path: Path, score_path: Path,
             gdf: gpd.GeoDataFrame) -> tuple[dict, gpd.GeoDataFrame]:
    """Run the full validation and return (metrics_dict, classified_points_gdf)."""
    inc_class = sample_raster_at_points(class_path, gdf)
    inc_score = sample_raster_at_points(score_path, gdf)

    gdf = gdf.copy()
    gdf["susc_class"] = inc_class
    gdf["susc_score"] = inc_score

    n_total = int(np.sum(np.isfinite(inc_class)))
    n_outside = int(np.sum(~np.isfinite(inc_class)))

    area_frac = class_area_fractions(class_path)
    per_class = {}
    for cid, label in CLASS_LABELS.items():
        n = int(np.sum(inc_class == cid))
        pct_inc = (n / n_total) if n_total else 0.0
        pct_area = area_frac[cid]
        fr = (pct_inc / pct_area) if pct_area > 0 else 0.0
        per_class[label] = {
            "class_id": cid,
            "n_incidents": n,
            "pct_incidents": round(pct_inc * 100, 2),
            "pct_area": round(pct_area * 100, 2),
            "frequency_ratio": round(fr, 3),
        }

    high_capture = sum(per_class[l]["pct_incidents"] for l in ("High", "Very high"))
    high_area    = sum(per_class[l]["pct_area"]      for l in ("High", "Very high"))

    metrics = {
        "n_incidents_total": int(len(gdf)),
        "n_incidents_inside_aoi": n_total,
        "n_incidents_outside_aoi": n_outside,
        "per_class": per_class,
        "high_vh_capture_pct": round(high_capture, 2),
        "high_vh_area_pct": round(high_area, 2),
        "high_vh_lift": round(high_capture / high_area, 2) if high_area else 0.0,
        "mean_score_at_incidents": round(float(np.nanmean(inc_score)), 4)
            if n_total else None,
        "success_rate": success_rate(score_path, inc_score),
    }
    return metrics, gdf


# ════════════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════════════

def run_pipeline(base: Path, incidents: Path, lon_col: str = "lon",
                 lat_col: str = "lat", verbose: bool = True) -> dict:
    def log(m):
        if verbose:
            print(m)

    paths = build_paths(base)
    for key in ("susc_class", "susc_score"):
        if not paths[key].exists():
            raise FileNotFoundError(
                f"{paths[key]} not found - run 04_composite.py first.")

    log(f"\n[1/3] Loading incidents from {incidents}...")
    gdf = load_incidents(incidents, lon_col, lat_col)
    log(f"  {len(gdf)} incident point(s) in {TARGET_CRS}")

    log("[2/3] Sampling susceptibility at incident locations...")
    metrics, gdf_out = validate(paths["susc_class"], paths["susc_score"], gdf)

    log("[3/3] Writing outputs...")
    paths["out_points"].parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(paths["out_points"], driver="GPKG")
    with open(paths["out_json"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"  → {paths['out_points']}")
    log(f"  → {paths['out_json']}")

    # Summary
    log("\n── Validation summary ──────────────────────────────")
    log(f"  Incidents inside AOI: {metrics['n_incidents_inside_aoi']}"
        f"  (outside: {metrics['n_incidents_outside_aoi']})")
    log(f"  {'Class':<12}{'incid%':>8}{'area%':>8}{'freq.ratio':>12}")
    for label, s in metrics["per_class"].items():
        log(f"  {label:<12}{s['pct_incidents']:>7.1f}%{s['pct_area']:>7.1f}%"
            f"{s['frequency_ratio']:>12.2f}")
    log(f"  High+VeryHigh capture: {metrics['high_vh_capture_pct']:.1f}% of incidents "
        f"on {metrics['high_vh_area_pct']:.1f}% of area  (lift {metrics['high_vh_lift']:.2f}×)")
    log("")
    return metrics


def _cli():
    parser = argparse.ArgumentParser(
        description="SwissAvalanche · validate susceptibility against avalanche incidents")
    parser.add_argument("--base", type=str, default=".",
                        help="Project root directory (default: current directory)")
    parser.add_argument("--incidents", type=str, required=True,
                        help="Path to incidents file (GPKG/SHP/GeoJSON or CSV)")
    parser.add_argument("--lon-col", type=str, default="lon",
                        help="Longitude column name for CSV input (default: lon)")
    parser.add_argument("--lat-col", type=str, default="lat",
                        help="Latitude column name for CSV input (default: lat)")
    parser.add_argument("--out", type=str, default=None,
                        help="Override path for the validation JSON")
    args = parser.parse_args()

    base = Path(args.base)
    metrics = run_pipeline(base, Path(args.incidents), args.lon_col, args.lat_col)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  → {args.out}")


if __name__ == "__main__":
    _cli()
