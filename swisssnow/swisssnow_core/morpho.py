"""
========================================================================
  SWISSSNOW · 02_morpho.py
  Morphological factor computation
  ----------------------------------------------------------------------
  Reads dtm_clip.tif produced by 01_dtm.py and computes:

    • slope.tif          - steepness in degrees
    • aspect.tif         - direction in degrees (0=N, clockwise)
    • curvature.tif      - profile curvature (convex positive)

  Uses WhiteboxTools (wbt) as primary engine.
  Falls back to a pure numpy/rasterio implementation if wbt is not
  available (lower accuracy, suitable for testing).

  Output:
      data/processed/rasters/slope.tif
      data/processed/rasters/aspect.tif
      data/processed/rasters/curvature.tif

  Usage:
      python 02_morpho.py
      python 02_morpho.py --base /path/to/project --no-wbt
========================================================================
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

warnings.filterwarnings("ignore")

# ── WhiteboxTools optional import ──────────────────────────────────
try:
    from whitebox import WhiteboxTools
    _WBT_AVAILABLE = True
except ImportError:
    _WBT_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
#  PATHS
# ════════════════════════════════════════════════════════════════════

def build_paths(base: Path) -> dict:
    r = base / "data" / "processed" / "rasters"
    return {
        "dtm":       r / "dtm_clip.tif",
        "slope":     r / "slope.tif",
        "aspect":    r / "aspect.tif",
        "curvature": r / "curvature.tif",
    }


# ════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ════════════════════════════════════════════════════════════════════

def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as src:
        arr  = src.read(1).astype(np.float32)
        prof = src.profile.copy()
        nd   = src.nodata
        res  = src.res   # (pixel_width, pixel_height) in CRS units
    if nd is not None:
        arr[arr == nd] = np.nan
    prof["res"] = res
    return arr, prof


def write_raster(arr: np.ndarray, profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {k: v for k, v in profile.items() if k != "res"}
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="lzw",
                tiled=True, blockxsize=256, blockysize=256, driver="GTiff")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════════════
#  WHITEBOX ENGINE
# ════════════════════════════════════════════════════════════════════

def _init_wbt(base: Path, verbose: bool = False) -> "WhiteboxTools":
    wbt = WhiteboxTools()
    # WhiteboxTools runs from its own install directory, so a relative working
    # dir is resolved against the wrong location ("No such file or directory").
    wbt.set_working_dir(str((base / "data" / "processed" / "rasters").resolve()))
    wbt.verbose = verbose
    return wbt


def compute_morpho_wbt(paths: dict, base: Path, verbose: bool = True) -> None:
    """
    Compute slope, aspect, and profile curvature using WhiteboxTools.

    WhiteboxTools expects relative filenames within the working directory.
    All files must be in the same folder (data/processed/rasters/).
    """
    def log(msg):
        if verbose:
            print(msg)

    wbt = _init_wbt(base)
    dtm_name  = paths["dtm"].name

    log("  Computing slope...")
    wbt.slope(
        dem=dtm_name,
        output=paths["slope"].name,
        units="degrees",
        zfactor=None,
    )

    log("  Computing aspect...")
    wbt.aspect(
        dem=dtm_name,
        output=paths["aspect"].name,
    )

    log("  Computing profile curvature...")
    wbt.profile_curvature(
        dem=dtm_name,
        output=paths["curvature"].name,
        log=False,
        zfactor=None,
    )


# ════════════════════════════════════════════════════════════════════
#  NUMPY FALLBACK ENGINE
# ════════════════════════════════════════════════════════════════════

def _gradient_2d(arr: np.ndarray, res_x: float, res_y: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Central-difference gradient of a 2-D array.
    Returns (dz_dx, dz_dy) in units of [elevation unit / metre].
    """
    dz_dy = np.gradient(arr, res_y, axis=0)   # N-S gradient
    dz_dx = np.gradient(arr, res_x, axis=1)   # E-W gradient
    return dz_dx, dz_dy


def compute_slope_numpy(arr: np.ndarray, res_x: float, res_y: float
                        ) -> np.ndarray:
    """
    Slope in degrees using the Horn (1981) 3×3 neighbourhood algorithm.
    For uniform resolution grids, central differences give an equivalent
    result with much less code.
    """
    dz_dx, dz_dy = _gradient_2d(arr, res_x, res_y)
    rise = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_deg = np.degrees(np.arctan(rise))
    slope_deg[~np.isfinite(arr)] = np.nan
    return slope_deg.astype(np.float32)


def compute_aspect_numpy(arr: np.ndarray, res_x: float, res_y: float
                         ) -> np.ndarray:
    """
    Aspect in degrees, 0 = North, clockwise (ESRI / WhiteboxTools convention).
    Flat areas receive -1 (standard nodata for aspect).
    """
    dz_dx, dz_dy = _gradient_2d(arr, res_x, res_y)

    # Mathematic angle → geographic bearing
    # atan2 returns angle from E axis, CCW; convert to N=0, CW
    aspect_math = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect_geo  = 90.0 - aspect_math
    aspect_geo  = aspect_geo % 360.0

    # Flat pixels (no gradient)
    flat = (dz_dx == 0) & (dz_dy == 0)
    aspect_geo[flat] = -1.0
    aspect_geo[~np.isfinite(arr)] = np.nan
    return aspect_geo.astype(np.float32)


def compute_curvature_numpy(arr: np.ndarray, res_x: float, res_y: float
                             ) -> np.ndarray:
    """
    Profile curvature (concavity/convexity in the direction of steepest
    descent). Positive = convex (unstable for avalanche release).

    Formula after Zevenbergen & Thorne (1987):
        profile_curv = -2 * (D*G² + E*H² + F*G*H) / (G² + H²)
    where D, E, F are second-order partial derivatives and G, H are
    first-order. For flat areas the denominator ≈ 0 → returns 0.
    """
    dx, dy = res_x, res_y

    # Second-order partial derivatives via Sobel-like stencil
    # Use np.gradient twice for simplicity
    dz_dx, dz_dy  = _gradient_2d(arr, dx, dy)
    d2z_dx2, _    = _gradient_2d(dz_dx, dx, dy)
    _, d2z_dy2    = _gradient_2d(dz_dy, dx, dy)
    d2z_dxdy, _   = _gradient_2d(dz_dx, dx, dy)  # cross term approximation

    G = dz_dx   # p
    H = dz_dy   # q
    D = d2z_dx2 # r
    E = d2z_dy2 # t
    F = d2z_dxdy

    denom = G ** 2 + H ** 2
    denom_safe = np.where(denom < 1e-10, 1e-10, denom)

    curv = -2.0 * (D * G ** 2 + E * H ** 2 + F * G * H) / denom_safe
    curv[denom < 1e-10] = 0.0   # flat areas
    curv[~np.isfinite(arr)] = np.nan
    return curv.astype(np.float32)


def compute_morpho_numpy(paths: dict, verbose: bool = True) -> None:
    """Pure numpy fallback - no WhiteboxTools required."""
    def log(m):
        if verbose:
            print(m)
    arr, prof = read_raster(paths["dtm"])
    res_x, res_y = prof["res"]

    log("  Computing slope (numpy)...")
    write_raster(compute_slope_numpy(arr, res_x, res_y), prof, paths["slope"])

    log("  Computing aspect (numpy)...")
    write_raster(compute_aspect_numpy(arr, res_x, res_y), prof, paths["aspect"])

    log("  Computing curvature (numpy)...")
    write_raster(compute_curvature_numpy(arr, res_x, res_y), prof, paths["curvature"])


def compute_morpho_gdal(paths: dict, verbose: bool = True) -> None:
    """
    GDAL-native engine, preferred inside QGIS (osgeo.gdal is always available
    there, no external binary needed). Slope and aspect use gdaldem; profile
    curvature is not provided by gdaldem, so it falls back to the numpy
    Zevenbergen & Thorne implementation. Output conventions (0=N clockwise
    aspect, degrees slope) match the rest of the pipeline.
    """
    from osgeo import gdal   # local import: only needed for this engine
    gdal.UseExceptions()

    def log(m):
        if verbose:
            print(m)

    dtm = str(paths["dtm"])

    log("  Computing slope (GDAL)...")
    gdal.DEMProcessing(str(paths["slope"]), dtm, "slope",
                       slopeFormat="degree", computeEdges=True)

    log("  Computing aspect (GDAL)...")
    gdal.DEMProcessing(str(paths["aspect"]), dtm, "aspect",
                       zeroForFlat=True, computeEdges=True)

    log("  Computing curvature (numpy)...")
    arr, prof = read_raster(paths["dtm"])
    res_x, res_y = prof["res"]
    write_raster(compute_curvature_numpy(arr, res_x, res_y), prof, paths["curvature"])


# ════════════════════════════════════════════════════════════════════
#  QUALITY CHECK
# ════════════════════════════════════════════════════════════════════

def print_stats(paths: dict) -> None:
    """Print a summary table of the three output rasters."""
    rows = []
    for name, path in [
        ("slope (°)",  paths["slope"]),
        ("aspect (°)", paths["aspect"]),
        ("curvature",  paths["curvature"]),
    ]:
        if not path.exists():
            continue
        arr, _ = read_raster(path)
        valid  = arr[np.isfinite(arr)].flatten()
        rows.append((name, valid.min(), valid.mean(), valid.max(), valid.std()))

    print("\n── Morphological rasters ────────────────────────────")
    print(f"  {'Factor':<14} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, mn, me, mx, sd in rows:
        print(f"  {name:<14} {mn:8.2f} {me:8.2f} {mx:8.2f} {sd:8.2f}")
    print()

    # Avalanche-relevant slope stats
    arr_slope, _ = read_raster(paths["slope"])
    valid_slope  = arr_slope[np.isfinite(arr_slope)].flatten()
    n_total = len(valid_slope)
    for lo, hi, label in [(0, 25, "<25°"), (25, 38, "25-38°"), (38, 55, "38-55°"), (55, 90, ">55°")]:
        n = int(np.sum((valid_slope >= lo) & (valid_slope < hi)))
        pct = n / n_total * 100
        bar = "█" * int(pct / 3)
        print(f"  Slope {label:<8} {pct:5.1f}%  {bar}")
    print()


# ════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def run_pipeline(
    base: Path,
    force_numpy: bool = False,
    engine: str = "auto",
    verbose: bool = True,
) -> None:
    """
    Compute slope, aspect, and curvature from dtm_clip.tif.

    Parameters
    ----------
    base : Path
        Project root.
    force_numpy : bool
        Skip WhiteboxTools even if available.
    verbose : bool
        Print progress.
    """
    def log(msg):
        if verbose:
            print(msg)

    paths = build_paths(base)

    if not paths["dtm"].exists():
        # Raise (never sys.exit) - this runs inside QGIS / the web app.
        raise FileNotFoundError(
            f"DTM not found: {paths['dtm']} - run the DTM stage first.")

    # Check if outputs already exist
    existing = all(paths[k].exists() for k in ("slope", "aspect", "curvature"))
    if existing:
        log("  All morphological rasters already exist - skipping computation.")
        log("  Use --force to recompute.\n")
        if verbose:
            print_stats(paths)
        return

    # Resolve the engine. "auto" keeps the historical behaviour (wbt if present,
    # else numpy); force_numpy is honoured for backward compatibility.
    eng = engine
    if force_numpy:
        eng = "numpy"
    if eng == "auto":
        eng = "wbt" if _WBT_AVAILABLE else "numpy"

    if eng == "wbt":
        if not _WBT_AVAILABLE:
            raise RuntimeError("engine='wbt' but WhiteboxTools is not installed.")
        log("\n[morpho] Using WhiteboxTools engine")
        compute_morpho_wbt(paths, base, verbose=verbose)
    elif eng == "gdal":
        log("\n[morpho] Using GDAL engine (slope/aspect) + numpy curvature")
        compute_morpho_gdal(paths, verbose=verbose)
    elif eng == "numpy":
        log("\n[morpho] Using numpy engine")
        compute_morpho_numpy(paths, verbose=verbose)
    else:
        raise ValueError(f"Unknown morphology engine: {engine!r}")

    if verbose:
        print_stats(paths)


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="SwissSnow · morphological factors (slope / aspect / curvature)"
    )
    parser.add_argument(
        "--base", type=str, default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--no-wbt", action="store_true",
        help="Force numpy fallback even if WhiteboxTools is installed"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if output rasters already exist"
    )
    args = parser.parse_args()

    base = Path(args.base)
    paths = build_paths(base)

    if args.force:
        for k in ("slope", "aspect", "curvature"):
            paths[k].unlink(missing_ok=True)

    run_pipeline(base=base, force_numpy=args.no_wbt)


if __name__ == "__main__":
    _cli()
