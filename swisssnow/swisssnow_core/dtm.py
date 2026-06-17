"""
========================================================================
  SWISSSNOW · 01_dtm.py
  DTM acquisition and preparation
  ----------------------------------------------------------------------
  Downloads swissALTI3D 2m tiles from swisstopo STAC API, mosaics them,
  clips to the AOI, and reprojects to EPSG:2056 (CH1903+ / LV95).

  Output:
      data/processed/rasters/dtm_clip.tif   (float32, EPSG:2056, 2m)

  Usage:
      python 01_dtm.py --aoi data/aoi.gpkg
      python 01_dtm.py --aoi data/aoi.gpkg --res 10 --keep-tiles

  Requirements:
      pip install rasterio geopandas requests shapely numpy
      (pystac-client is optional - speeds up tile discovery)
========================================================================
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box, mapping

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════

# swisstopo STAC API - swissALTI3D collection
STAC_API    = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION  = "ch.swisstopo.swissalti3d"

# Tile index CSV (1 km grid, EPSG:2056)
# Alternative if STAC is unavailable: swisstopo WCS endpoint
WCS_URL     = "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/product"

TARGET_CRS  = CRS.from_epsg(2056)   # CH1903+ / LV95

# Default output resolution (m) - 2 m native, 10 m for quick tests
DEFAULT_RES = 2


# ════════════════════════════════════════════════════════════════════
#  PATHS
# ════════════════════════════════════════════════════════════════════

def build_paths(base: Path) -> dict:
    r = base / "data"
    return {
        "tiles_dir":  r / "raw" / "dtm" / "tiles",
        "mosaic":     r / "raw" / "dtm" / "mosaic.tif",
        "dtm_clip":   r / "processed" / "rasters" / "dtm_clip.tif",
    }


# ════════════════════════════════════════════════════════════════════
#  AOI LOADING
# ════════════════════════════════════════════════════════════════════

def load_aoi(path: Path, target_crs: CRS = TARGET_CRS) -> gpd.GeoDataFrame:
    """
    Load the AOI GeoPackage/Shapefile and reproject to target CRS.
    Dissolves all features into a single geometry (union).
    """
    if not path.exists():
        raise FileNotFoundError(f"AOI not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("AOI has no CRS - set it before running the pipeline.")
    gdf = gdf.to_crs(target_crs)
    # unary_union (not dissolve()) for compatibility with old geopandas in QGIS.
    union = gdf.geometry.unary_union
    print(f"  AOI loaded: {path.name}  bounds={union.bounds}")
    return gpd.GeoDataFrame(geometry=[union], crs=target_crs)


# ════════════════════════════════════════════════════════════════════
#  TILE DISCOVERY  (swisstopo STAC API)
# ════════════════════════════════════════════════════════════════════

def _select_tile_asset(assets: dict, target_res: float = DEFAULT_RES) -> Optional[str]:
    """
    Pick the GeoTIFF href for the requested resolution in EPSG:2056.

    swissALTI3D asset keys are filenames such as
    ``swissalti3d_2019_2776-1180_2_2056_5728.tif`` - the fields are
    ``<product>_<year>_<tile>_<res>_<crs>_<edge>.<ext>``. We match the
    resolution and CRS tokens (``_2_2056_`` for the 2 m product) and require
    a ``.tif`` extension to skip the ``.xyz.zip`` point-cloud variant.
    """
    res_token = "0.5" if target_res < 1 else "2"
    for key, asset in assets.items():
        href = asset.get("href", "")
        if href.endswith(".tif") and f"_{res_token}_2056_" in href:
            return href
    return None


def _stac_search(aoi_wgs84_geom, target_res: float = DEFAULT_RES,
                 page_size: int = 100, max_pages: int = 50) -> list[str]:
    """
    Query the swisstopo STAC API and return download URLs for all
    swissALTI3D tiles intersecting the AOI (in WGS84), following pagination.
    """
    bbox = aoi_wgs84_geom.bounds   # (minx, miny, maxx, maxy)
    url  = f"{STAC_API}/collections/{COLLECTION}/items"
    params = {
        "bbox":  f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": page_size,
    }

    urls: list[str] = []
    for _ in range(max_pages):
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("features", []):
            href = _select_tile_asset(item.get("assets", {}), target_res)
            if href:
                urls.append(href)

        # Follow the STAC 'next' link if present (href is a full URL).
        next_link = next((l for l in data.get("links", [])
                          if l.get("rel") == "next"), None)
        if not next_link:
            break
        url, params = next_link["href"], None

    print(f"  STAC found {len(urls)} tile(s)")
    return urls


def discover_tiles(aoi_gdf: gpd.GeoDataFrame,
                   target_res: float = DEFAULT_RES) -> list[str]:
    """
    Convert AOI to WGS84, query STAC, return tile URLs.
    Falls back to a manual bounding-box message if STAC is unavailable.
    """
    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")
    geom_wgs84 = aoi_wgs84.geometry.iloc[0]

    try:
        urls = _stac_search(geom_wgs84, target_res=target_res)
        if not urls:
            _manual_download_hint(geom_wgs84)
        return urls
    except Exception as e:
        print(f"  ⚠ STAC query failed ({e}) - see manual download hint below.")
        _manual_download_hint(geom_wgs84)
        return []


def _manual_download_hint(geom_wgs84) -> None:
    b = geom_wgs84.bounds
    print(
        f"\n  Manual download:\n"
        f"  https://www.swisstopo.admin.ch/en/geodata/height/alti3d.html\n"
        f"  Bounding box (WGS84): {b[0]:.5f},{b[1]:.5f},{b[2]:.5f},{b[3]:.5f}\n"
        f"  Or use the swisstopo map viewer:\n"
        f"  https://map.geo.admin.ch/?topic=ech&lang=en"
        f"&bgLayer=ch.swisstopo.pixelkarte-farbe\n"
    )


# ════════════════════════════════════════════════════════════════════
#  TILE DOWNLOAD
# ════════════════════════════════════════════════════════════════════

def download_tiles(urls: list[str], out_dir: Path) -> list[Path]:
    """Download tile GeoTIFFs; skip files already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls, 1):
        fname = out_dir / Path(url).name
        if fname.exists():
            print(f"  [{i}/{len(urls)}] cached  {fname.name}")
        else:
            print(f"  [{i}/{len(urls)}] downloading {fname.name}...", end=" ")
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            print("done")
        paths.append(fname)
    return paths


# ════════════════════════════════════════════════════════════════════
#  MOSAIC
# ════════════════════════════════════════════════════════════════════

def mosaic_tiles(tile_paths: list[Path], out_path: Path) -> None:
    """Merge all tiles into a single GeoTIFF using rasterio.merge."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = rio_merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    for src in srcs:
        src.close()
    print(f"  → mosaic: {out_path}  ({mosaic.shape[2]}×{mosaic.shape[1]} px)")


# ════════════════════════════════════════════════════════════════════
#  CLIP + REPROJECT
# ════════════════════════════════════════════════════════════════════

def clip_and_reproject(
    src_path: Path,
    aoi_gdf: gpd.GeoDataFrame,
    out_path: Path,
    target_crs: CRS = TARGET_CRS,
    target_res: float = DEFAULT_RES,
) -> None:
    """
    Clip the mosaic to the AOI and reproject to target_crs at target_res.

    Steps:
      1. Reproject AOI to source raster CRS for accurate masking
      2. Apply rio_mask (clip to AOI boundary)
      3. Reproject clipped raster to target CRS at target resolution
      4. Write LZW-compressed COG-compatible GeoTIFF
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_crs = src.crs

        # Step 1: reproject AOI to source CRS
        aoi_src = aoi_gdf.to_crs(src_crs)
        geoms   = [mapping(aoi_src.geometry.iloc[0])]

        # Step 2: clip
        clipped, clip_transform = rio_mask(src, geoms, crop=True, nodata=np.nan)
        clip_profile = src.profile.copy()
        clip_profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clip_transform,
            nodata=np.nan,
            dtype="float32",
        )

        # Step 3: reproject to target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, target_crs,
            clip_profile["width"], clip_profile["height"],
            *rasterio.transform.array_bounds(
                clip_profile["height"], clip_profile["width"], clip_transform
            ),
            resolution=target_res,
        )

        out_profile = clip_profile.copy()
        out_profile.update(
            crs=target_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            driver="GTiff",
        )

        out_arr = np.full(
            (1, dst_height, dst_width), np.nan, dtype=np.float32
        )

        # Step 4: reproject
        reproject(
            source=clipped.astype(np.float32),
            destination=out_arr,
            src_transform=clip_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out_arr)

    print(
        f"  → dtm_clip: {out_path}\n"
        f"     CRS={target_crs.to_epsg()}  res={target_res} m  "
        f"size={dst_width}×{dst_height} px"
    )


# ════════════════════════════════════════════════════════════════════
#  QUICK STATS
# ════════════════════════════════════════════════════════════════════

def print_dtm_stats(path: Path) -> None:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    valid = arr[np.isfinite(arr)]
    print(
        f"\n── DTM statistics ──────────────────────────────────\n"
        f"  Min elevation:  {np.min(valid):8.1f} m\n"
        f"  Max elevation:  {np.max(valid):8.1f} m\n"
        f"  Mean elevation: {np.mean(valid):8.1f} m\n"
        f"  Std deviation:  {np.std(valid):8.1f} m\n"
        f"  Valid pixels:   {len(valid):,}\n"
    )


# ════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def run_pipeline(
    base: Path,
    aoi_path: Path,
    target_res: float = DEFAULT_RES,
    keep_tiles: bool = False,
    tiles_dir: Path = None,
    max_tiles: int = 400,
    verbose: bool = True,
) -> Path:
    """
    Full DTM preparation pipeline.

    Parameters
    ----------
    tiles_dir : Path, optional
        Where to store downloaded tiles. Pass a persistent shared folder to
        cache tiles across runs (they are not deleted in that case).
    max_tiles : int
        Refuse if the AOI needs more than this many tiles, to avoid an
        accidental multi-GB download. Raise the limit deliberately for big runs.

    Returns
    -------
    Path to dtm_clip.tif
    """
    def log(m):
        if verbose:
            print(m)

    paths = build_paths(base)
    tiles_dir = Path(tiles_dir) if tiles_dir is not None else paths["tiles_dir"]
    tiles_dir.mkdir(parents=True, exist_ok=True)
    # A caller-supplied (shared) cache is never auto-deleted.
    keep = keep_tiles or (tiles_dir != paths["tiles_dir"])

    log("\n[1/4] Loading AOI...")
    aoi_gdf = load_aoi(aoi_path)

    log("[2/4] Discovering tiles...")
    urls = discover_tiles(aoi_gdf, target_res=target_res)

    if urls and len(urls) > max_tiles:
        raise RuntimeError(
            f"Area of interest needs {len(urls)} swissALTI3D tiles "
            f"(limit {max_tiles}). Choose a smaller area, use a coarser "
            f"resolution, or raise the tile limit deliberately.")

    if not urls:
        manual = list(tiles_dir.glob("*.tif"))
        if manual:
            log(f"  Using {len(manual)} manually placed tile(s) in {tiles_dir}")
            tile_paths = manual
        else:
            raise RuntimeError(
                "No swissALTI3D tiles found for this area of interest. "
                "Is it inside Switzerland? (The DTM only covers Swiss territory.) "
                f"Alternatively place tiles manually in {tiles_dir}.")
    else:
        log("[3/4] Downloading tiles...")
        tile_paths = download_tiles(urls, tiles_dir)

    log("[3/4] Mosaicking tiles...")
    mosaic_path = paths["mosaic"]
    mosaic_tiles(tile_paths, mosaic_path)

    log("[4/4] Clipping and reprojecting...")
    clip_and_reproject(
        src_path=mosaic_path,
        aoi_gdf=aoi_gdf,
        out_path=paths["dtm_clip"],
        target_crs=TARGET_CRS,
        target_res=target_res,
    )

    if verbose:
        print_dtm_stats(paths["dtm_clip"])

    if not keep and urls:
        log("  Cleaning up raw tiles... ")
        for p in tile_paths:
            p.unlink(missing_ok=True)

    return paths["dtm_clip"]


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="SwissSnow · DTM preparation (swissALTI3D)"
    )
    parser.add_argument(
        "--aoi", type=str, required=True,
        help="Path to AOI GeoPackage or Shapefile"
    )
    parser.add_argument(
        "--base", type=str, default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--res", type=float, default=DEFAULT_RES,
        help=f"Output resolution in metres (default: {DEFAULT_RES})"
    )
    parser.add_argument(
        "--keep-tiles", action="store_true",
        help="Keep raw downloaded tiles (default: delete after mosaic)"
    )
    args = parser.parse_args()

    run_pipeline(
        base=Path(args.base),
        aoi_path=Path(args.aoi),
        target_res=args.res,
        keep_tiles=args.keep_tiles,
    )


if __name__ == "__main__":
    _cli()
