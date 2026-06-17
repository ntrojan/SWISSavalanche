"""
SwissAvalanche Processing algorithm.

Thin QGIS wrapper around swissavalanche_core.run_analysis: it collects parameters
from the standard Processing dialog, drives the shared engine, maps progress
onto the QGIS feedback object, and loads the result layers into the project.

AOI selection mirrors the SwissMorph plugin exactly: a polygon layer and/or a
map extent (polygon layer takes priority), resolved to a bounding box in
EPSG:2056.
"""

import os
import sys
from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsPalettedRasterRenderer,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsWkbTypes,
)
from qgis.PyQt.QtGui import QColor, QIcon

# Susceptibility palette (mirrors swissavalanche_core.composite CLASS_COLORS/LABELS).
CLASS_COLORS = {1: "#1a9850", 2: "#a6d96a", 3: "#fdae61", 4: "#d73027"}
CLASS_LABELS = {1: "Low", 2: "Moderate", 3: "High", 4: "Very high"}

# Post-processors are referenced by QGIS only weakly; keep them alive here so
# they are not garbage-collected before the layers are loaded.
_KEEP_ALIVE = []


class _ClassRasterStyler(QgsProcessingLayerPostProcessorInterface):
    """Colour the susceptibility class raster (values 1-4) as a palette."""

    def postProcessLayer(self, layer, context, feedback):
        if not isinstance(layer, QgsRasterLayer) or not layer.isValid():
            return
        classes = [
            QgsPalettedRasterRenderer.Class(cid, QColor(CLASS_COLORS[cid]),
                                            CLASS_LABELS[cid])
            for cid in (1, 2, 3, 4)
        ]
        renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes)
        layer.setRenderer(renderer)
        layer.triggerRepaint()


class _ScoreRasterStyler(QgsProcessingLayerPostProcessorInterface):
    """Colour the continuous score raster green→red over its value range."""

    def postProcessLayer(self, layer, context, feedback):
        if not isinstance(layer, QgsRasterLayer) or not layer.isValid():
            return
        st = layer.dataProvider().bandStatistics(1)
        mn, mx = st.minimumValue, st.maximumValue
        if mx <= mn:
            mx = mn + 1e-6
        ramp = ["#1a9850", "#a6d96a", "#fdae61", "#d73027"]
        items = [QgsColorRampShader.ColorRampItem(
                    mn + f * (mx - mn), QColor(c), f"{mn + f * (mx - mn):.2f}")
                 for f, c in zip((0.0, 0.33, 0.66, 1.0), ramp)]
        fcn = QgsColorRampShader(mn, mx)
        fcn.setColorRampType(QgsColorRampShader.Interpolated)
        fcn.setColorRampItemList(items)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(fcn)
        layer.setRenderer(QgsSingleBandPseudoColorRenderer(
            layer.dataProvider(), 1, shader))
        layer.triggerRepaint()


class _ZonesVectorStyler(QgsProcessingLayerPostProcessorInterface):
    """Colour the susceptibility zone polygons by class_id."""

    def postProcessLayer(self, layer, context, feedback):
        # Vector layers expose fields(); skip anything that does not.
        if not hasattr(layer, "fields") or not layer.isValid():
            return
        cats = []
        for cid in (1, 2, 3, 4):
            sym = QgsFillSymbol.createSimple({
                "color": CLASS_COLORS[cid],
                "outline_color": "#404040",
                "outline_width": "0.2",
            })
            cats.append(QgsRendererCategory(cid, sym, CLASS_LABELS[cid]))
        layer.setRenderer(QgsCategorizedSymbolRenderer("class_id", cats))
        layer.triggerRepaint()


def _ensure_core_importable():
    """Make `swissavalanche_core` importable, whether bundled or used from the repo."""
    try:
        import swissavalanche_core  # noqa: F401
        return
    except ImportError:
        pass
    # Bundled: swissavalanche_core sits next to this file (see build_plugin.py).
    # Dev: it sits one level up from qgis_plugin/.
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "swissavalanche_core" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
    raise QgsProcessingException(
        "swissavalanche_core package not found. Bundle it inside the plugin folder "
        "or install it on the QGIS Python path.")


class SwissAvalancheAlgorithm(QgsProcessingAlgorithm):
    AOI_LAYER = "AOI_LAYER"
    AOI_EXTENT = "AOI_EXTENT"
    SNOW_MODE = "SNOW_MODE"
    START = "START"
    END = "END"
    N_WINTERS = "N_WINTERS"
    WINTER_START = "WINTER_START"
    WINTER_END = "WINTER_END"
    RESOLUTION = "RESOLUTION"
    SNOW_LAPSE = "SNOW_LAPSE"
    WIND_WEIGHT = "WIND_WEIGHT"
    LOAD_FACTORS = "LOAD_FACTORS"
    LOAD_SCORE = "LOAD_SCORE"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    # Advanced
    W_SLOPE = "W_SLOPE"
    W_SNOW = "W_SNOW"
    W_ASPECT = "W_ASPECT"
    W_CURV = "W_CURV"
    LAPSE_PER_100M = "LAPSE_PER_100M"
    SNOW_THRESHOLD = "SNOW_THRESHOLD"
    MAX_TILES = "MAX_TILES"
    TILE_CACHE = "TILE_CACHE"

    def name(self):
        return "avalanche_susceptibility"

    def displayName(self):
        return "Avalanche susceptibility"

    def group(self):
        return "SwissAvalanche"

    def groupId(self):
        return "swissavalanche"

    def icon(self):
        return QIcon(
            os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        )

    def shortHelpString(self):
        return (
            "<h3>Avalanche susceptibility</h3>"
            "<p>Produces a 4-class avalanche-susceptibility map (Low → Very high) "
            "for the chosen area. It blends the <b>terrain</b> (slope, aspect, "
            "profile curvature from the 2&nbsp;m swissALTI3D DTM, downloaded "
            "automatically) with a <b>seasonal snow load</b> from the Open-Meteo "
            "historical archive.</p>"

            "<h4>Area of interest</h4>"
            "<p>Give a <b>polygon layer</b> <i>or</i> a <b>map extent</b> "
            "(if both are set, the polygon layer wins). The analysis runs on the "
            "bounding box of what you provide. Keep it small for a first test - "
            "the terrain is downloaded as 1&nbsp;km tiles, so large areas take "
            "much longer.</p>"

            "<h4>Snow basis</h4>"
            "<p>How the snow load is measured (the only time-dependent input - "
            "terrain factors never change):</p>"
            "<ul>"
            "<li><b>Climatology (recommended):</b> averages the winter snow over "
            "the last N winters (default 10), using a fixed winter window "
            "(default 1 Dec - 31 Mar, set in Advanced). This gives the "
            "<i>typical</i> snow load, the right basis for a stable "
            "susceptibility map - a single winter can be anomalous.</li>"
            "<li><b>Single winter:</b> uses one specific winter from the "
            "<i>Single-winter start/end</i> dates (past dates, end after start). "
            "Use this to study a particular season or compare a snowy vs. a dry "
            "year.</li>"
            "</ul>"
            "<p>A snowier basis raises susceptibility, a drier one lowers it; the "
            "terrain stays identical.</p>"

            "<h4>Apply elevation snow-lapse</h4>"
            "<p>When on, the snow factor is scaled with elevation across the area "
            "(more snow higher up), instead of one uniform value. Recommended for "
            "areas with large elevation range.</p>"

            "<h4>Wind-loading weight</h4>"
            "<p>Weight of an optional <b>wind-loading</b> factor that highlights "
            "convex, lee-facing slopes in the avalanche slope band - where "
            "wind-blown snow tends to accumulate into slabs.<br>"
            "<b>0 = off</b> (default). A typical value to enable it is "
            "<code>0.10-0.20</code>; the other factor weights are re-scaled "
            "automatically so everything still sums to 1. Higher values give the "
            "wind effect more influence on the final score.</p>"

            "<h4>Output folder</h4>"
            "<p>Where results are written. The susceptibility class raster and "
            "the zone polygons are added to your project automatically.</p>"

            "<p><i>Needs an internet connection (terrain tiles + snow API). "
            "Susceptibility is not hazard - not for operational safety "
            "decisions.</i></p>")

    def createInstance(self):
        return SwissAvalancheAlgorithm()

    def initAlgorithm(self, config=None):
        # ── Area of interest (same pattern as SwissMorph) ───────────────
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.AOI_LAYER,
            "Area of interest - polygon layer (takes priority)",
            types=[QgsWkbTypes.PolygonGeometry], optional=True))
        self.addParameter(QgsProcessingParameterExtent(
            self.AOI_EXTENT,
            "Area of interest - map extent (used if no polygon layer)",
            optional=True))

        # ── Snow timing ─────────────────────────────────────────────────
        self.addParameter(QgsProcessingParameterEnum(
            self.SNOW_MODE, "Snow basis",
            options=["Climatology (multi-year average - recommended)",
                     "Single winter (specific dates below)"],
            defaultValue=0))
        self.addParameter(QgsProcessingParameterString(
            self.START,
            "Single-winter start - YYYY-MM-DD (only if 'Single winter')",
            defaultValue="2023-12-01"))
        self.addParameter(QgsProcessingParameterString(
            self.END,
            "Single-winter end - YYYY-MM-DD (only if 'Single winter')",
            defaultValue="2024-03-31"))

        # ── Model options ───────────────────────────────────────────────
        self.addParameter(QgsProcessingParameterNumber(
            self.RESOLUTION, "Output resolution (m)",
            QgsProcessingParameterNumber.Double, defaultValue=10.0, minValue=2.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SNOW_LAPSE, "Apply elevation snow-lapse (more snow higher up)",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.WIND_WEIGHT,
            "Wind-loading weight (0 = off; e.g. 0.15 to enable)",
            QgsProcessingParameterNumber.Double, defaultValue=0.0,
            minValue=0.0, maxValue=0.5))

        self.addParameter(QgsProcessingParameterBoolean(
            self.LOAD_FACTORS, "Also load terrain factors (slope/aspect/curvature)",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.LOAD_SCORE, "Also load the continuous score raster",
            defaultValue=True))

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_FOLDER, "Output folder"))

        # ── Advanced parameters (collapsed by default) ──────────────────
        adv = [
            QgsProcessingParameterNumber(
                self.N_WINTERS, "Climatology: number of winters to average",
                QgsProcessingParameterNumber.Integer, defaultValue=10, minValue=1),
            QgsProcessingParameterString(
                self.WINTER_START, "Climatology: winter window start (MM-DD)",
                defaultValue="12-01"),
            QgsProcessingParameterString(
                self.WINTER_END, "Climatology: winter window end (MM-DD)",
                defaultValue="03-31"),
            QgsProcessingParameterNumber(
                self.W_SLOPE, "Weight: slope",
                QgsProcessingParameterNumber.Double, defaultValue=0.35, minValue=0.0),
            QgsProcessingParameterNumber(
                self.W_SNOW, "Weight: snow",
                QgsProcessingParameterNumber.Double, defaultValue=0.25, minValue=0.0),
            QgsProcessingParameterNumber(
                self.W_ASPECT, "Weight: aspect",
                QgsProcessingParameterNumber.Double, defaultValue=0.20, minValue=0.0),
            QgsProcessingParameterNumber(
                self.W_CURV, "Weight: curvature",
                QgsProcessingParameterNumber.Double, defaultValue=0.20, minValue=0.0),
            QgsProcessingParameterNumber(
                self.LAPSE_PER_100M, "Snow lapse gradient (fraction per 100 m)",
                QgsProcessingParameterNumber.Double, defaultValue=0.06, minValue=0.0),
            QgsProcessingParameterNumber(
                self.SNOW_THRESHOLD, "Snow-day threshold (cm)",
                QgsProcessingParameterNumber.Double, defaultValue=5.0, minValue=0.0),
            QgsProcessingParameterNumber(
                self.MAX_TILES, "Max DTM tiles (download guard)",
                QgsProcessingParameterNumber.Integer, defaultValue=400, minValue=1),
            QgsProcessingParameterFile(
                self.TILE_CACHE, "DTM tile cache folder (blank = default)",
                behavior=QgsProcessingParameterFile.Folder, optional=True),
        ]
        for prm in adv:
            prm.setFlags(prm.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(prm)

    def _resolve_aoi(self, parameters, context, feedback, target_crs):
        """Bounding box in *target_crs*; polygon layer beats map extent."""
        layer = self.parameterAsVectorLayer(parameters, self.AOI_LAYER, context)
        extent_rect = self.parameterAsExtent(parameters, self.AOI_EXTENT, context)
        extent_crs = self.parameterAsExtentCrs(parameters, self.AOI_EXTENT, context)

        if layer is None and (extent_rect is None or extent_rect.isEmpty()):
            raise QgsProcessingException(
                "Provide at least one AOI: a polygon layer or a map extent.")

        if layer is not None:
            feedback.pushInfo(f"AOI source: polygon layer '{layer.name()}'")
            source_rect, source_crs = layer.extent(), layer.crs()
        else:
            feedback.pushInfo("AOI source: map extent")
            source_rect, source_crs = extent_rect, extent_crs

        if source_crs != target_crs:
            tr = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
            aoi_rect = tr.transformBoundingBox(source_rect)
        else:
            aoi_rect = source_rect

        feedback.pushInfo(
            f"AOI in EPSG:2056 - xmin={aoi_rect.xMinimum():.0f} "
            f"ymin={aoi_rect.yMinimum():.0f} xmax={aoi_rect.xMaximum():.0f} "
            f"ymax={aoi_rect.yMaximum():.0f}")
        return aoi_rect

    def processAlgorithm(self, parameters, context, feedback):
        _ensure_core_importable()
        from swissavalanche_core import run_analysis
        import geopandas as gpd
        from shapely.geometry import box

        crs_lv95 = QgsCoordinateReferenceSystem("EPSG:2056")
        aoi_rect = self._resolve_aoi(parameters, context, feedback, crs_lv95)
        aoi_gdf = gpd.GeoDataFrame(
            geometry=[box(aoi_rect.xMinimum(), aoi_rect.yMinimum(),
                          aoi_rect.xMaximum(), aoi_rect.yMaximum())],
            crs="EPSG:2056")

        snow_mode_idx = self.parameterAsEnum(parameters, self.SNOW_MODE, context)
        snow_mode = "climatology" if snow_mode_idx == 0 else "season"
        start = self.parameterAsString(parameters, self.START, context)
        end = self.parameterAsString(parameters, self.END, context)
        n_winters = self.parameterAsInt(parameters, self.N_WINTERS, context)
        winter_start = self.parameterAsString(parameters, self.WINTER_START, context)
        winter_end = self.parameterAsString(parameters, self.WINTER_END, context)
        res = self.parameterAsDouble(parameters, self.RESOLUTION, context)
        snow_lapse = self.parameterAsBool(parameters, self.SNOW_LAPSE, context)
        wind_w = self.parameterAsDouble(parameters, self.WIND_WEIGHT, context)
        load_factors = self.parameterAsBool(parameters, self.LOAD_FACTORS, context)
        load_score = self.parameterAsBool(parameters, self.LOAD_SCORE, context)
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)

        weights = {
            "slope":  self.parameterAsDouble(parameters, self.W_SLOPE, context),
            "snow":   self.parameterAsDouble(parameters, self.W_SNOW, context),
            "aspect": self.parameterAsDouble(parameters, self.W_ASPECT, context),
            "curv":   self.parameterAsDouble(parameters, self.W_CURV, context),
        }
        lapse = self.parameterAsDouble(parameters, self.LAPSE_PER_100M, context)
        snow_thr = self.parameterAsDouble(parameters, self.SNOW_THRESHOLD, context)
        max_tiles = self.parameterAsInt(parameters, self.MAX_TILES, context)
        tile_cache = self.parameterAsString(parameters, self.TILE_CACHE, context) or None

        def progress(frac, msg):
            if feedback.isCanceled():
                raise QgsProcessingException("Canceled by user.")
            feedback.setProgress(frac * 100.0)
            feedback.pushInfo(msg)

        try:
            result = run_analysis(
                aoi_gdf, start, end,
                snow_mode=snow_mode, n_winters=n_winters,
                winter_start_md=winter_start, winter_end_md=winter_end,
                base=out_dir, target_res=res,
                weights=weights, snow_lapse=snow_lapse, wind_weight=wind_w,
                lapse_per_100m=lapse, snow_threshold_cm=snow_thr,
                max_tiles=max_tiles, tile_cache_dir=tile_cache,
                morpho_engine="gdal", progress=progress,
            )
        except QgsProcessingException:
            raise
        except Exception as e:   # surface engine errors as QGIS errors
            raise QgsProcessingException(f"SwissAvalanche analysis failed: {e}")

        cls = result.stats.get("classes", {})
        feedback.pushInfo(
            "Susceptibility: " + ", ".join(
                f"{k} {v['pct']:.0f}%" for k, v in cls.items()))
        feedback.pushInfo(f"Snow factor: {result.snow_factor:.3f}")

        # Styled layers (class + zones always; score when requested).
        styled = [
            (result.class_path, "Avalanche susceptibility (class)", "SUSC_CLASS",
             _ClassRasterStyler()),
            (result.zones_path, "Susceptibility zones", "ZONES",
             _ZonesVectorStyler()),
        ]
        if load_score:
            styled.append((result.score_path, "Susceptibility score",
                           "SUSC_SCORE", _ScoreRasterStyler()))
        for path, label, key, styler in styled:
            details = QgsProcessingContext.LayerDetails(
                label, context.project(), key)
            _KEEP_ALIVE.append(styler)
            details.setPostProcessor(styler)
            context.addLayerToLoadOnCompletion(str(path), details)

        # Optional terrain-factor layers (default QGIS rendering).
        if load_factors:
            for path, label, key in [
                (result.rasters["slope"], "Slope (°)", "SLOPE"),
                (result.rasters["aspect"], "Aspect (°)", "ASPECT"),
                (result.rasters["curvature"], "Profile curvature", "CURVATURE"),
            ]:
                context.addLayerToLoadOnCompletion(
                    str(path),
                    QgsProcessingContext.LayerDetails(label, context.project(), key))

        return {
            "SUSC_CLASS": str(result.class_path),
            "SUSC_SCORE": str(result.score_path),
            "ZONES": str(result.zones_path),
            "SNOW_FACTOR": result.snow_factor,
            "OUTPUT_FOLDER": out_dir,
        }
