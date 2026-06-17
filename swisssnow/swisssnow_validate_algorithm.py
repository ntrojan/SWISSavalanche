"""
SwissSnow validation Processing algorithm.

Validates a susceptibility map against observed avalanche incidents: it samples
the susceptibility class/score at each incident and reports the frequency ratio
per class (FR>1 = the class is over-represented among incidents = good
discrimination) plus a success-rate / lift figure. Wraps swisssnow_core.validate.
"""

import json
import os
from pathlib import Path

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterRasterLayer,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMarkerSymbol,
    QgsProject,
    QgsRendererCategory,
)
from qgis.PyQt.QtGui import QColor

from .swisssnow_algorithm import (
    _ensure_core_importable, CLASS_COLORS, CLASS_LABELS, _KEEP_ALIVE,
)


class _IncidentsPointStyler(QgsProcessingLayerPostProcessorInterface):
    """Colour incident points by the susceptibility class they fall in."""

    def postProcessLayer(self, layer, context, feedback):
        if not hasattr(layer, "fields") or not layer.isValid():
            return
        cats = []
        # 0 = outside the susceptibility raster.
        out_sym = QgsMarkerSymbol.createSimple(
            {"name": "circle", "color": "200,200,200,90",
             "outline_color": "#808080", "size": "2.0"})
        cats.append(QgsRendererCategory(0, out_sym, "Outside AOI"))
        for cid in (1, 2, 3, 4):
            sym = QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": CLASS_COLORS[cid],
                 "outline_color": "#222222", "outline_width": "0.2", "size": "3.2"})
            cats.append(QgsRendererCategory(cid, sym, CLASS_LABELS[cid]))
        layer.setRenderer(QgsCategorizedSymbolRenderer("susc_class", cats))
        layer.triggerRepaint()


class SwissSnowValidateAlgorithm(QgsProcessingAlgorithm):
    INCIDENTS = "INCIDENTS"
    FETCH_SLF = "FETCH_SLF"
    CLASS_RASTER = "CLASS_RASTER"
    SCORE_RASTER = "SCORE_RASTER"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def name(self):
        return "validate_susceptibility"

    def displayName(self):
        return "Validate against incidents"

    def group(self):
        return "SwissSnow"

    def groupId(self):
        return "swisssnow"

    def shortHelpString(self):
        return (
            "<h3>Validate against incidents</h3>"
            "<p>Checks how well a susceptibility map discriminates real avalanche "
            "events. Provide a <b>point layer of incidents</b> and the "
            "<b>class</b> and <b>score</b> rasters produced by <i>Avalanche "
            "susceptibility</i>.</p>"
            "<p>The tool reports, per class, the share of incidents vs. the share "
            "of area, and their ratio - the <b>frequency ratio</b>:<br>"
            "&nbsp;&nbsp;FR &gt; 1 → incidents concentrate there (good);<br>"
            "&nbsp;&nbsp;FR ≈ 1 → no better than random.<br>"
            "It also gives the High+Very-high capture and a lift figure.</p>"
            "<p><b>Incidents:</b> either provide your own point layer, or leave it "
            "empty and keep <i>Download SLF incidents</i> on - the plugin then "
            "fetches the official SLF avalanche-accident dataset (Switzerland, "
            "since 1970) automatically and clips it to the map area. Output: a "
            "classified incidents layer and a <code>validation.json</code>.</p>"
            "<p><i>Validation is only meaningful over an area large enough to "
            "contain several past accidents.</i></p>")

    def createInstance(self):
        return SwissSnowValidateAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INCIDENTS, "Avalanche incidents (points) - leave empty to use SLF",
            [QgsProcessing.TypeVectorPoint], optional=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.FETCH_SLF,
            "Download SLF incidents (Switzerland, since 1970) clipped to the map area",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.CLASS_RASTER, "Susceptibility class raster"))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.SCORE_RASTER, "Susceptibility score raster"))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_FOLDER, "Output folder"))

    def processAlgorithm(self, parameters, context, feedback):
        _ensure_core_importable()
        from swisssnow_core import validate as ssv

        class_layer = self.parameterAsRasterLayer(parameters, self.CLASS_RASTER, context)
        score_layer = self.parameterAsRasterLayer(parameters, self.SCORE_RASTER, context)
        fetch_slf = self.parameterAsBool(parameters, self.FETCH_SLF, context)
        inc_source = self.parameterAsSource(parameters, self.INCIDENTS, context)
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)

        try:
            if inc_source is not None:
                inc_path = self.parameterAsCompatibleSourceLayerPath(
                    parameters, self.INCIDENTS, context, ["gpkg"], "gpkg", feedback)
                gdf = ssv.load_incidents(Path(inc_path))
                feedback.pushInfo(f"Using provided incidents layer ({len(gdf)} points).")
            elif fetch_slf:
                # Clip the SLF dataset to the class raster extent (in WGS84).
                ext = class_layer.extent()
                tr = QgsCoordinateTransform(
                    class_layer.crs(), QgsCoordinateReferenceSystem("EPSG:4326"),
                    QgsProject.instance())
                b = tr.transformBoundingBox(ext)
                feedback.pushInfo("Downloading SLF avalanche incidents…")
                gdf = ssv.fetch_slf_incidents(
                    bbox_wgs84=(b.xMinimum(), b.yMinimum(),
                                b.xMaximum(), b.yMaximum()))
                feedback.pushInfo(f"SLF incidents in this area: {len(gdf)}")
                if len(gdf) == 0:
                    raise QgsProcessingException(
                        "No SLF incidents fall within this area. Use a larger area "
                        "or provide your own incidents layer.")
            else:
                raise QgsProcessingException(
                    "Provide an incidents layer or enable 'Download SLF incidents'.")

            metrics, gdf_out = ssv.validate(
                Path(class_layer.source()), Path(score_layer.source()), gdf)
        except QgsProcessingException:
            raise
        except Exception as e:
            raise QgsProcessingException(f"Validation failed: {e}")

        points_path = Path(out_dir) / "incidents_classified.gpkg"
        json_path = Path(out_dir) / "validation.json"
        # Clean integer class field (0 = outside AOI) for categorized styling.
        gdf_out["susc_class"] = gdf_out["susc_class"].fillna(0).astype(int)
        gdf_out.to_file(points_path, driver="GPKG")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        feedback.pushInfo(
            f"Incidents inside AOI: {metrics['n_incidents_inside_aoi']} "
            f"(outside: {metrics['n_incidents_outside_aoi']})")
        feedback.pushInfo(f"{'Class':<12}{'incid%':>9}{'area%':>9}{'freq.ratio':>12}")
        for label, s in metrics["per_class"].items():
            feedback.pushInfo(
                f"{label:<12}{s['pct_incidents']:>8.1f}%{s['pct_area']:>8.1f}%"
                f"{s['frequency_ratio']:>12.2f}")
        feedback.pushInfo(
            f"High+VeryHigh capture: {metrics['high_vh_capture_pct']:.1f}% of "
            f"incidents on {metrics['high_vh_area_pct']:.1f}% of area "
            f"(lift {metrics['high_vh_lift']:.2f}x)")

        details = QgsProcessingContext.LayerDetails(
            "Incidents (classified)", context.project(), "INCIDENTS_CLASSIFIED")
        styler = _IncidentsPointStyler()
        _KEEP_ALIVE.append(styler)
        details.setPostProcessor(styler)
        context.addLayerToLoadOnCompletion(str(points_path), details)

        return {"INCIDENTS_CLASSIFIED": str(points_path),
                "VALIDATION_JSON": str(json_path)}
