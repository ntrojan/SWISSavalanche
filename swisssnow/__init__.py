"""SwissSnow QGIS plugin - entry point."""


def classFactory(iface):   # noqa: N802 (QGIS-mandated name)
    from .swisssnow_plugin import SwissSnowPlugin
    return SwissSnowPlugin(iface)
