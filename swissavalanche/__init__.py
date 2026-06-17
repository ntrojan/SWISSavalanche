"""SwissAvalanche QGIS plugin - entry point."""


def classFactory(iface):   # noqa: N802 (QGIS-mandated name)
    from .swissavalanche_plugin import SwissAvalanchePlugin
    return SwissAvalanchePlugin(iface)
