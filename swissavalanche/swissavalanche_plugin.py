"""SwissAvalanche plugin: registers the Processing provider with QGIS."""

from qgis.core import QgsApplication

from .swissavalanche_provider import SwissAvalancheProvider


class SwissAvalanchePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = SwissAvalancheProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
