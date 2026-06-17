"""Processing provider for SwissAvalanche."""

import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .swissavalanche_algorithm import SwissAvalancheAlgorithm
from .swissavalanche_validate_algorithm import SwissAvalancheValidateAlgorithm

_ICON_PATH = os.path.join(os.path.dirname(__file__), "resources", "icon.png")


class SwissAvalancheProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(SwissAvalancheAlgorithm())
        self.addAlgorithm(SwissAvalancheValidateAlgorithm())

    def icon(self):
        return QIcon(_ICON_PATH)

    def id(self):
        return "swissavalanche"

    def name(self):
        return "SwissAvalanche"

    def longName(self):
        return "SwissAvalanche - avalanche susceptibility"
