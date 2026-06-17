"""Processing provider for SwissSnow."""

from qgis.core import QgsProcessingProvider

from .swisssnow_algorithm import SwissSnowAlgorithm
from .swisssnow_validate_algorithm import SwissSnowValidateAlgorithm


class SwissSnowProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(SwissSnowAlgorithm())
        self.addAlgorithm(SwissSnowValidateAlgorithm())

    def id(self):
        return "swisssnow"

    def name(self):
        return "SwissSnow"

    def longName(self):
        return "SwissSnow - avalanche susceptibility"
