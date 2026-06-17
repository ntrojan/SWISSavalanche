"""Processing provider for SwissAvalanche."""

from qgis.core import QgsProcessingProvider

from .swissavalanche_algorithm import SwissAvalancheAlgorithm
from .swissavalanche_validate_algorithm import SwissAvalancheValidateAlgorithm


class SwissAvalancheProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(SwissAvalancheAlgorithm())
        self.addAlgorithm(SwissAvalancheValidateAlgorithm())

    def id(self):
        return "swissavalanche"

    def name(self):
        return "SwissAvalanche"

    def longName(self):
        return "SwissAvalanche - avalanche susceptibility"
