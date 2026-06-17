"""
SwissAvalanche computational core.

A self-contained, UI-agnostic engine for seasonal avalanche-susceptibility
analysis. The same code backs both the QGIS plugin and the web app - neither
QGIS nor Streamlit is imported here.

Public entry point: :func:`swissavalanche_core.analysis.run_analysis`.
"""

from .analysis import run_analysis, AnalysisResult  # noqa: F401

__all__ = ["run_analysis", "AnalysisResult"]
__version__ = "0.2.0"
