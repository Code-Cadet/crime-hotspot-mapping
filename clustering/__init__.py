"""
Spatial clustering analysis package.

Provides clustering algorithms for identifying spatial patterns
and hotspots in crime data.
"""

__version__ = "0.1.0"

from .cluster_analysis import CrimeClusterAnalyzer

__all__ = [
    'CrimeClusterAnalyzer'
]
