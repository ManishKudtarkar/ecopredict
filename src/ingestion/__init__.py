"""Data ingestion modules for EcoPredict"""

from .climate_loader import ClimateLoader
from .landuse_loader import LandUseLoader
from .species_loader import SpeciesLoader

__all__ = ['ClimateLoader', 'LandUseLoader', 'SpeciesLoader']