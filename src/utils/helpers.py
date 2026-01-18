"""Helper functions for EcoPredict"""

import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from .logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_model(model: BaseEstimator, filepath: str) -> None:
    """
    Save a trained model to disk
    
    Args:
        model: Trained scikit-learn model
        filepath: Path to save the model
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model(filepath: str) -> BaseEstimator:
    """
    Load a trained model from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        True if coordinates are valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r


def create_grid(bounds: tuple, resolution: float) -> pd.DataFrame:
    """
    Create a regular grid of points within given bounds
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        resolution: Grid resolution in degrees
        
    Returns:
        DataFrame with lat, lon columns
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    lons = np.arange(min_lon, max_lon + resolution, resolution)
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    return pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })


def normalize_features(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalize specified columns using min-max scaling
    
    Args:
        data: Input DataFrame
        columns: Columns to normalize
        
    Returns:
        DataFrame with normalized columns
    """
    result = data.copy()
    
    for col in columns:
        if col in result.columns:
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val > min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
    
    return result


def get_risk_category(risk_score: float, thresholds: Dict[str, float]) -> str:
    """
    Categorize risk score based on thresholds
    
    Args:
        risk_score: Numerical risk score (0-1)
        thresholds: Dictionary with 'low' and 'medium' thresholds
        
    Returns:
        Risk category ('low', 'medium', 'high')
    """
    if risk_score < thresholds['low']:
        return 'low'
    elif risk_score < thresholds['medium']:
        return 'medium'
    else:
        return 'high'