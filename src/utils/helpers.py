"""Helper functions for EcoPredict"""

import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
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


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")


def load_data_file(filepath: str) -> pd.DataFrame:
    """Load data from various file formats
    
    Args:
        filepath: Path to data file
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(filepath)
    elif filepath.suffix.lower() == '.json':
        return pd.read_json(filepath)
    elif filepath.suffix.lower() == '.parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def get_timestamp() -> str:
    """Get current timestamp as string
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_data_quality(data: pd.DataFrame, 
                         required_columns: List[str],
                         min_rows: int = 10) -> Dict[str, Any]:
    """Validate data quality
    
    Args:
        data: Input DataFrame
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Check minimum rows
    if len(data) < min_rows:
        results['valid'] = False
        results['issues'].append(f"Insufficient data: {len(data)} rows (minimum: {min_rows})")
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        results['valid'] = False
        results['issues'].append(f"Missing columns: {missing_cols}")
    
    # Check for empty data
    if data.empty:
        results['valid'] = False
        results['issues'].append("DataFrame is empty")
    
    # Calculate statistics
    results['stats'] = {
        'rows': len(data),
        'columns': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    return results


def clip_to_bounds(value: float, min_val: float, max_val: float) -> float:
    """Clip value to specified bounds
    
    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))