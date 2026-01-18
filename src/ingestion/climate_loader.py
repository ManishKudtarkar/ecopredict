"""Climate data loader for EcoPredict"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates

logger = get_logger(__name__)


class ClimateLoader:
    """Loads and processes climate data from various sources"""
    
    def __init__(self, data_dir: str = "data/raw/climate"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load climate data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with climate data
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded climate data from {filepath}")
            return self._validate_climate_data(df)
        except Exception as e:
            logger.error(f"Error loading climate data: {e}")
            raise
    
    def fetch_weather_api(self, 
                         lat: float, 
                         lon: float, 
                         start_date: str, 
                         end_date: str,
                         api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch climate data from weather API
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            api_key: API key for weather service
            
        Returns:
            DataFrame with climate data
        """
        if not validate_coordinates(lat, lon):
            raise ValueError("Invalid coordinates")
        
        # Mock API call - replace with actual weather API
        logger.info(f"Fetching weather data for {lat}, {lon}")
        
        # Generate mock data for demonstration
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = {
            'date': dates,
            'latitude': lat,
            'longitude': lon,
            'temperature': np.random.normal(25, 5, len(dates)),
            'precipitation': np.random.exponential(2, len(dates)),
            'humidity': np.random.normal(60, 15, len(dates)),
            'wind_speed': np.random.exponential(3, len(dates))
        }
        
        df = pd.DataFrame(data)
        return self._validate_climate_data(df)
    
    def generate_synthetic_data(self, 
                              bounds: tuple, 
                              resolution: float = 0.1,
                              num_days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic climate data for testing
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: Grid resolution in degrees
            num_days: Number of days to generate
            
        Returns:
            DataFrame with synthetic climate data
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create spatial grid
        lons = np.arange(min_lon, max_lon + resolution, resolution)
        lats = np.arange(min_lat, max_lat + resolution, resolution)
        
        # Create temporal range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        
        for lat in lats:
            for lon in lons:
                for date in dates:
                    # Add spatial and temporal variation
                    temp_base = 20 + (lat - min_lat) * 2  # Temperature varies with latitude
                    temp_seasonal = 5 * np.sin(2 * np.pi * date.dayofyear / 365)
                    
                    data.append({
                        'date': date,
                        'latitude': lat,
                        'longitude': lon,
                        'temperature': temp_base + temp_seasonal + np.random.normal(0, 2),
                        'precipitation': np.random.exponential(2),
                        'humidity': np.clip(np.random.normal(60, 15), 0, 100),
                        'wind_speed': np.random.exponential(3)
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic climate records")
        return self._validate_climate_data(df)
    
    def aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily climate data to monthly averages
        
        Args:
            df: DataFrame with daily climate data
            
        Returns:
            DataFrame with monthly aggregated data
        """
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        agg_funcs = {
            'temperature': 'mean',
            'precipitation': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }
        
        monthly_df = df.groupby(['latitude', 'longitude', 'year_month']).agg(agg_funcs).reset_index()
        monthly_df['date'] = monthly_df['year_month'].dt.start_time
        monthly_df = monthly_df.drop('year_month', axis=1)
        
        logger.info(f"Aggregated to {len(monthly_df)} monthly records")
        return monthly_df
    
    def _validate_climate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean climate data
        
        Args:
            df: Raw climate DataFrame
            
        Returns:
            Validated DataFrame
        """
        required_columns = ['latitude', 'longitude', 'temperature', 'precipitation', 'humidity', 'wind_speed']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate coordinates
        invalid_coords = ~df.apply(lambda row: validate_coordinates(row['latitude'], row['longitude']), axis=1)
        if invalid_coords.any():
            logger.warning(f"Removing {invalid_coords.sum()} records with invalid coordinates")
            df = df[~invalid_coords]
        
        # Clean data ranges
        df['humidity'] = df['humidity'].clip(0, 100)
        df['precipitation'] = df['precipitation'].clip(0, None)
        df['wind_speed'] = df['wind_speed'].clip(0, None)
        
        # Remove outliers (simple method)
        for col in ['temperature', 'precipitation', 'humidity', 'wind_speed']:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        
        logger.info(f"Validated climate data: {len(df)} records")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed climate data to CSV
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = self.data_dir.parent / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed climate data to {output_path}")