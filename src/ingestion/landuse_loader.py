"""Land use data loader for EcoPredict"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon

from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates, create_grid

logger = get_logger(__name__)


class LandUseLoader:
    """Loads and processes land use data from various sources"""
    
    def __init__(self, data_dir: str = "data/raw/land_use"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Land use categories
        self.landuse_categories = {
            1: 'forest',
            2: 'agricultural',
            3: 'urban',
            4: 'water',
            5: 'grassland',
            6: 'barren'
        }
    
    def load_from_shapefile(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load land use data from shapefile
        
        Args:
            shapefile_path: Path to shapefile
            
        Returns:
            GeoDataFrame with land use data
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded land use shapefile: {len(gdf)} features")
            return self._validate_landuse_data(gdf)
        except Exception as e:
            logger.error(f"Error loading shapefile: {e}")
            raise
    
    def load_from_raster(self, raster_path: str, bounds: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Load land use data from raster file
        
        Args:
            raster_path: Path to raster file
            bounds: Optional bounds (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            DataFrame with land use data
        """
        try:
            with rasterio.open(raster_path) as src:
                # Read raster data
                if bounds:
                    window = rasterio.windows.from_bounds(*bounds, src.transform)
                    data = src.read(1, window=window)
                    transform = rasterio.windows.transform(window, src.transform)
                else:
                    data = src.read(1)
                    transform = src.transform
                
                # Convert to coordinates
                rows, cols = np.where(data != src.nodata)
                coords = rasterio.transform.xy(transform, rows, cols)
                
                df = pd.DataFrame({
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'landuse_code': data[rows, cols]
                })
                
                # Map codes to categories
                df['landuse_type'] = df['landuse_code'].map(self.landuse_categories)
                
                logger.info(f"Loaded land use raster: {len(df)} pixels")
                return df
                
        except Exception as e:
            logger.error(f"Error loading raster: {e}")
            raise
    
    def generate_synthetic_data(self, 
                              bounds: Tuple, 
                              resolution: float = 0.01) -> pd.DataFrame:
        """
        Generate synthetic land use data for testing
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: Grid resolution in degrees
            
        Returns:
            DataFrame with synthetic land use data
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create grid
        grid_df = create_grid(bounds, resolution)
        
        # Generate land use patterns
        data = []
        
        for _, row in grid_df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            
            # Simple land use model based on location
            # Urban areas near center
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            dist_to_center = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            
            # Assign land use based on distance and random factors
            if dist_to_center < 0.1:  # Urban center
                landuse_probs = [0.1, 0.2, 0.6, 0.05, 0.03, 0.02]  # forest, ag, urban, water, grass, barren
            elif dist_to_center < 0.3:  # Suburban
                landuse_probs = [0.3, 0.4, 0.2, 0.05, 0.03, 0.02]
            else:  # Rural
                landuse_probs = [0.5, 0.3, 0.05, 0.1, 0.03, 0.02]
            
            landuse_code = np.random.choice(list(self.landuse_categories.keys()), p=landuse_probs)
            
            # Calculate coverage percentages for each type
            coverage = self._calculate_coverage(lat, lon, landuse_code)
            
            data.append({
                'latitude': lat,
                'longitude': lon,
                'primary_landuse': self.landuse_categories[landuse_code],
                'forest_cover': coverage['forest'],
                'agricultural_area': coverage['agricultural'],
                'urban_area': coverage['urban'],
                'water_bodies': coverage['water']
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic land use records")
        return df
    
    def calculate_landuse_metrics(self, gdf: gpd.GeoDataFrame, grid_points: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate land use metrics for grid points
        
        Args:
            gdf: GeoDataFrame with land use polygons
            grid_points: DataFrame with lat/lon grid points
            
        Returns:
            DataFrame with land use metrics for each point
        """
        results = []
        
        for _, point_row in grid_points.iterrows():
            lat, lon = point_row['latitude'], point_row['longitude']
            point_geom = Point(lon, lat)
            
            # Create buffer around point (e.g., 1km radius)
            buffer = point_geom.buffer(0.01)  # ~1km at equator
            
            # Find intersecting polygons
            intersecting = gdf[gdf.geometry.intersects(buffer)]
            
            if len(intersecting) == 0:
                # No data available
                metrics = {
                    'latitude': lat,
                    'longitude': lon,
                    'forest_cover': 0,
                    'agricultural_area': 0,
                    'urban_area': 0,
                    'water_bodies': 0
                }
            else:
                # Calculate coverage percentages
                total_area = buffer.area
                coverage = {'forest': 0, 'agricultural': 0, 'urban': 0, 'water': 0}
                
                for _, poly_row in intersecting.iterrows():
                    intersection = poly_row.geometry.intersection(buffer)
                    area_fraction = intersection.area / total_area
                    
                    landuse_type = poly_row.get('landuse_type', 'unknown')
                    if landuse_type in coverage:
                        coverage[landuse_type] += area_fraction
                
                metrics = {
                    'latitude': lat,
                    'longitude': lon,
                    'forest_cover': min(coverage['forest'], 1.0),
                    'agricultural_area': min(coverage['agricultural'], 1.0),
                    'urban_area': min(coverage['urban'], 1.0),
                    'water_bodies': min(coverage['water'], 1.0)
                }
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        logger.info(f"Calculated land use metrics for {len(df)} points")
        return df
    
    def _calculate_coverage(self, lat: float, lon: float, primary_landuse: int) -> Dict[str, float]:
        """
        Calculate land use coverage percentages for a location
        
        Args:
            lat: Latitude
            lon: Longitude
            primary_landuse: Primary land use code
            
        Returns:
            Dictionary with coverage percentages
        """
        # Base coverage based on primary land use
        coverage = {'forest': 0.0, 'agricultural': 0.0, 'urban': 0.0, 'water': 0.0}
        
        primary_type = self.landuse_categories[primary_landuse]
        
        if primary_type == 'forest':
            coverage['forest'] = np.random.uniform(0.6, 0.9)
            remaining = 1.0 - coverage['forest']
            coverage['agricultural'] = remaining * np.random.uniform(0.1, 0.3)
            coverage['water'] = remaining * np.random.uniform(0.05, 0.15)
            coverage['urban'] = remaining * np.random.uniform(0.0, 0.1)
            
        elif primary_type == 'agricultural':
            coverage['agricultural'] = np.random.uniform(0.5, 0.8)
            remaining = 1.0 - coverage['agricultural']
            coverage['forest'] = remaining * np.random.uniform(0.1, 0.4)
            coverage['water'] = remaining * np.random.uniform(0.05, 0.15)
            coverage['urban'] = remaining * np.random.uniform(0.0, 0.2)
            
        elif primary_type == 'urban':
            coverage['urban'] = np.random.uniform(0.4, 0.7)
            remaining = 1.0 - coverage['urban']
            coverage['forest'] = remaining * np.random.uniform(0.1, 0.3)
            coverage['agricultural'] = remaining * np.random.uniform(0.1, 0.3)
            coverage['water'] = remaining * np.random.uniform(0.0, 0.1)
            
        elif primary_type == 'water':
            coverage['water'] = np.random.uniform(0.3, 0.6)
            remaining = 1.0 - coverage['water']
            coverage['forest'] = remaining * np.random.uniform(0.2, 0.5)
            coverage['agricultural'] = remaining * np.random.uniform(0.1, 0.4)
            coverage['urban'] = remaining * np.random.uniform(0.0, 0.2)
        
        # Normalize to ensure sum <= 1
        total = sum(coverage.values())
        if total > 1.0:
            for key in coverage:
                coverage[key] /= total
        
        return coverage
    
    def _validate_landuse_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate and clean land use data
        
        Args:
            gdf: Raw GeoDataFrame
            
        Returns:
            Validated GeoDataFrame
        """
        # Check for required columns
        if 'geometry' not in gdf.columns:
            raise ValueError("Missing geometry column")
        
        # Ensure valid geometries
        invalid_geom = ~gdf.geometry.is_valid
        if invalid_geom.any():
            logger.warning(f"Fixing {invalid_geom.sum()} invalid geometries")
            gdf.loc[invalid_geom, 'geometry'] = gdf.loc[invalid_geom, 'geometry'].buffer(0)
        
        # Remove empty geometries
        empty_geom = gdf.geometry.is_empty
        if empty_geom.any():
            logger.warning(f"Removing {empty_geom.sum()} empty geometries")
            gdf = gdf[~empty_geom]
        
        logger.info(f"Validated land use data: {len(gdf)} features")
        return gdf
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed land use data to CSV
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = self.data_dir.parent / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed land use data to {output_path}")