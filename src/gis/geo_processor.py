"""Geographic data processing utilities for EcoPredict"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, box
from typing import Dict, List, Tuple, Optional, Any
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import folium
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates, create_grid

logger = get_logger(__name__)


class GeoProcessor:
    """Geographic data processing and spatial analysis"""
    
    def __init__(self, crs: str = "EPSG:4326"):
        self.crs = crs
        
    def merge_predictions(self, shapefile: str, predictions_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Merge predictions with geographic boundaries
        
        Args:
            shapefile: Path to shapefile
            predictions_df: DataFrame with predictions
            
        Returns:
            GeoDataFrame with merged data
        """
        try:
            geo = gpd.read_file(shapefile)
            merged = geo.merge(predictions_df, on="region", how="left")
            logger.info(f"Merged predictions with {len(geo)} geographic features")
            return merged
        except Exception as e:
            logger.error(f"Failed to merge predictions: {e}")
            raise
    
    def points_to_geodataframe(self, 
                              df: pd.DataFrame,
                              lat_col: str = "latitude",
                              lon_col: str = "longitude") -> gpd.GeoDataFrame:
        """Convert DataFrame with coordinates to GeoDataFrame
        
        Args:
            df: DataFrame with coordinate columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            GeoDataFrame with Point geometries
        """
        try:
            # Create Point geometries
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)
            
            logger.info(f"Created GeoDataFrame with {len(gdf)} points")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to create GeoDataFrame: {e}")
            raise
    
    def create_spatial_grid(self, 
                           bounds: Tuple[float, float, float, float],
                           resolution: float) -> gpd.GeoDataFrame:
        """Create spatial grid for analysis
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: Grid cell size in degrees
            
        Returns:
            GeoDataFrame with grid polygons
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create grid coordinates
        x_coords = np.arange(min_lon, max_lon + resolution, resolution)
        y_coords = np.arange(min_lat, max_lat + resolution, resolution)
        
        # Create grid cells
        polygons = []
        grid_ids = []
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                # Create cell polygon
                cell = box(x, y, x + resolution, y + resolution)
                polygons.append(cell)
                grid_ids.append(f"cell_{i}_{j}")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'grid_id': grid_ids,
            'geometry': polygons
        }, crs=self.crs)
        
        logger.info(f"Created spatial grid with {len(gdf)} cells")
        return gdf
    
    def spatial_join_predictions(self,
                               predictions_gdf: gpd.GeoDataFrame,
                               boundaries_gdf: gpd.GeoDataFrame,
                               how: str = "inner") -> gpd.GeoDataFrame:
        """Perform spatial join between predictions and boundaries
        
        Args:
            predictions_gdf: GeoDataFrame with prediction points
            boundaries_gdf: GeoDataFrame with boundary polygons
            how: Type of join ("inner", "left", "right")
            
        Returns:
            GeoDataFrame with spatially joined data
        """
        try:
            # Ensure same CRS
            if predictions_gdf.crs != boundaries_gdf.crs:
                predictions_gdf = predictions_gdf.to_crs(boundaries_gdf.crs)
            
            # Perform spatial join
            joined = gpd.sjoin(predictions_gdf, boundaries_gdf, how=how, predicate="within")
            
            logger.info(f"Spatial join resulted in {len(joined)} records")
            return joined
            
        except Exception as e:
            logger.error(f"Spatial join failed: {e}")
            raise
    
    def calculate_spatial_statistics(self,
                                   gdf: gpd.GeoDataFrame,
                                   value_column: str,
                                   group_column: Optional[str] = None) -> Dict[str, Any]:
        """Calculate spatial statistics
        
        Args:
            gdf: GeoDataFrame with data
            value_column: Column to calculate statistics for
            group_column: Optional grouping column
            
        Returns:
            Dictionary with spatial statistics
        """
        try:
            stats = {}
            
            if group_column:
                # Group statistics
                grouped_stats = gdf.groupby(group_column)[value_column].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).to_dict('index')
                stats['grouped'] = grouped_stats
            
            # Overall statistics
            stats['overall'] = {
                'count': len(gdf),
                'mean': gdf[value_column].mean(),
                'std': gdf[value_column].std(),
                'min': gdf[value_column].min(),
                'max': gdf[value_column].max(),
                'median': gdf[value_column].median()
            }
            
            # Spatial extent
            bounds = gdf.total_bounds
            stats['spatial_extent'] = {
                'min_x': bounds[0],
                'min_y': bounds[1],
                'max_x': bounds[2],
                'max_y': bounds[3],
                'area_degrees': (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            }
            
            logger.info("Calculated spatial statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate spatial statistics: {e}")
            raise
    
    def buffer_analysis(self,
                       gdf: gpd.GeoDataFrame,
                       buffer_distance: float,
                       target_gdf: gpd.GeoDataFrame,
                       target_column: str) -> gpd.GeoDataFrame:
        """Perform buffer analysis around points
        
        Args:
            gdf: GeoDataFrame with points to buffer
            buffer_distance: Buffer distance in degrees
            target_gdf: GeoDataFrame with target features
            target_column: Column to analyze in target features
            
        Returns:
            GeoDataFrame with buffer analysis results
        """
        try:
            # Create buffers
            gdf_buffered = gdf.copy()
            gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_distance)
            
            # Spatial join with targets
            joined = gpd.sjoin(gdf_buffered, target_gdf, how="left", predicate="intersects")
            
            # Calculate statistics within buffers
            buffer_stats = joined.groupby(joined.index)[target_column].agg([
                'count', 'mean', 'sum'
            ]).reset_index()
            
            # Merge back with original data
            result = gdf.merge(buffer_stats, left_index=True, right_on='index', how='left')
            
            logger.info(f"Completed buffer analysis with {buffer_distance} degree radius")
            return result
            
        except Exception as e:
            logger.error(f"Buffer analysis failed: {e}")
            raise
    
    def create_voronoi_polygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create Voronoi polygons from points
        
        Args:
            gdf: GeoDataFrame with point geometries
            
        Returns:
            GeoDataFrame with Voronoi polygons
        """
        try:
            from scipy.spatial import Voronoi
            from shapely.geometry import Polygon
            
            # Extract coordinates
            coords = np.array([[point.x, point.y] for point in gdf.geometry])
            
            # Create Voronoi diagram
            vor = Voronoi(coords)
            
            # Create polygons
            polygons = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    polygon_coords = [vor.vertices[i] for i in region]
                    if len(polygon_coords) >= 3:
                        polygons.append(Polygon(polygon_coords))
                    else:
                        polygons.append(None)
                else:
                    polygons.append(None)
            
            # Create GeoDataFrame
            voronoi_gdf = gpd.GeoDataFrame({
                'geometry': polygons[:len(gdf)]
            }, crs=gdf.crs)
            
            # Merge with original data
            result = pd.concat([gdf.reset_index(drop=True), voronoi_gdf], axis=1)
            result = gpd.GeoDataFrame(result, crs=gdf.crs)
            
            logger.info(f"Created {len(result)} Voronoi polygons")
            return result
            
        except Exception as e:
            logger.error(f"Voronoi polygon creation failed: {e}")
            raise
    
    def rasterize_predictions(self,
                            gdf: gpd.GeoDataFrame,
                            value_column: str,
                            bounds: Tuple[float, float, float, float],
                            resolution: float,
                            output_path: Optional[str] = None) -> np.ndarray:
        """Rasterize prediction data
        
        Args:
            gdf: GeoDataFrame with predictions
            value_column: Column with values to rasterize
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: Pixel resolution in degrees
            output_path: Optional path to save raster
            
        Returns:
            Numpy array with rasterized data
        """
        try:
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # Calculate raster dimensions
            width = int((max_lon - min_lon) / resolution)
            height = int((max_lat - min_lat) / resolution)
            
            # Create transform
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
            
            # Prepare data for rasterization
            shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[value_column])]
            
            # Rasterize
            raster = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.float32
            )
            
            # Save if output path provided
            if output_path:
                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=height, width=width,
                    count=1, dtype=raster.dtype,
                    crs=self.crs,
                    transform=transform
                ) as dst:
                    dst.write(raster, 1)
                
                logger.info(f"Saved raster to {output_path}")
            
            logger.info(f"Rasterized data to {width}x{height} grid")
            return raster
            
        except Exception as e:
            logger.error(f"Rasterization failed: {e}")
            raise
    
    def create_interactive_map(self,
                             gdf: gpd.GeoDataFrame,
                             value_column: str,
                             center: Optional[Tuple[float, float]] = None,
                             zoom: int = 10) -> folium.Map:
        """Create interactive map with predictions
        
        Args:
            gdf: GeoDataFrame with data
            value_column: Column to visualize
            center: Map center (lat, lon)
            zoom: Initial zoom level
            
        Returns:
            Folium map object
        """
        try:
            # Calculate center if not provided
            if center is None:
                bounds = gdf.total_bounds
                center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
            
            # Create base map
            m = folium.Map(location=center, zoom_start=zoom)
            
            # Add data to map
            if gdf.geometry.iloc[0].geom_type == 'Point':
                # Add points as markers
                for idx, row in gdf.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        popup=f"{value_column}: {row[value_column]:.3f}",
                        color='red' if row[value_column] > 0.6 else 'orange' if row[value_column] > 0.3 else 'green',
                        fill=True
                    ).add_to(m)
            else:
                # Add polygons
                folium.GeoJson(
                    gdf,
                    style_function=lambda feature: {
                        'fillColor': 'red' if feature['properties'][value_column] > 0.6 else 'orange' if feature['properties'][value_column] > 0.3 else 'green',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    popup=folium.GeoJsonPopup(fields=[value_column])
                ).add_to(m)
            
            logger.info("Created interactive map")
            return m
            
        except Exception as e:
            logger.error(f"Map creation failed: {e}")
            raise
