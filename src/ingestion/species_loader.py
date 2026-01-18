"""Species occurrence data loader for EcoPredict"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import requests
from collections import Counter

from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates, calculate_distance

logger = get_logger(__name__)


class SpeciesLoader:
    """Loads and processes species occurrence data from various sources"""
    
    def __init__(self, data_dir: str = "data/raw/species_occurrence"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Conservation status categories
        self.conservation_status = {
            'LC': 'Least Concern',
            'NT': 'Near Threatened', 
            'VU': 'Vulnerable',
            'EN': 'Endangered',
            'CR': 'Critically Endangered',
            'EW': 'Extinct in Wild',
            'EX': 'Extinct'
        }
        
        # Threat levels
        self.threat_levels = {
            'LC': 0, 'NT': 1, 'VU': 2, 'EN': 3, 'CR': 4, 'EW': 5, 'EX': 6
        }
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load species occurrence data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with species occurrence data
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded species data from {filepath}")
            return self._validate_species_data(df)
        except Exception as e:
            logger.error(f"Error loading species data: {e}")
            raise
    
    def fetch_gbif_data(self, 
                       bounds: Tuple,
                       limit: int = 10000,
                       basis_of_record: str = "OBSERVATION") -> pd.DataFrame:
        """
        Fetch species occurrence data from GBIF API
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            limit: Maximum number of records
            basis_of_record: Type of record (OBSERVATION, SPECIMEN, etc.)
            
        Returns:
            DataFrame with GBIF occurrence data
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # GBIF API endpoint
        url = "https://api.gbif.org/v1/occurrence/search"
        
        params = {
            'decimalLatitude': f"{min_lat},{max_lat}",
            'decimalLongitude': f"{min_lon},{max_lon}",
            'basisOfRecord': basis_of_record,
            'hasCoordinate': 'true',
            'limit': min(limit, 300)  # GBIF API limit per request
        }
        
        try:
            # Mock GBIF response for demonstration
            logger.info(f"Fetching GBIF data for bounds {bounds}")
            
            # Generate synthetic GBIF-like data
            data = self._generate_gbif_like_data(bounds, limit)
            
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} GBIF records")
            return self._validate_species_data(df)
            
        except Exception as e:
            logger.error(f"Error fetching GBIF data: {e}")
            # Return synthetic data as fallback
            return self.generate_synthetic_data(bounds, limit)
    
    def generate_synthetic_data(self, 
                              bounds: Tuple, 
                              num_records: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic species occurrence data for testing
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            num_records: Number of records to generate
            
        Returns:
            DataFrame with synthetic species data
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Common species in Maharashtra region
        species_list = [
            'Panthera pardus', 'Melursus ursinus', 'Axis axis',
            'Sus scrofa', 'Macaca mulatta', 'Semnopithecus entellus',
            'Corvus splendens', 'Passer domesticus', 'Acridotheres tristis',
            'Columba livia', 'Psittacula krameri', 'Dicrurus macrocercus',
            'Naja naja', 'Python molurus', 'Varanus bengalensis',
            'Crocodylus palustris', 'Testudo elegans', 'Hemidactylus flaviviridis'
        ]
        
        # Generate random occurrences
        data = []
        
        for i in range(num_records):
            species = np.random.choice(species_list)
            
            # Generate coordinates with some clustering
            if np.random.random() < 0.3:  # 30% clustered around hotspots
                # Create hotspots
                hotspot_lat = np.random.uniform(min_lat, max_lat)
                hotspot_lon = np.random.uniform(min_lon, max_lon)
                lat = np.clip(np.random.normal(hotspot_lat, 0.05), min_lat, max_lat)
                lon = np.clip(np.random.normal(hotspot_lon, 0.05), min_lon, max_lon)
            else:
                lat = np.random.uniform(min_lat, max_lat)
                lon = np.random.uniform(min_lon, max_lon)
            
            # Assign conservation status
            status_probs = [0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02]
            status = np.random.choice(list(self.conservation_status.keys()), p=status_probs)
            
            # Determine if endemic (simplified)
            is_endemic = np.random.random() < 0.1  # 10% endemic
            
            data.append({
                'species_name': species,
                'latitude': lat,
                'longitude': lon,
                'observation_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 3650)),
                'conservation_status': status,
                'is_endemic': is_endemic,
                'observer': f"Observer_{np.random.randint(1, 100)}",
                'basis_of_record': np.random.choice(['OBSERVATION', 'SPECIMEN'], p=[0.8, 0.2])
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic species records")
        return self._validate_species_data(df)
    
    def calculate_species_metrics(self, 
                                df: pd.DataFrame, 
                                grid_points: pd.DataFrame,
                                radius_km: float = 5.0) -> pd.DataFrame:
        """
        Calculate species diversity metrics for grid points
        
        Args:
            df: DataFrame with species occurrence data
            grid_points: DataFrame with lat/lon grid points
            radius_km: Search radius in kilometers
            
        Returns:
            DataFrame with species metrics for each point
        """
        results = []
        
        for _, point_row in grid_points.iterrows():
            point_lat, point_lon = point_row['latitude'], point_row['longitude']
            
            # Find species within radius
            nearby_species = []
            
            for _, species_row in df.iterrows():
                distance = calculate_distance(
                    point_lat, point_lon,
                    species_row['latitude'], species_row['longitude']
                )
                
                if distance <= radius_km:
                    nearby_species.append(species_row)
            
            if not nearby_species:
                metrics = {
                    'latitude': point_lat,
                    'longitude': point_lon,
                    'species_count': 0,
                    'endemic_species': 0,
                    'threatened_species': 0,
                    'species_diversity': 0.0,
                    'threat_index': 0.0
                }
            else:
                nearby_df = pd.DataFrame(nearby_species)
                
                # Calculate metrics
                species_count = nearby_df['species_name'].nunique()
                endemic_count = nearby_df[nearby_df['is_endemic'] == True]['species_name'].nunique()
                
                # Count threatened species (VU, EN, CR)
                threatened_statuses = ['VU', 'EN', 'CR']
                threatened_count = nearby_df[
                    nearby_df['conservation_status'].isin(threatened_statuses)
                ]['species_name'].nunique()
                
                # Calculate Shannon diversity index
                species_counts = nearby_df['species_name'].value_counts()
                proportions = species_counts / species_counts.sum()
                shannon_diversity = -np.sum(proportions * np.log(proportions))
                
                # Calculate threat index (higher = more threatened)
                threat_scores = nearby_df['conservation_status'].map(self.threat_levels)
                threat_index = threat_scores.mean() / 6.0  # Normalize to 0-1
                
                metrics = {
                    'latitude': point_lat,
                    'longitude': point_lon,
                    'species_count': species_count,
                    'endemic_species': endemic_count,
                    'threatened_species': threatened_count,
                    'species_diversity': shannon_diversity,
                    'threat_index': threat_index
                }
            
            results.append(metrics)
        
        result_df = pd.DataFrame(results)
        logger.info(f"Calculated species metrics for {len(result_df)} points")
        return result_df
    
    def identify_biodiversity_hotspots(self, 
                                     df: pd.DataFrame,
                                     threshold_percentile: float = 90) -> pd.DataFrame:
        """
        Identify biodiversity hotspots based on species diversity
        
        Args:
            df: DataFrame with species metrics
            threshold_percentile: Percentile threshold for hotspot identification
            
        Returns:
            DataFrame with hotspot locations
        """
        # Calculate composite biodiversity score
        df['biodiversity_score'] = (
            df['species_count'] * 0.4 +
            df['endemic_species'] * 0.3 +
            df['species_diversity'] * 0.2 +
            df['threatened_species'] * 0.1
        )
        
        # Identify hotspots
        threshold = df['biodiversity_score'].quantile(threshold_percentile / 100)
        hotspots = df[df['biodiversity_score'] >= threshold].copy()
        
        hotspots['hotspot_rank'] = hotspots['biodiversity_score'].rank(ascending=False)
        
        logger.info(f"Identified {len(hotspots)} biodiversity hotspots")
        return hotspots.sort_values('hotspot_rank')
    
    def _generate_gbif_like_data(self, bounds: Tuple, limit: int) -> List[Dict]:
        """Generate GBIF-like synthetic data"""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        species_list = [
            'Panthera pardus', 'Melursus ursinus', 'Axis axis',
            'Sus scrofa', 'Macaca mulatta', 'Semnopithecus entellus'
        ]
        
        data = []
        for i in range(min(limit, 1000)):
            data.append({
                'species_name': np.random.choice(species_list),
                'latitude': np.random.uniform(min_lat, max_lat),
                'longitude': np.random.uniform(min_lon, max_lon),
                'observation_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 1000)),
                'conservation_status': np.random.choice(['LC', 'NT', 'VU'], p=[0.7, 0.2, 0.1]),
                'is_endemic': np.random.random() < 0.05,
                'observer': f"GBIF_User_{i}",
                'basis_of_record': 'OBSERVATION'
            })
        
        return data
    
    def _validate_species_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean species occurrence data
        
        Args:
            df: Raw species DataFrame
            
        Returns:
            Validated DataFrame
        """
        required_columns = ['species_name', 'latitude', 'longitude']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate coordinates
        invalid_coords = ~df.apply(
            lambda row: validate_coordinates(row['latitude'], row['longitude']), 
            axis=1
        )
        if invalid_coords.any():
            logger.warning(f"Removing {invalid_coords.sum()} records with invalid coordinates")
            df = df[~invalid_coords]
        
        # Clean species names
        df['species_name'] = df['species_name'].str.strip()
        df = df[df['species_name'].str.len() > 0]
        
        # Add missing columns with defaults
        if 'conservation_status' not in df.columns:
            df['conservation_status'] = 'LC'  # Default to Least Concern
        
        if 'is_endemic' not in df.columns:
            df['is_endemic'] = False
        
        if 'observation_date' not in df.columns:
            df['observation_date'] = pd.Timestamp.now()
        
        # Validate conservation status
        valid_statuses = list(self.conservation_status.keys())
        invalid_status = ~df['conservation_status'].isin(valid_statuses)
        if invalid_status.any():
            logger.warning(f"Setting {invalid_status.sum()} invalid conservation statuses to LC")
            df.loc[invalid_status, 'conservation_status'] = 'LC'
        
        logger.info(f"Validated species data: {len(df)} records")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed species data to CSV
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = self.data_dir.parent / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed species data to {output_path}")