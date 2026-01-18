"""Feature engineering utilities for EcoPredict"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from ..utils.logger import get_logger
from ..utils.helpers import calculate_distance

logger = get_logger(__name__)


class FeatureEngineer:
    """Handles feature engineering and creation"""
    
    def __init__(self):
        self.polynomial_features = None
        self.pca = None
        self.feature_selector = None
        self.feature_names = []
    
    def engineer_features(self, 
                         df: pd.DataFrame,
                         target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Comprehensive feature engineering pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering for {len(df)} records")
        
        engineered_df = df.copy()
        
        # 1. Create spatial features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            engineered_df = self._create_spatial_features(engineered_df)
        
        # 2. Create temporal features
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for date_col in date_columns:
            engineered_df = self._create_temporal_features(engineered_df, date_col)
        
        # 3. Create interaction features
        engineered_df = self._create_interaction_features(engineered_df)
        
        # 4. Create aggregation features
        engineered_df = self._create_aggregation_features(engineered_df)
        
        # 5. Create domain-specific features
        engineered_df = self._create_ecological_features(engineered_df)
        
        # 6. Create statistical features
        engineered_df = self._create_statistical_features(engineered_df)
        
        logger.info(f"Feature engineering completed: {len(engineered_df.columns)} features")
        return engineered_df
    
    def _create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial-based features"""
        df_spatial = df.copy()
        
        # Distance from center point
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        df_spatial['distance_from_center'] = df.apply(
            lambda row: calculate_distance(
                row['latitude'], row['longitude'],
                center_lat, center_lon
            ), axis=1
        )
        
        # Coordinate transformations
        df_spatial['lat_squared'] = df_spatial['latitude'] ** 2
        df_spatial['lon_squared'] = df_spatial['longitude'] ** 2
        df_spatial['lat_lon_product'] = df_spatial['latitude'] * df_spatial['longitude']
        
        # Spatial bins (for capturing regional patterns)
        df_spatial['lat_bin'] = pd.cut(df_spatial['latitude'], bins=10, labels=False)
        df_spatial['lon_bin'] = pd.cut(df_spatial['longitude'], bins=10, labels=False)
        
        # Distance to boundaries
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        df_spatial['distance_to_north'] = lat_max - df_spatial['latitude']
        df_spatial['distance_to_south'] = df_spatial['latitude'] - lat_min
        df_spatial['distance_to_east'] = lon_max - df_spatial['longitude']
        df_spatial['distance_to_west'] = df_spatial['longitude'] - lon_min
        
        logger.info("Created spatial features")
        return df_spatial
    
    def _create_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create temporal features from date columns"""
        df_temporal = df.copy()
        
        # Ensure datetime format
        df_temporal[date_column] = pd.to_datetime(df_temporal[date_column])
        
        # Extract temporal components
        df_temporal[f'{date_column}_year'] = df_temporal[date_column].dt.year
        df_temporal[f'{date_column}_month'] = df_temporal[date_column].dt.month
        df_temporal[f'{date_column}_day'] = df_temporal[date_column].dt.day
        df_temporal[f'{date_column}_dayofyear'] = df_temporal[date_column].dt.dayofyear
        df_temporal[f'{date_column}_weekday'] = df_temporal[date_column].dt.weekday
        df_temporal[f'{date_column}_quarter'] = df_temporal[date_column].dt.quarter
        
        # Cyclical encoding for seasonal patterns
        df_temporal[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df_temporal[f'{date_column}_month'] / 12)
        df_temporal[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df_temporal[f'{date_column}_month'] / 12)
        df_temporal[f'{date_column}_day_sin'] = np.sin(2 * np.pi * df_temporal[f'{date_column}_dayofyear'] / 365)
        df_temporal[f'{date_column}_day_cos'] = np.cos(2 * np.pi * df_temporal[f'{date_column}_dayofyear'] / 365)
        
        # Time since reference point
        reference_date = df_temporal[date_column].min()
        df_temporal[f'{date_column}_days_since_start'] = (df_temporal[date_column] - reference_date).dt.days
        
        logger.info(f"Created temporal features for {date_column}")
        return df_temporal
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df_interaction = df.copy()
        
        # Climate interactions
        climate_cols = ['temperature', 'precipitation', 'humidity', 'wind_speed']
        existing_climate = [col for col in climate_cols if col in df.columns]
        
        if len(existing_climate) >= 2:
            # Temperature-precipitation interaction
            if 'temperature' in existing_climate and 'precipitation' in existing_climate:
                df_interaction['temp_precip_interaction'] = (
                    df_interaction['temperature'] * df_interaction['precipitation']
                )
            
            # Humidity-temperature interaction
            if 'humidity' in existing_climate and 'temperature' in existing_climate:
                df_interaction['humidity_temp_interaction'] = (
                    df_interaction['humidity'] * df_interaction['temperature']
                )
        
        # Land use interactions
        landuse_cols = ['forest_cover', 'agricultural_area', 'urban_area', 'water_bodies']
        existing_landuse = [col for col in landuse_cols if col in df.columns]
        
        if len(existing_landuse) >= 2:
            # Forest-urban interaction (fragmentation indicator)
            if 'forest_cover' in existing_landuse and 'urban_area' in existing_landuse:
                df_interaction['forest_urban_ratio'] = (
                    df_interaction['forest_cover'] / (df_interaction['urban_area'] + 0.001)
                )
            
            # Agricultural-forest interaction
            if 'agricultural_area' in existing_landuse and 'forest_cover' in existing_landuse:
                df_interaction['ag_forest_ratio'] = (
                    df_interaction['agricultural_area'] / (df_interaction['forest_cover'] + 0.001)
                )
        
        # Species-environment interactions
        species_cols = ['species_count', 'endemic_species', 'threatened_species']
        existing_species = [col for col in species_cols if col in df.columns]
        
        if 'species_count' in existing_species and 'forest_cover' in existing_landuse:
            df_interaction['species_forest_interaction'] = (
                df_interaction['species_count'] * df_interaction['forest_cover']
            )
        
        logger.info("Created interaction features")
        return df_interaction
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features"""
        df_agg = df.copy()
        
        # Land use totals and ratios
        landuse_cols = ['forest_cover', 'agricultural_area', 'urban_area', 'water_bodies']
        existing_landuse = [col for col in landuse_cols if col in df.columns]
        
        if len(existing_landuse) >= 2:
            df_agg['total_developed_area'] = df_agg[existing_landuse].sum(axis=1)
            df_agg['natural_area_ratio'] = (
                df_agg.get('forest_cover', 0) + df_agg.get('water_bodies', 0)
            ) / (df_agg['total_developed_area'] + 0.001)
        
        # Climate composite indices
        climate_cols = ['temperature', 'precipitation', 'humidity']
        existing_climate = [col for col in climate_cols if col in df.columns]
        
        if len(existing_climate) >= 2:
            # Normalize and create composite climate index
            climate_normalized = df_agg[existing_climate].copy()
            for col in existing_climate:
                climate_normalized[col] = (
                    (climate_normalized[col] - climate_normalized[col].min()) /
                    (climate_normalized[col].max() - climate_normalized[col].min())
                )
            df_agg['climate_favorability_index'] = climate_normalized.mean(axis=1)
        
        # Species diversity indices
        species_cols = ['species_count', 'endemic_species', 'threatened_species']
        existing_species = [col for col in species_cols if col in df.columns]
        
        if len(existing_species) >= 2:
            df_agg['total_species_richness'] = df_agg[existing_species].sum(axis=1)
            
            if 'species_count' in existing_species and 'endemic_species' in existing_species:
                df_agg['endemic_ratio'] = (
                    df_agg['endemic_species'] / (df_agg['species_count'] + 0.001)
                )
        
        logger.info("Created aggregation features")
        return df_agg
    
    def _create_ecological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific ecological features"""
        df_eco = df.copy()
        
        # Habitat quality index
        if all(col in df.columns for col in ['forest_cover', 'water_bodies', 'urban_area']):
            df_eco['habitat_quality_index'] = (
                df_eco['forest_cover'] * 0.6 +
                df_eco['water_bodies'] * 0.3 -
                df_eco['urban_area'] * 0.4
            )
        
        # Fragmentation index
        if 'forest_cover' in df.columns and 'urban_area' in df.columns:
            df_eco['fragmentation_index'] = (
                df_eco['urban_area'] / (df_eco['forest_cover'] + 0.001)
            )
        
        # Climate stress index
        if all(col in df.columns for col in ['temperature', 'precipitation']):
            # Simple stress index based on extreme values
            temp_stress = np.abs(df_eco['temperature'] - df_eco['temperature'].median())
            precip_stress = np.abs(df_eco['precipitation'] - df_eco['precipitation'].median())
            
            df_eco['climate_stress_index'] = (
                temp_stress / df_eco['temperature'].std() +
                precip_stress / df_eco['precipitation'].std()
            )
        
        # Biodiversity threat index
        if 'threatened_species' in df.columns and 'species_count' in df.columns:
            df_eco['threat_ratio'] = (
                df_eco['threatened_species'] / (df_eco['species_count'] + 0.001)
            )
        
        # Human pressure index
        if all(col in df.columns for col in ['urban_area', 'agricultural_area']):
            df_eco['human_pressure_index'] = (
                df_eco['urban_area'] * 0.7 + df_eco['agricultural_area'] * 0.3
            )
        
        logger.info("Created ecological features")
        return df_eco
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        df_stats = df.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove coordinate and ID columns
        exclude_cols = ['latitude', 'longitude', 'id', 'index']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) >= 3:
            # Rolling statistics (if we have enough features)
            df_stats['feature_mean'] = df_stats[feature_cols].mean(axis=1)
            df_stats['feature_std'] = df_stats[feature_cols].std(axis=1)
            df_stats['feature_min'] = df_stats[feature_cols].min(axis=1)
            df_stats['feature_max'] = df_stats[feature_cols].max(axis=1)
            df_stats['feature_range'] = df_stats['feature_max'] - df_stats['feature_min']
            
            # Coefficient of variation
            df_stats['feature_cv'] = df_stats['feature_std'] / (df_stats['feature_mean'] + 0.001)
        
        logger.info("Created statistical features")
        return df_stats
    
    def create_polynomial_features(self, 
                                 df: pd.DataFrame, 
                                 feature_columns: List[str],
                                 degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        
        if self.polynomial_features is None:
            self.polynomial_features = PolynomialFeatures(
                degree=degree, 
                include_bias=False,
                interaction_only=False
            )
        
        # Select only specified columns
        feature_data = df[feature_columns]
        
        # Generate polynomial features
        poly_features = self.polynomial_features.fit_transform(feature_data)
        
        # Create feature names
        feature_names = self.polynomial_features.get_feature_names_out(feature_columns)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Combine with original DataFrame (excluding original feature columns to avoid duplication)
        result_df = df.drop(feature_columns, axis=1)
        result_df = pd.concat([result_df, poly_df], axis=1)
        
        logger.info(f"Created {len(feature_names)} polynomial features")
        return result_df
    
    def apply_pca(self, 
                  df: pd.DataFrame, 
                  feature_columns: List[str],
                  n_components: Optional[int] = None,
                  variance_threshold: float = 0.95) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        
        feature_data = df[feature_columns]
        
        if n_components is None:
            # Determine number of components based on variance threshold
            pca_temp = PCA()
            pca_temp.fit(feature_data)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(feature_data)
        
        # Create PCA feature names
        pca_columns = [f'pca_component_{i+1}' for i in range(n_components)]
        
        # Create DataFrame with PCA features
        pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)
        
        # Combine with original DataFrame (excluding original feature columns)
        result_df = df.drop(feature_columns, axis=1)
        result_df = pd.concat([result_df, pca_df], axis=1)
        
        logger.info(f"Applied PCA: {len(feature_columns)} -> {n_components} components")
        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return result_df
    
    def select_features(self, 
                       df: pd.DataFrame,
                       target_column: str,
                       k: int = 10,
                       method: str = 'f_regression') -> pd.DataFrame:
        """Select top k features based on statistical tests"""
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        # Choose selection method
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Fit selector and transform features
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        # Create result DataFrame
        result_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        result_df[target_column] = y
        
        self.feature_selector = selector
        self.feature_names = selected_features
        
        logger.info(f"Selected {k} features using {method}")
        logger.info(f"Selected features: {selected_features}")
        
        return result_df
    
    def get_feature_importance_scores(self) -> Dict[str, float]:
        """Get feature importance scores from the last selection"""
        
        if self.feature_selector is None or not self.feature_names:
            return {}
        
        scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)
        
        importance_dict = {
            self.feature_names[i]: scores[selected_indices[i]] 
            for i in range(len(self.feature_names))
        }
        
        return importance_dict