"""Data cleaning utilities for EcoPredict"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

from ..utils.logger import get_logger
from ..utils.helpers import validate_coordinates

logger = get_logger(__name__)


class DataCleaner:
    """Handles data cleaning and quality assurance"""
    
    def __init__(self):
        self.label_encoders = {}
        self.imputers = {}
    
    def clean_dataset(self, 
                     df: pd.DataFrame, 
                     target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if any)
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {len(df)} records")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Remove duplicate rows
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # 2. Validate coordinates if present
        if 'latitude' in cleaned_df.columns and 'longitude' in cleaned_df.columns:
            cleaned_df = self._validate_coordinates(cleaned_df)
        
        # 3. Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df, target_column)
        
        # 4. Remove outliers
        cleaned_df = self._remove_outliers(cleaned_df, target_column)
        
        # 5. Standardize data types
        cleaned_df = self._standardize_dtypes(cleaned_df)
        
        # 6. Validate data ranges
        cleaned_df = self._validate_ranges(cleaned_df)
        
        logger.info(f"Data cleaning completed: {len(cleaned_df)} records remaining")
        return cleaned_df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_count = len(df)
        
        # Remove exact duplicates
        df_clean = df.drop_duplicates()
        
        # Remove duplicates based on coordinates (if present)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Round coordinates to avoid floating point precision issues
            df_clean['lat_rounded'] = df_clean['latitude'].round(6)
            df_clean['lon_rounded'] = df_clean['longitude'].round(6)
            
            df_clean = df_clean.drop_duplicates(subset=['lat_rounded', 'lon_rounded'])
            df_clean = df_clean.drop(['lat_rounded', 'lon_rounded'], axis=1)
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate records")
        
        return df_clean
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinate data"""
        initial_count = len(df)
        
        # Check for valid coordinate ranges
        valid_coords = df.apply(
            lambda row: validate_coordinates(row['latitude'], row['longitude']), 
            axis=1
        )
        
        df_clean = df[valid_coords].copy()
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} records with invalid coordinates")
        
        return df_clean
    
    def _handle_missing_values(self, 
                              df: pd.DataFrame, 
                              target_column: Optional[str] = None) -> pd.DataFrame:
        """Handle missing values using various strategies"""
        
        # Don't impute target column - remove those rows instead
        if target_column and target_column in df.columns:
            initial_count = len(df)
            df = df.dropna(subset=[target_column])
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} records with missing target values")
        
        # Identify columns with missing values
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0].index.tolist()
        
        if not missing_cols:
            return df
        
        logger.info(f"Handling missing values in {len(missing_cols)} columns")
        
        df_imputed = df.copy()
        
        for col in missing_cols:
            if col == target_column:
                continue
                
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct > 50:
                # Drop columns with >50% missing values
                logger.warning(f"Dropping column '{col}' with {missing_pct:.1f}% missing values")
                df_imputed = df_imputed.drop(col, axis=1)
                continue
            
            # Choose imputation strategy based on data type and missing percentage
            if df[col].dtype in ['object', 'category']:
                # Categorical data - use mode
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                
            elif missing_pct < 10:
                # Low missing percentage - use KNN imputation
                numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                else:
                    # Fallback to median
                    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                    
            else:
                # Higher missing percentage - use median/mode
                if df[col].dtype in [np.number]:
                    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                else:
                    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
        
        return df_imputed
    
    def _remove_outliers(self, 
                        df: pd.DataFrame, 
                        target_column: Optional[str] = None,
                        method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Don't remove outliers from coordinate columns or target
        exclude_cols = ['latitude', 'longitude']
        if target_column:
            exclude_cols.append(target_column)
        
        outlier_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not outlier_cols:
            return df
        
        initial_count = len(df)
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in outlier_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        
        elif method == 'zscore':
            from scipy import stats
            
            for col in outlier_cols:
                z_scores = np.abs(stats.zscore(df_clean[col]))
                df_clean = df_clean[z_scores < 3]  # Remove points with |z-score| > 3
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier records using {method} method")
        
        return df_clean
    
    def _standardize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types"""
        df_clean = df.copy()
        
        # Convert string columns that should be numeric
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass
        
        # Convert boolean-like strings
        bool_cols = ['is_endemic', 'is_threatened', 'is_protected']
        for col in bool_cols:
            if col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].map({
                        'True': True, 'False': False, 'true': True, 'false': False,
                        'Yes': True, 'No': False, 'yes': True, 'no': False,
                        '1': True, '0': False, 1: True, 0: False
                    })
        
        # Ensure coordinate columns are float
        coord_cols = ['latitude', 'longitude']
        for col in coord_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(float)
        
        return df_clean
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges for known columns"""
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # Define expected ranges for common columns
        range_validations = {
            'temperature': (-50, 60),  # Celsius
            'precipitation': (0, 1000),  # mm
            'humidity': (0, 100),  # percentage
            'wind_speed': (0, 200),  # km/h
            'forest_cover': (0, 1),  # proportion
            'agricultural_area': (0, 1),  # proportion
            'urban_area': (0, 1),  # proportion
            'water_bodies': (0, 1),  # proportion
            'species_count': (0, None),  # count
            'endemic_species': (0, None),  # count
            'threatened_species': (0, None),  # count
        }
        
        for col, (min_val, max_val) in range_validations.items():
            if col in df_clean.columns:
                if min_val is not None:
                    df_clean = df_clean[df_clean[col] >= min_val]
                if max_val is not None:
                    df_clean = df_clean[df_clean[col] <= max_val]
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} records with invalid data ranges")
        
        return df_clean
    
    def encode_categorical_features(self, 
                                  df: pd.DataFrame, 
                                  categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    # Transform using existing encoder
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a data quality report"""
        
        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'duplicates': 0,
            'coordinate_issues': 0
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        report['missing_values'] = {
            col: {
                'count': int(count),
                'percentage': round((count / len(df)) * 100, 2)
            }
            for col, count in missing_counts.items() if count > 0
        }
        
        # Data types
        report['data_types'] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }
        
        # Duplicates
        report['duplicates'] = len(df) - len(df.drop_duplicates())
        
        # Coordinate validation
        if 'latitude' in df.columns and 'longitude' in df.columns:
            invalid_coords = ~df.apply(
                lambda row: validate_coordinates(row['latitude'], row['longitude']), 
                axis=1
            )
            report['coordinate_issues'] = int(invalid_coords.sum())
        
        return report