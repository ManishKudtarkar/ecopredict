"""Data normalization utilities for EcoPredict"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataNormalizer:
    """Handles data normalization and scaling"""
    
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        self.feature_ranges = {}
    
    def normalize_features(self, 
                          df: pd.DataFrame,
                          method: str = 'standard',
                          feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize features using specified method
        
        Args:
            df: Input DataFrame
            method: Normalization method ('standard', 'minmax', 'robust', 'quantile')
            feature_columns: Columns to normalize (if None, auto-detect numeric columns)
            
        Returns:
            DataFrame with normalized features
        """
        
        if feature_columns is None:
            # Auto-detect numeric columns, excluding coordinates and IDs
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['latitude', 'longitude', 'id', 'index']
            feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_columns:
            logger.warning("No numeric features found for normalization")
            return df
        
        logger.info(f"Normalizing {len(feature_columns)} features using {method} method")
        
        df_normalized = df.copy()
        
        # Choose scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='uniform')
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform features
        df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        logger.info(f"Normalization completed using {method} method")
        return df_normalized
    
    def apply_power_transform(self, 
                            df: pd.DataFrame,
                            feature_columns: Optional[List[str]] = None,
                            method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Apply power transformation to make features more Gaussian
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to transform
            method: Transformation method ('yeo-johnson', 'box-cox')
            
        Returns:
            DataFrame with transformed features
        """
        
        if feature_columns is None:
            # Auto-detect numeric columns with skewness
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['latitude', 'longitude', 'id', 'index']
            candidate_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Only transform columns with significant skewness
            feature_columns = []
            for col in candidate_cols:
                skewness = abs(df[col].skew())
                if skewness > 0.5:  # Threshold for skewness
                    feature_columns.append(col)
        
        if not feature_columns:
            logger.info("No features require power transformation")
            return df
        
        logger.info(f"Applying {method} power transform to {len(feature_columns)} features")
        
        df_transformed = df.copy()
        
        # Apply transformation
        transformer = PowerTransformer(method=method, standardize=False)
        
        try:
            df_transformed[feature_columns] = transformer.fit_transform(df[feature_columns])
            self.transformers[f'power_{method}'] = transformer
            logger.info(f"Power transformation completed")
        except Exception as e:
            logger.warning(f"Power transformation failed: {e}")
            logger.info("Falling back to log transformation")
            df_transformed = self._apply_log_transform(df_transformed, feature_columns)
        
        return df_transformed
    
    def _apply_log_transform(self, 
                           df: pd.DataFrame, 
                           feature_columns: List[str]) -> pd.DataFrame:
        """Apply log transformation as fallback"""
        
        df_log = df.copy()
        
        for col in feature_columns:
            # Ensure positive values for log transform
            min_val = df_log[col].min()
            if min_val <= 0:
                # Shift values to make them positive
                shift = abs(min_val) + 1
                df_log[col] = df_log[col] + shift
            
            # Apply log transformation
            df_log[col] = np.log1p(df_log[col])  # log1p is more stable for small values
        
        logger.info(f"Applied log transformation to {len(feature_columns)} features")
        return df_log
    
    def handle_skewed_features(self, 
                             df: pd.DataFrame,
                             skewness_threshold: float = 0.5) -> pd.DataFrame:
        """
        Automatically handle skewed features
        
        Args:
            df: Input DataFrame
            skewness_threshold: Threshold for considering a feature skewed
            
        Returns:
            DataFrame with skewness-corrected features
        """
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['latitude', 'longitude', 'id', 'index']
        feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        df_corrected = df.copy()
        skewed_features = []
        
        for col in feature_columns:
            skewness = abs(df[col].skew())
            
            if skewness > skewness_threshold:
                skewed_features.append(col)
                
                # Choose transformation based on skewness level
                if skewness > 2.0:
                    # Highly skewed - use power transform
                    try:
                        transformer = PowerTransformer(method='yeo-johnson')
                        df_corrected[col] = transformer.fit_transform(df[[col]]).ravel()
                        self.transformers[f'{col}_power'] = transformer
                    except:
                        # Fallback to log transform
                        df_corrected = self._apply_log_transform(df_corrected, [col])
                
                elif skewness > 1.0:
                    # Moderately skewed - use square root or log
                    if df[col].min() >= 0:
                        df_corrected[col] = np.sqrt(df_corrected[col])
                    else:
                        df_corrected = self._apply_log_transform(df_corrected, [col])
                
                else:
                    # Mildly skewed - use quantile transform
                    transformer = QuantileTransformer(output_distribution='normal')
                    df_corrected[col] = transformer.fit_transform(df[[col]]).ravel()
                    self.transformers[f'{col}_quantile'] = transformer
        
        if skewed_features:
            logger.info(f"Corrected skewness in {len(skewed_features)} features: {skewed_features}")
        else:
            logger.info("No significantly skewed features found")
        
        return df_corrected
    
    def normalize_by_groups(self, 
                           df: pd.DataFrame,
                           group_column: str,
                           feature_columns: List[str],
                           method: str = 'standard') -> pd.DataFrame:
        """
        Normalize features within groups (e.g., by region or time period)
        
        Args:
            df: Input DataFrame
            group_column: Column to group by
            feature_columns: Features to normalize
            method: Normalization method
            
        Returns:
            DataFrame with group-wise normalized features
        """
        
        logger.info(f"Applying group-wise normalization by {group_column}")
        
        df_normalized = df.copy()
        
        # Choose scaler
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Apply normalization within each group
        for group_name, group_data in df.groupby(group_column):
            group_indices = group_data.index
            
            scaler = scaler_class()
            normalized_features = scaler.fit_transform(group_data[feature_columns])
            
            df_normalized.loc[group_indices, feature_columns] = normalized_features
            
            # Store scaler for this group
            self.scalers[f'{group_column}_{group_name}'] = scaler
        
        logger.info(f"Group-wise normalization completed for {len(df[group_column].unique())} groups")
        return df_normalized
    
    def create_feature_ranges(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Calculate and store feature ranges for validation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature ranges
        """
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        ranges = {}
        for col in numeric_cols:
            ranges[col] = (df[col].min(), df[col].max())
        
        self.feature_ranges = ranges
        logger.info(f"Calculated ranges for {len(ranges)} features")
        
        return ranges
    
    def validate_feature_ranges(self, 
                              df: pd.DataFrame,
                              tolerance: float = 0.1) -> Dict[str, bool]:
        """
        Validate that features are within expected ranges
        
        Args:
            df: DataFrame to validate
            tolerance: Tolerance for range validation (as fraction)
            
        Returns:
            Dictionary indicating which features are within range
        """
        
        if not self.feature_ranges:
            logger.warning("No feature ranges stored for validation")
            return {}
        
        validation_results = {}
        
        for col in df.columns:
            if col in self.feature_ranges:
                expected_min, expected_max = self.feature_ranges[col]
                actual_min, actual_max = df[col].min(), df[col].max()
                
                # Calculate tolerance bounds
                range_size = expected_max - expected_min
                tolerance_margin = range_size * tolerance
                
                min_ok = actual_min >= (expected_min - tolerance_margin)
                max_ok = actual_max <= (expected_max + tolerance_margin)
                
                validation_results[col] = min_ok and max_ok
                
                if not validation_results[col]:
                    logger.warning(
                        f"Feature {col} out of range: "
                        f"expected [{expected_min:.3f}, {expected_max:.3f}], "
                        f"got [{actual_min:.3f}, {actual_max:.3f}]"
                    )
        
        return validation_results
    
    def inverse_transform(self, 
                         df: pd.DataFrame,
                         scaler_name: str,
                         feature_columns: List[str]) -> pd.DataFrame:
        """
        Apply inverse transformation to denormalize features
        
        Args:
            df: Normalized DataFrame
            scaler_name: Name of the scaler to use
            feature_columns: Columns to denormalize
            
        Returns:
            DataFrame with denormalized features
        """
        
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler {scaler_name} not found")
        
        df_denormalized = df.copy()
        scaler = self.scalers[scaler_name]
        
        df_denormalized[feature_columns] = scaler.inverse_transform(df[feature_columns])
        
        logger.info(f"Applied inverse transformation using {scaler_name}")
        return df_denormalized
    
    def get_normalization_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of normalization statistics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with normalization statistics
        """
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = {
            'total_features': len(numeric_cols),
            'feature_statistics': {},
            'skewness_analysis': {},
            'normalization_recommendations': []
        }
        
        for col in numeric_cols:
            stats = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
            
            summary['feature_statistics'][col] = stats
            
            # Skewness analysis
            abs_skewness = abs(stats['skewness'])
            if abs_skewness > 2.0:
                skew_level = 'highly_skewed'
                recommendation = 'power_transform'
            elif abs_skewness > 1.0:
                skew_level = 'moderately_skewed'
                recommendation = 'log_transform'
            elif abs_skewness > 0.5:
                skew_level = 'mildly_skewed'
                recommendation = 'quantile_transform'
            else:
                skew_level = 'normal'
                recommendation = 'standard_scaling'
            
            summary['skewness_analysis'][col] = {
                'level': skew_level,
                'recommendation': recommendation
            }
            
            summary['normalization_recommendations'].append({
                'feature': col,
                'method': recommendation,
                'reason': f"Skewness: {stats['skewness']:.3f}"
            })
        
        return summary


class CustomScaler(BaseEstimator, TransformerMixin):
    """Custom scaler for domain-specific normalization"""
    
    def __init__(self, method='minmax', feature_ranges=None):
        self.method = method
        self.feature_ranges = feature_ranges or {}
        self.scalers_ = {}
    
    def fit(self, X, y=None):
        """Fit the scaler to the data"""
        
        if isinstance(X, pd.DataFrame):
            columns = X.columns
        else:
            columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        for i, col in enumerate(columns):
            if col in self.feature_ranges:
                # Use predefined ranges
                min_val, max_val = self.feature_ranges[col]
                self.scalers_[col] = (min_val, max_val - min_val)
            else:
                # Calculate from data
                if isinstance(X, pd.DataFrame):
                    col_data = X[col]
                else:
                    col_data = X[:, i]
                
                if self.method == 'minmax':
                    min_val = col_data.min()
                    scale = col_data.max() - min_val
                elif self.method == 'standard':
                    min_val = col_data.mean()
                    scale = col_data.std()
                else:
                    min_val = col_data.min()
                    scale = col_data.max() - min_val
                
                self.scalers_[col] = (min_val, scale)
        
        return self
    
    def transform(self, X):
        """Transform the data"""
        
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in X.columns:
                if col in self.scalers_:
                    min_val, scale = self.scalers_[col]
                    X_transformed[col] = (X[col] - min_val) / (scale + 1e-8)
            return X_transformed
        else:
            X_transformed = X.copy()
            for i, col in enumerate(self.scalers_.keys()):
                min_val, scale = self.scalers_[col]
                X_transformed[:, i] = (X[:, i] - min_val) / (scale + 1e-8)
            return X_transformed
    
    def inverse_transform(self, X):
        """Inverse transform the data"""
        
        if isinstance(X, pd.DataFrame):
            X_inverse = X.copy()
            for col in X.columns:
                if col in self.scalers_:
                    min_val, scale = self.scalers_[col]
                    X_inverse[col] = X[col] * scale + min_val
            return X_inverse
        else:
            X_inverse = X.copy()
            for i, col in enumerate(self.scalers_.keys()):
                min_val, scale = self.scalers_[col]
                X_inverse[:, i] = X[:, i] * scale + min_val
            return X_inverse