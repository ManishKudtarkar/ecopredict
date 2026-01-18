"""Tests for EcoPredict preprocessing modules"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.clean_data import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.normalize import DataNormalizer


@pytest.fixture
def sample_raw_data():
    """Generate sample raw data with various issues"""
    np.random.seed(42)
    
    data = {
        'latitude': [19.0760, 18.5204, np.nan, 19.2183, 200.0],  # Contains NaN and invalid value
        'longitude': [72.8777, 73.8567, 75.1333, 72.9781, 73.2478],
        'temperature': [25.5, 28.2, 22.1, 26.8, -999],  # Contains outlier
        'precipitation': [2.1, 0.0, 5.2, 1.8, 3.4],
        'humidity': [65, 70, 55, 68, 150],  # Contains invalid value (>100)
        'forest_cover': [0.3, 0.5, 0.2, 0.4, 0.6],
        'species_count': [15, 12, 18, 14, 16],
        'duplicate_col': [1, 2, 3, 4, 5],  # Will be used to test duplicate handling
        'text_data': ['A', 'B', 'C', 'D', 'E']  # Non-numeric data
    }
    
    df = pd.DataFrame(data)
    
    # Add duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    return df


@pytest.fixture
def clean_data():
    """Generate clean data for feature engineering tests"""
    np.random.seed(42)
    
    data = {
        'latitude': np.random.uniform(15.6, 22.0, 100),
        'longitude': np.random.uniform(72.6, 80.9, 100),
        'temperature': np.random.normal(25, 5, 100),
        'precipitation': np.random.exponential(2, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'forest_cover': np.random.uniform(0, 1, 100),
        'agricultural_area': np.random.uniform(0, 1, 100),
        'urban_area': np.random.uniform(0, 1, 100),
        'species_count': np.random.poisson(15, 100),
        'elevation': np.random.normal(500, 200, 100),
        'population_density': np.random.exponential(100, 100)
    }
    
    return pd.DataFrame(data)


class TestDataCleaner:
    """Test DataCleaner functionality"""
    
    def test_initialization(self):
        """Test cleaner initialization"""
        cleaner = DataCleaner()
        assert cleaner is not None
    
    def test_remove_duplicates(self, sample_raw_data):
        """Test duplicate removal"""
        cleaner = DataCleaner()
        
        # Check that we have duplicates initially
        assert sample_raw_data.duplicated().any()
        
        cleaned = cleaner.remove_duplicates(sample_raw_data)
        
        # Check that duplicates are removed
        assert not cleaned.duplicated().any()
        assert len(cleaned) < len(sample_raw_data)
    
    def test_handle_missing_values(self, sample_raw_data):
        """Test missing value handling"""
        cleaner = DataCleaner()
        
        # Check that we have missing values initially
        assert sample_raw_data.isnull().any().any()
        
        cleaned = cleaner.handle_missing_values(sample_raw_data)
        
        # Check that missing values are handled
        # (depending on strategy, they might be filled or rows removed)
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        assert not cleaned[numeric_cols].isnull().any().any()
    
    def test_remove_outliers(self, sample_raw_data):
        """Test outlier removal"""
        cleaner = DataCleaner()
        
        # Temperature column has an outlier (-999)
        original_temp_min = sample_raw_data['temperature'].min()
        assert original_temp_min < -100  # Confirm outlier exists
        
        cleaned = cleaner.remove_outliers(sample_raw_data, ['temperature'])
        
        # Check that outlier is removed
        if len(cleaned) > 0:
            cleaned_temp_min = cleaned['temperature'].min()
            assert cleaned_temp_min > original_temp_min
    
    def test_validate_coordinates(self, sample_raw_data):
        """Test coordinate validation"""
        cleaner = DataCleaner()
        
        # Check that we have invalid coordinates initially
        invalid_lat = (sample_raw_data['latitude'] < -90) | (sample_raw_data['latitude'] > 90)
        assert invalid_lat.any()
        
        cleaned = cleaner.validate_coordinates(sample_raw_data)
        
        # Check that invalid coordinates are removed
        if len(cleaned) > 0:
            assert (cleaned['latitude'] >= -90).all()
            assert (cleaned['latitude'] <= 90).all()
            assert (cleaned['longitude'] >= -180).all()
            assert (cleaned['longitude'] <= 180).all()
    
    def test_clean_dataframe(self, sample_raw_data):
        """Test complete dataframe cleaning"""
        cleaner = DataCleaner()
        
        cleaned = cleaner.clean_dataframe(sample_raw_data)
        
        # Check that cleaning was applied
        assert len(cleaned) <= len(sample_raw_data)
        
        # Check that numeric columns don't have obvious issues
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        if len(cleaned) > 0 and len(numeric_cols) > 0:
            assert not cleaned[numeric_cols].isnull().any().any()


class TestFeatureEngineer:
    """Test FeatureEngineer functionality"""
    
    def test_initialization(self):
        """Test engineer initialization"""
        engineer = FeatureEngineer()
        assert engineer is not None
    
    def test_create_spatial_features(self, clean_data):
        """Test spatial feature creation"""
        engineer = FeatureEngineer()
        
        original_cols = set(clean_data.columns)
        enhanced = engineer.create_spatial_features(clean_data)
        new_cols = set(enhanced.columns) - original_cols
        
        # Should have new spatial features
        assert len(new_cols) > 0
        
        # Check for expected spatial features
        expected_features = ['lat_lon_interaction', 'distance_to_center']
        for feature in expected_features:
            if feature in enhanced.columns:
                assert not enhanced[feature].isnull().all()
    
    def test_create_climate_features(self, clean_data):
        """Test climate feature creation"""
        engineer = FeatureEngineer()
        
        original_cols = set(clean_data.columns)
        enhanced = engineer.create_climate_features(clean_data)
        new_cols = set(enhanced.columns) - original_cols
        
        # Should have new climate features
        assert len(new_cols) > 0
        
        # Check for expected climate features
        if 'temperature_precipitation_ratio' in enhanced.columns:
            assert not enhanced['temperature_precipitation_ratio'].isnull().all()
    
    def test_create_biodiversity_features(self, clean_data):
        """Test biodiversity feature creation"""
        engineer = FeatureEngineer()
        
        original_cols = set(clean_data.columns)
        enhanced = engineer.create_biodiversity_features(clean_data)
        new_cols = set(enhanced.columns) - original_cols
        
        # Should have new biodiversity features
        assert len(new_cols) > 0
        
        # Check for expected biodiversity features
        if 'species_density' in enhanced.columns:
            assert not enhanced['species_density'].isnull().all()
    
    def test_create_land_use_features(self, clean_data):
        """Test land use feature creation"""
        engineer = FeatureEngineer()
        
        original_cols = set(clean_data.columns)
        enhanced = engineer.create_land_use_features(clean_data)
        new_cols = set(enhanced.columns) - original_cols
        
        # Should have new land use features
        assert len(new_cols) > 0
        
        # Check for expected land use features
        expected_features = ['total_developed_area', 'natural_area_ratio']
        for feature in expected_features:
            if feature in enhanced.columns:
                assert not enhanced[feature].isnull().all()
    
    def test_create_interaction_features(self, clean_data):
        """Test interaction feature creation"""
        engineer = FeatureEngineer()
        
        feature_pairs = [('temperature', 'humidity'), ('forest_cover', 'species_count')]
        
        original_cols = set(clean_data.columns)
        enhanced = engineer.create_interaction_features(clean_data, feature_pairs)
        new_cols = set(enhanced.columns) - original_cols
        
        # Should have interaction features
        assert len(new_cols) > 0
        
        # Check that interaction features are created
        for col1, col2 in feature_pairs:
            if col1 in clean_data.columns and col2 in clean_data.columns:
                interaction_name = f"{col1}_{col2}_interaction"
                if interaction_name in enhanced.columns:
                    assert not enhanced[interaction_name].isnull().all()
    
    def test_create_polynomial_features(self, clean_data):
        """Test polynomial feature creation"""
        engineer = FeatureEngineer()
        
        columns = ['temperature', 'precipitation']
        
        original_cols = set(clean_data.columns)
        enhanced = engineer.create_polynomial_features(clean_data, columns, degree=2)
        new_cols = set(enhanced.columns) - original_cols
        
        # Should have polynomial features
        assert len(new_cols) > 0
        
        # Check for squared terms
        for col in columns:
            if col in clean_data.columns:
                squared_name = f"{col}_squared"
                if squared_name in enhanced.columns:
                    assert not enhanced[squared_name].isnull().all()
    
    def test_create_features(self, clean_data):
        """Test complete feature creation pipeline"""
        engineer = FeatureEngineer()
        
        original_shape = clean_data.shape
        enhanced = engineer.create_features(clean_data)
        
        # Should have more features
        assert enhanced.shape[1] >= original_shape[1]
        assert enhanced.shape[0] == original_shape[0]  # Same number of rows
        
        # Should not have NaN values in new features (mostly)
        new_cols = set(enhanced.columns) - set(clean_data.columns)
        if new_cols:
            # Allow some NaN values but not all
            for col in new_cols:
                if col in enhanced.columns:
                    nan_ratio = enhanced[col].isnull().mean()
                    assert nan_ratio < 0.5  # Less than 50% NaN


class TestDataNormalizer:
    """Test DataNormalizer functionality"""
    
    def test_initialization(self):
        """Test normalizer initialization"""
        normalizer = DataNormalizer()
        assert normalizer is not None
    
    def test_standard_scaling(self, clean_data):
        """Test standard scaling"""
        normalizer = DataNormalizer()
        
        columns = ['temperature', 'precipitation', 'humidity']
        scaled = normalizer.standard_scale(clean_data, columns)
        
        # Check that specified columns are scaled
        for col in columns:
            if col in clean_data.columns:
                # Mean should be close to 0, std close to 1
                assert abs(scaled[col].mean()) < 0.1
                assert abs(scaled[col].std() - 1.0) < 0.1
    
    def test_min_max_scaling(self, clean_data):
        """Test min-max scaling"""
        normalizer = DataNormalizer()
        
        columns = ['forest_cover', 'agricultural_area', 'urban_area']
        scaled = normalizer.min_max_scale(clean_data, columns)
        
        # Check that specified columns are scaled to [0, 1]
        for col in columns:
            if col in clean_data.columns:
                assert scaled[col].min() >= 0
                assert scaled[col].max() <= 1
    
    def test_robust_scaling(self, clean_data):
        """Test robust scaling"""
        normalizer = DataNormalizer()
        
        columns = ['elevation', 'population_density']
        scaled = normalizer.robust_scale(clean_data, columns)
        
        # Check that scaling was applied (median should be close to 0)
        for col in columns:
            if col in clean_data.columns:
                assert abs(scaled[col].median()) < 0.1
    
    def test_normalize_features(self, clean_data):
        """Test complete normalization pipeline"""
        normalizer = DataNormalizer()
        
        normalized = normalizer.normalize_features(clean_data)
        
        # Should have same shape
        assert normalized.shape == clean_data.shape
        
        # Numeric columns should be normalized
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['latitude', 'longitude']:
                continue  # Skip coordinate columns
            
            # Values should be in a reasonable range after normalization
            col_range = normalized[col].max() - normalized[col].min()
            assert col_range > 0  # Should have some variation
    
    def test_fit_transform_pattern(self, clean_data):
        """Test fit/transform pattern for normalization"""
        normalizer = DataNormalizer()
        
        # Split data
        train_data = clean_data.iloc[:80]
        test_data = clean_data.iloc[80:]
        
        # Fit on training data
        normalizer.fit(train_data)
        
        # Transform both training and test data
        train_normalized = normalizer.transform(train_data)
        test_normalized = normalizer.transform(test_data)
        
        # Check that transformations are consistent
        assert train_normalized.shape == train_data.shape
        assert test_normalized.shape == test_data.shape
        
        # Training data should be properly normalized
        numeric_cols = train_normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['latitude', 'longitude']:
                # Training data should have reasonable statistics
                assert not train_normalized[col].isnull().all()


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline"""
    
    def test_full_pipeline(self, sample_raw_data):
        """Test complete preprocessing pipeline"""
        # Initialize components
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        normalizer = DataNormalizer()
        
        # Apply preprocessing steps
        cleaned = cleaner.clean_dataframe(sample_raw_data)
        
        if len(cleaned) > 0:
            enhanced = engineer.create_features(cleaned)
            normalized = normalizer.normalize_features(enhanced)
            
            # Check final result
            assert len(normalized) <= len(sample_raw_data)
            assert normalized.shape[1] >= cleaned.shape[1]  # Should have more features
            
            # Should not have obvious data quality issues
            numeric_cols = normalized.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Most columns should not be all NaN
                for col in numeric_cols:
                    nan_ratio = normalized[col].isnull().mean()
                    assert nan_ratio < 0.9  # Less than 90% NaN
    
    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal valid data"""
        # Create minimal valid dataset
        minimal_data = pd.DataFrame({
            'latitude': [19.0, 18.5],
            'longitude': [72.8, 73.8],
            'temperature': [25.0, 26.0],
            'precipitation': [2.0, 1.5]
        })
        
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        normalizer = DataNormalizer()
        
        # Should handle minimal data without errors
        cleaned = cleaner.clean_dataframe(minimal_data)
        enhanced = engineer.create_features(cleaned)
        normalized = normalizer.normalize_features(enhanced)
        
        assert len(normalized) > 0
        assert not normalized.empty


if __name__ == "__main__":
    pytest.main([__file__])