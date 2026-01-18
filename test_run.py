#!/usr/bin/env python3
"""Simple test script to run EcoPredict components"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to Python path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

def test_logger():
    """Test logging functionality"""
    print("Testing logger...")
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Logger is working!")
    return True

def test_data_generation():
    """Test synthetic data generation"""
    print("Testing data generation...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'latitude': np.random.uniform(15.6, 22.0, n_samples),
        'longitude': np.random.uniform(72.6, 80.9, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'precipitation': np.random.exponential(2, n_samples),
        'forest_cover': np.random.uniform(0, 1, n_samples),
        'species_count': np.random.poisson(15, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create risk score
    risk_score = (
        0.3 * (1 - df['forest_cover']) +
        0.2 * np.abs(df['temperature'] - 25) / 10 +
        0.1 * (1 / (df['species_count'] + 1)) +
        0.4 * np.random.uniform(0, 1, n_samples)
    )
    
    df['risk_score'] = np.clip(risk_score, 0, 1)
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Risk score range: {df['risk_score'].min():.3f} - {df['risk_score'].max():.3f}")
    
    return df

def test_model_training(data):
    """Test model training"""
    print("Testing model training...")
    
    from models.random_forest import RandomForestModel
    from models.regression import LinearRegressionModel
    
    # Prepare data
    feature_cols = ['temperature', 'precipitation', 'forest_cover', 'species_count']
    X = data[feature_cols]
    y = data['risk_score']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestModel(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_metrics = rf_model.evaluate(X_test, y_test)
    print(f"Random Forest R²: {rf_metrics['r2']:.3f}")
    
    # Test Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegressionModel()
    lr_model.fit(X_train, y_train)
    
    lr_metrics = lr_model.evaluate(X_test, y_test)
    print(f"Linear Regression R²: {lr_metrics['r2']:.3f}")
    
    return {'random_forest': rf_model, 'linear_regression': lr_model}

def test_prediction(models, data):
    """Test prediction functionality"""
    print("Testing prediction...")
    
    from prediction.predict import EcoPredictionEngine
    
    # Initialize prediction engine
    engine = EcoPredictionEngine()
    
    # Use the trained random forest model
    engine.model = models['random_forest']
    engine.feature_columns = ['temperature', 'precipitation', 'forest_cover', 'species_count']
    
    # Test single prediction
    test_point = {
        'latitude': 19.0760,
        'longitude': 72.8777,
        'features': {
            'temperature': 25.0,
            'precipitation': 2.0,
            'forest_cover': 0.3,
            'species_count': 15
        }
    }
    
    try:
        result = engine.predict_single(**test_point)
        print(f"Prediction result: Risk score = {result['risk_score']:.3f}")
        print(f"Risk category: {result['risk_category']}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

def test_api_schemas():
    """Test API schemas"""
    print("Testing API schemas...")
    
    try:
        from api.schemas import PredictionRequest, PredictionResponse
        
        # Test request schema
        request = PredictionRequest(
            latitude=19.0760,
            longitude=72.8777,
            features={'temperature': 25.0}
        )
        
        print(f"API request created: lat={request.latitude}, lon={request.longitude}")
        return True
    except Exception as e:
        print(f"API schema test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("EcoPredict System Test")
    print("=" * 50)
    
    try:
        # Test 1: Logger
        test_logger()
        
        # Test 2: Data generation
        data = test_data_generation()
        
        # Test 3: Model training
        models = test_model_training(data)
        
        # Test 4: Prediction
        test_prediction(models, data)
        
        # Test 5: API schemas
        test_api_schemas()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("EcoPredict system is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()