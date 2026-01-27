#!/usr/bin/env python3
"""
Main pipeline script for EcoPredict
Orchestrates data loading, preprocessing, training, and evaluation
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logger import get_logger
from utils.helpers import load_config
from ingestion.climate_loader import ClimateDataLoader
from ingestion.landuse_loader import LandUseDataLoader
from ingestion.species_loader import SpeciesDataLoader
from preprocessing.clean_data import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.normalize import DataNormalizer
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.regression import LinearRegressionModel
from training.train import ModelTrainer
from training.evaluate import ModelEvaluator

logger = get_logger(__name__)


def load_data(config: dict) -> pd.DataFrame:
    """Load and combine all data sources"""
    
    logger.info("Loading data from all sources")
    
    # Define bounds for synthetic data generation
    bounds = (72.0, 15.0, 80.0, 22.0)  # Maharashtra approximate bounds
    
    # Load climate data
    climate_loader = ClimateDataLoader()
    climate_data = climate_loader.generate_synthetic_data(bounds, resolution=0.1, num_days=365)
    climate_monthly = climate_loader.aggregate_monthly(climate_data)
    
    # Load land use data
    landuse_loader = LandUseDataLoader()
    landuse_data = landuse_loader.generate_synthetic_data(bounds, resolution=0.1)
    
    # Load species data
    species_loader = SpeciesDataLoader()
    species_data = species_loader.generate_synthetic_data(bounds, num_records=5000)
    
    # Create grid for analysis
    from utils.helpers import create_grid
    grid_points = create_grid(bounds, resolution=0.1)
    
    # Calculate species metrics for grid points
    species_metrics = species_loader.calculate_species_metrics(
        species_data, grid_points, radius_km=10.0
    )
    
    # Merge all data sources
    # Start with climate data (monthly aggregated)
    combined_data = climate_monthly.copy()
    
    # Merge with land use data
    combined_data = pd.merge(
        combined_data, landuse_data,
        on=['latitude', 'longitude'],
        how='left'
    )
    
    # Merge with species metrics
    combined_data = pd.merge(
        combined_data, species_metrics,
        on=['latitude', 'longitude'],
        how='left'
    )
    
    # Fill missing values
    combined_data = combined_data.fillna(0)
    
    logger.info(f"Combined dataset shape: {combined_data.shape}")
    return combined_data


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create ecological risk target variable"""
    
    logger.info("Creating target variable (ecological risk)")
    
    # Create a composite risk score based on multiple factors
    risk_components = []
    
    # Climate stress component
    if 'temperature' in df.columns and 'precipitation' in df.columns:
        temp_stress = np.abs(df['temperature'] - df['temperature'].median()) / df['temperature'].std()
        precip_stress = np.abs(df['precipitation'] - df['precipitation'].median()) / df['precipitation'].std()
        climate_risk = (temp_stress + precip_stress) / 2
        risk_components.append(climate_risk * 0.3)
    
    # Habitat fragmentation component
    if 'forest_cover' in df.columns and 'urban_area' in df.columns:
        fragmentation_risk = df['urban_area'] / (df['forest_cover'] + 0.01)
        fragmentation_risk = (fragmentation_risk - fragmentation_risk.min()) / (fragmentation_risk.max() - fragmentation_risk.min())
        risk_components.append(fragmentation_risk * 0.25)
    
    # Species threat component
    if 'threatened_species' in df.columns and 'species_count' in df.columns:
        threat_ratio = df['threatened_species'] / (df['species_count'] + 0.01)
        threat_ratio = (threat_ratio - threat_ratio.min()) / (threat_ratio.max() - threat_ratio.min())
        risk_components.append(threat_ratio * 0.25)
    
    # Biodiversity loss component
    if 'species_diversity' in df.columns:
        # Inverse of diversity (higher diversity = lower risk)
        diversity_risk = 1 - ((df['species_diversity'] - df['species_diversity'].min()) / 
                             (df['species_diversity'].max() - df['species_diversity'].min()))
        risk_components.append(diversity_risk * 0.2)
    
    # Combine all components
    if risk_components:
        ecological_risk = sum(risk_components)
        # Normalize to 0-1 range
        ecological_risk = (ecological_risk - ecological_risk.min()) / (ecological_risk.max() - ecological_risk.min())
    else:
        # Fallback: random risk scores
        ecological_risk = np.random.beta(2, 5, len(df))  # Skewed towards lower risk
    
    df['ecological_risk'] = ecological_risk
    
    logger.info(f"Target variable created - Risk range: [{ecological_risk.min():.3f}, {ecological_risk.max():.3f}]")
    return df


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Preprocess the combined dataset"""
    
    logger.info("Starting data preprocessing")
    
    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean_dataset(df, target_column='ecological_risk')
    
    # Feature engineering
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df_clean, target_column='ecological_risk')
    
    # Normalize features
    normalizer = DataNormalizer()
    
    # Get numeric columns (excluding target and coordinates)
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['ecological_risk', 'latitude', 'longitude']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    df_normalized = normalizer.normalize_features(
        df_engineered, 
        method='standard',
        feature_columns=feature_cols
    )
    
    logger.info(f"Preprocessing completed - Final shape: {df_normalized.shape}")
    return df_normalized


def train_models(df: pd.DataFrame, config: dict) -> dict:
    """Train multiple models and return results"""
    
    logger.info("Training models")
    
    # Prepare features and target
    target_col = 'ecological_risk'
    feature_cols = [col for col in df.columns if col not in [target_col, 'latitude', 'longitude']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize models
    models = {}
    
    # Random Forest
    rf_params = config.get('random_forest', {})
    models['random_forest'] = RandomForestModel(**rf_params)
    
    # XGBoost (if available)
    try:
        xgb_params = config.get('xgboost', {})
        models['xgboost'] = XGBoostModel(**xgb_params)
    except ImportError:
        logger.warning("XGBoost not available, skipping")
    
    # Linear Regression
    models['linear_regression'] = LinearRegressionModel(model_type='ridge', alpha=1.0)
    
    # Train all models
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}")
        
        try:
            # Train model
            model.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            # Evaluate
            train_metrics = model.evaluate(X_train, y_train)
            val_metrics = model.evaluate(X_val, y_val)
            test_metrics = model.evaluate(X_test, y_test)
            
            # Store results
            trained_models[name] = model
            results[name] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_importance': model.get_feature_importance()
            }
            
            logger.info(f"{name} - Test R²: {test_metrics['r2']:.3f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    return trained_models, results


def save_results(models: dict, results: dict, config: dict):
    """Save trained models and results"""
    
    logger.info("Saving models and results")
    
    # Create output directories
    models_dir = Path(config['paths']['models'])
    outputs_dir = Path(config['paths']['outputs'])
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for name, model in models.items():
        model_path = models_dir / f"{name}_model.pkl"
        model.save(str(model_path))
    
    # Save results summary
    import json
    results_path = outputs_dir / "training_results.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON serialization
    clean_results = {}
    for model_name, model_results in results.items():
        clean_results[model_name] = {}
        for metric_type, metrics in model_results.items():
            if isinstance(metrics, dict):
                clean_results[model_name][metric_type] = {
                    k: convert_numpy(v) for k, v in metrics.items()
                }
            else:
                clean_results[model_name][metric_type] = convert_numpy(metrics)
    
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Run EcoPredict pipeline')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load data
        data = load_data(config)
        
        # Create target variable
        data = create_target_variable(data)
        
        # Preprocess data
        processed_data = preprocess_data(data, config)
        
        # Save processed data
        processed_path = Path(config['paths']['processed_data']) / "processed_dataset.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        if not args.skip_training:
            # Train models
            trained_models, results = train_models(processed_data, config)
            
            # Save results
            save_results(trained_models, results, config)
            
            # Print summary
            print("\n" + "="*50)
            print("TRAINING RESULTS SUMMARY")
            print("="*50)
            
            for model_name, model_results in results.items():
                test_r2 = model_results['test_metrics']['r2']
                test_rmse = model_results['test_metrics']['rmse']
                print(f"{model_name:20s} - R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()