#!/usr/bin/env python3
"""Train machine learning models for EcoPredict"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.train import ModelTrainer, train_pipeline
from training.evaluate import ModelEvaluator
from training.cross_validation import CrossValidator
from preprocessing.clean_data import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.normalize import DataNormalizer
from utils.logger import get_logger
from utils.helpers import load_config, create_directories, save_results

logger = get_logger(__name__)


def prepare_training_data(config: dict) -> pd.DataFrame:
    """Prepare training data from raw sources"""
    logger.info("Preparing training data...")
    
    try:
        # Check if processed data exists
        processed_data_path = Path("data/processed/training_data.csv")
        
        if processed_data_path.exists():
            logger.info("Loading existing processed data")
            return pd.read_csv(processed_data_path)
        
        # Generate synthetic training data for demonstration
        logger.info("Generating synthetic training data")
        
        # Get region bounds
        bounds = config.get('region_bounds', [72.6, 15.6, 80.9, 22.0])
        n_samples = config.get('training_samples', 5000)
        
        # Generate features
        np.random.seed(42)
        
        data = {
            # Location features
            'latitude': np.random.uniform(bounds[1], bounds[3], n_samples),
            'longitude': np.random.uniform(bounds[0], bounds[2], n_samples),
            
            # Climate features
            'temperature': np.random.normal(25, 5, n_samples),
            'precipitation': np.random.exponential(2, n_samples),
            'humidity': np.random.normal(60, 15, n_samples),
            'wind_speed': np.random.exponential(3, n_samples),
            
            # Land use features
            'forest_cover': np.random.uniform(0, 1, n_samples),
            'agricultural_area': np.random.uniform(0, 1, n_samples),
            'urban_area': np.random.uniform(0, 1, n_samples),
            'water_bodies': np.random.uniform(0, 0.3, n_samples),
            
            # Species features
            'species_count': np.random.poisson(15, n_samples),
            'endemic_species': np.random.poisson(2, n_samples),
            'threatened_species': np.random.poisson(1, n_samples),
            'species_diversity': np.random.uniform(0, 3, n_samples),
            
            # Elevation and terrain
            'elevation': np.random.normal(500, 300, n_samples),
            'slope': np.random.exponential(5, n_samples),
            
            # Human impact
            'population_density': np.random.exponential(100, n_samples),
            'road_density': np.random.exponential(2, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate target variable (risk score) based on features
        # This is a simplified model for demonstration
        risk_score = (
            0.3 * (1 - df['forest_cover']) +  # Less forest = higher risk
            0.2 * df['urban_area'] +  # More urban = higher risk
            0.15 * (df['temperature'] - 25) / 10 +  # Extreme temp = higher risk
            0.1 * (1 / (df['species_count'] + 1)) +  # Less species = higher risk
            0.1 * df['population_density'] / 1000 +  # More people = higher risk
            0.15 * np.random.normal(0, 0.1, n_samples)  # Random noise
        )
        
        # Normalize risk score to 0-1 range
        df['risk_score'] = np.clip(risk_score, 0, 1)
        
        # Clean data
        cleaner = DataCleaner()
        df = cleaner.clean_dataframe(df)
        
        # Feature engineering
        engineer = FeatureEngineer()
        df = engineer.create_features(df)
        
        # Save processed data
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_data_path, index=False)
        
        logger.info(f"Training data prepared: {len(df)} samples, {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        raise


def train_models(data: pd.DataFrame, config: dict, output_dir: str):
    """Train all models"""
    logger.info("Starting model training...")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config_path="config/model_params.yaml")
        
        # Prepare data
        target_column = 'risk_score'
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            data, target_column, test_size=0.2, random_state=42
        )
        
        # Initialize models
        trainer.initialize_models()
        
        # Train all models
        trained_models = trainer.train_all_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X_test, y_test)
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model(evaluation_results)
        
        # Save all models
        models_dir = Path(output_dir) / "models"
        trainer.save_all_models(str(models_dir))
        
        # Save best model separately
        best_model_path = Path("models/trained/best_model.joblib")
        trainer.save_model(best_model_name, str(best_model_path))
        
        # Save evaluation results
        results_path = Path(output_dir) / "training_results.json"
        training_results = {
            'trained_models': list(trained_models.keys()),
            'evaluation_results': evaluation_results,
            'best_model': best_model_name,
            'data_shape': data.shape,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_names': X_train.columns.tolist()
        }
        
        save_results(training_results, str(results_path))
        
        logger.info(f"Model training completed. Best model: {best_model_name}")
        return training_results
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def evaluate_models(data: pd.DataFrame, config: dict, output_dir: str):
    """Comprehensive model evaluation"""
    logger.info("Starting comprehensive model evaluation...")
    
    try:
        # Load trained models
        models_dir = Path(output_dir) / "models"
        if not models_dir.exists():
            raise FileNotFoundError("No trained models found. Run training first.")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(output_dir=str(Path(output_dir) / "evaluation"))
        
        # Load models
        from joblib import load
        models = {}
        
        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            models[model_name] = load(model_file)
        
        if not models:
            raise FileNotFoundError("No model files found")
        
        # Prepare test data
        target_column = 'risk_score'
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data (use same split as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Generate comprehensive evaluation report
        evaluation_report = evaluator.generate_evaluation_report(
            models, X_test, y_test, task_type="regression"
        )
        
        logger.info("Model evaluation completed")
        return evaluation_report
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def cross_validate_models(data: pd.DataFrame, config: dict, output_dir: str):
    """Perform cross-validation"""
    logger.info("Starting cross-validation...")
    
    try:
        # Initialize cross-validator
        cv = CrossValidator(
            cv_folds=config.get('cv_folds', 5),
            random_state=42,
            output_dir=str(Path(output_dir) / "cross_validation")
        )
        
        # Prepare data
        target_column = 'risk_score'
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Initialize models for CV
        trainer = ModelTrainer(config_path="config/model_params.yaml")
        models = trainer.initialize_models()
        
        # Perform cross-validation comparison
        cv_results = cv.compare_models_cv(
            models, X, y, task_type="regression"
        )
        
        # Generate learning curves
        for model_name, model in models.items():
            try:
                learning_results = cv.learning_curve(
                    model, X, y, model_name, task_type="regression"
                )
                
                # Save learning curve results
                learning_path = Path(output_dir) / "cross_validation" / f"learning_curve_{model_name}.json"
                save_results(learning_results, str(learning_path))
                
            except Exception as e:
                logger.warning(f"Failed to generate learning curve for {model_name}: {e}")
        
        logger.info("Cross-validation completed")
        return cv_results
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train models for EcoPredict")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/training/",
        help="Output directory for training results"
    )
    parser.add_argument(
        "--data-path",
        help="Path to training data CSV file (optional)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "cv", "all"],
        default="all",
        help="Training mode"
    )
    parser.add_argument(
        "--target-column",
        default="risk_score",
        help="Target column name"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create output directory
        create_directories([args.output_dir])
        
        # Load or prepare data
        if args.data_path and Path(args.data_path).exists():
            logger.info(f"Loading data from {args.data_path}")
            data = pd.read_csv(args.data_path)
        else:
            data = prepare_training_data(config)
        
        # Execute based on mode
        if args.mode in ["train", "all"]:
            training_results = train_models(data, config, args.output_dir)
            print(f"Training completed. Best model: {training_results['best_model']}")
        
        if args.mode in ["evaluate", "all"]:
            evaluation_results = evaluate_models(data, config, args.output_dir)
            print(f"Evaluation completed. Check {args.output_dir}/evaluation/ for results")
        
        if args.mode in ["cv", "all"]:
            cv_results = cross_validate_models(data, config, args.output_dir)
            print(f"Cross-validation completed. Check {args.output_dir}/cross_validation/ for results")
        
        print(f"All results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()