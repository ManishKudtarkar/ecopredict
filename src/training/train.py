"""Model training utilities for EcoPredict"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import yaml

from ..utils.logger import get_logger
from ..models.base_model import BaseModel
from ..models.random_forest import RandomForestModel
from ..models.xgboost_model import XGBoostModel
from ..models.regression import LinearRegressionModel

logger = get_logger(__name__)


class ModelTrainer:
    """Handles model training and management"""
    
    def __init__(self, config_path: str = "config/model_params.yaml"):
        """Initialize the model trainer
        
        Args:
            config_path: Path to model configuration file
        """
        self.config_path = config_path
        self.models = {}
        self.trained_models = {}
        self.load_config()
        
    def load_config(self):
        """Load model configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded model configuration from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'linear_regression': {
                'fit_intercept': True,
                'normalize': False
            }
        }
    
    def prepare_data(self, 
                    data: pd.DataFrame, 
                    target_column: str,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training
        
        Args:
            data: Input dataframe
            target_column: Name of target column
            test_size: Fraction of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing data with {len(data)} samples")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self) -> Dict[str, BaseModel]:
        """Initialize all available models"""
        models = {}
        
        # Random Forest
        if 'random_forest' in self.config:
            models['random_forest'] = RandomForestModel(**self.config['random_forest'])
            
        # XGBoost
        if 'xgboost' in self.config:
            models['xgboost'] = XGBoostModel(**self.config['xgboost'])
            
        # Linear Regression
        if 'linear_regression' in self.config:
            models['linear_regression'] = LinearRegressionModel(**self.config['linear_regression'])
        
        self.models = models
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models
    
    def train_model(self, 
                   model_name: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None) -> BaseModel:
        """Train a specific model
        
        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        logger.info(f"Training {model_name} model...")
        model = self.models[model_name]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Validate if validation data provided
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"{model_name} validation score: {val_score:.4f}")
        
        self.trained_models[model_name] = model
        logger.info(f"Successfully trained {model_name}")
        
        return model
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None) -> Dict[str, BaseModel]:
        """Train all available models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training all models...")
        
        if not self.models:
            self.initialize_models()
        
        results = {}
        for model_name in self.models:
            try:
                model = self.train_model(model_name, X_train, y_train, X_val, y_val)
                results[model_name] = model
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
        
        logger.info(f"Successfully trained {len(results)} models")
        return results
    
    def evaluate_models(self,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of model evaluation metrics
        """
        if not self.trained_models:
            raise ValueError("No trained models found. Train models first.")
        
        logger.info("Evaluating models...")
        results = {}
        
        for model_name, model in self.trained_models.items():
            try:
                metrics = model.evaluate(X_test, y_test)
                results[model_name] = metrics
                logger.info(f"{model_name} - R²: {metrics.get('r2', 'N/A'):.4f}, "
                           f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
        
        return results
    
    def get_best_model(self, 
                      evaluation_results: Dict[str, Dict[str, float]],
                      metric: str = 'r2') -> Tuple[str, BaseModel]:
        """Get the best performing model
        
        Args:
            evaluation_results: Results from evaluate_models
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model)
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        # Find best model based on metric
        best_score = float('-inf') if metric in ['r2', 'accuracy'] else float('inf')
        best_model_name = None
        
        for model_name, metrics in evaluation_results.items():
            if metric not in metrics:
                continue
                
            score = metrics[metric]
            if metric in ['r2', 'accuracy'] and score > best_score:
                best_score = score
                best_model_name = model_name
            elif metric in ['rmse', 'mae'] and score < best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"No models found with metric {metric}")
        
        logger.info(f"Best model: {best_model_name} ({metric}: {best_score:.4f})")
        return best_model_name, self.trained_models[best_model_name]
    
    def save_model(self, 
                  model_name: str,
                  save_path: str) -> None:
        """Save a trained model
        
        Args:
            model_name: Name of model to save
            save_path: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, save_path)
        logger.info(f"Saved {model_name} to {save_path}")
    
    def save_all_models(self, save_dir: str = "models/trained/") -> None:
        """Save all trained models
        
        Args:
            save_dir: Directory to save models
        """
        if not self.trained_models:
            logger.warning("No trained models to save")
            return
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            save_path = Path(save_dir) / f"{model_name}.joblib"
            self.save_model(model_name, str(save_path))
        
        logger.info(f"Saved {len(self.trained_models)} models to {save_dir}")
    
    def load_model(self, model_path: str) -> BaseModel:
        """Load a saved model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise


def train_pipeline(data_path: str,
                  target_column: str,
                  config_path: str = "config/model_params.yaml",
                  save_dir: str = "models/trained/") -> Dict[str, Any]:
    """Complete training pipeline
    
    Args:
        data_path: Path to training data
        target_column: Name of target column
        config_path: Path to model configuration
        save_dir: Directory to save trained models
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting training pipeline...")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Initialize trainer
    trainer = ModelTrainer(config_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(data, target_column)
    
    # Initialize and train models
    trainer.initialize_models()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model(evaluation_results)
    
    # Save models
    trainer.save_all_models(save_dir)
    
    results = {
        'trained_models': list(trained_models.keys()),
        'evaluation_results': evaluation_results,
        'best_model': best_model_name,
        'data_shape': data.shape,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    logger.info("Training pipeline completed successfully")
    return results


if __name__ == "__main__":
    # Example usage
    results = train_pipeline(
        data_path="data/processed/training_data.csv",
        target_column="risk_score"
    )
    print("Training Results:", results)