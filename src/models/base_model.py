"""Base model class for EcoPredict"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.helpers import save_model, load_model

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.target_name = None
        self.model_params = kwargs
        self.training_metrics = {}
        self.validation_metrics = {}
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying ML model"""
        pass
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional validation data (X_val, y_val)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name} model")
        
        # Store feature and target names
        self.feature_names = list(X.columns)
        self.target_name = y.name if hasattr(y, 'name') else 'target'
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X)
        self.training_metrics = self._calculate_metrics(y, y_pred_train)
        
        # Calculate validation metrics if validation data provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_pred_val = self.model.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, y_pred_val)
            
            logger.info(f"Training completed - Train R²: {self.training_metrics['r2']:.3f}, "
                       f"Val R²: {self.validation_metrics['r2']:.3f}")
        else:
            logger.info(f"Training completed - Train R²: {self.training_metrics['r2']:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate features
        self._validate_features(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification models)
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support probability predictions")
        
        self._validate_features(X)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            logger.warning("Model does not provide feature importance")
            return {}
        
        return dict(zip(self.feature_names, importances))
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        return self._calculate_metrics(y, y_pred)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'is_trained': self.is_trained
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_data['model_name'], **model_data['model_params'])
        
        # Restore state
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.target_name = model_data['target_name']
        instance.training_metrics = model_data['training_metrics']
        instance.validation_metrics = model_data['validation_metrics']
        instance.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__ if self.model else None,
            'is_trained': self.is_trained,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_params': self.model_params,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }
        
        # Add feature importance if available
        if self.is_trained:
            try:
                info['feature_importance'] = self.get_feature_importance()
            except:
                info['feature_importance'] = {}
        
        return info
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
        
        # Add MAPE (Mean Absolute Percentage Error)
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = float(mape)
        
        return metrics
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """Validate that features match training features"""
        
        if not self.feature_names:
            return
        
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
        
        # Reorder columns to match training order
        X = X[self.feature_names]
    
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      cv: int = 5,
                      scoring: str = 'r2') -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import cross_val_score, cross_validate
        
        if self.model is None:
            self.model = self._create_model()
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model, X, y, 
            cv=cv, 
            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=True
        )
        
        # Process results
        results = {
            'test_r2_mean': cv_results['test_r2'].mean(),
            'test_r2_std': cv_results['test_r2'].std(),
            'train_r2_mean': cv_results['train_r2'].mean(),
            'train_r2_std': cv_results['train_r2'].std(),
            'test_mse_mean': -cv_results['test_neg_mean_squared_error'].mean(),
            'test_mse_std': cv_results['test_neg_mean_squared_error'].std(),
            'test_mae_mean': -cv_results['test_neg_mean_absolute_error'].mean(),
            'test_mae_std': cv_results['test_neg_mean_absolute_error'].std(),
            'cv_scores': cv_results
        }
        
        logger.info(f"Cross-validation completed: R² = {results['test_r2_mean']:.3f} ± {results['test_r2_std']:.3f}")
        
        return results
    
    def learning_curve(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      train_sizes: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate learning curve data
        
        Args:
            X: Features
            y: Target
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Learning curve data
        """
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        if self.model is None:
            self.model = self._create_model()
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
        
        return results
    
    def __str__(self) -> str:
        """String representation of the model"""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name} ({status})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', is_trained={self.is_trained})"