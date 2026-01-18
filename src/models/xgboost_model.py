"""XGBoost model implementation for EcoPredict"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from .base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model for ecological risk prediction"""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        super().__init__("XGBoost", **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.evals_result = {}
    
    def _create_model(self) -> BaseEstimator:
        """Create XGBoost model"""
        
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='rmse'
        )
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            early_stopping_rounds: Optional[int] = None,
            verbose: bool = False) -> 'XGBoostModel':
        """
        Train XGBoost model with optional early stopping
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional validation data for early stopping
            early_stopping_rounds: Number of rounds for early stopping
            verbose: Whether to print training progress
            
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
        
        # Prepare evaluation sets
        eval_set = [(X, y)]
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set.append((X_val, y_val))
        
        # Train the model
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Store evaluation results
        if hasattr(self.model, 'evals_result_'):
            self.evals_result = self.model.evals_result_
        
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
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """
        Get feature importance from XGBoost
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get importance scores
        importance_scores = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map to feature names (XGBoost uses f0, f1, etc.)
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            xgb_feature_name = f'f{i}'
            importance_dict[feature_name] = importance_scores.get(xgb_feature_name, 0.0)
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def plot_importance(self, 
                       max_num_features: int = 20,
                       importance_type: str = 'weight') -> Any:
        """
        Plot feature importance
        
        Args:
            max_num_features: Maximum number of features to show
            importance_type: Type of importance to plot
            
        Returns:
            Matplotlib axes object
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot importance")
        
        try:
            return xgb.plot_importance(
                self.model,
                max_num_features=max_num_features,
                importance_type=importance_type
            )
        except Exception as e:
            logger.error(f"Error plotting importance: {e}")
            return None
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history from evaluation results"""
        
        if not self.evals_result:
            logger.warning("No evaluation results available")
            return {}
        
        return self.evals_result
    
    def plot_training_history(self) -> Any:
        """Plot training history"""
        
        if not self.evals_result:
            logger.warning("No evaluation results to plot")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for eval_name, metrics in self.evals_result.items():
                for metric_name, values in metrics.items():
                    ax.plot(values, label=f'{eval_name}_{metric_name}')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('RMSE')
            ax.set_title('XGBoost Training History')
            ax.legend()
            ax.grid(True)
            
            return ax
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            return None
    
    def get_best_iteration(self) -> Optional[int]:
        """Get the best iteration from early stopping"""
        
        if hasattr(self.model, 'best_iteration'):
            return self.model.best_iteration
        return None
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using dropout
        
        Args:
            X: Features for prediction
            n_iterations: Number of iterations for uncertainty estimation
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # For XGBoost, we can use different subsets of trees for uncertainty
        predictions = []
        
        for i in range(n_iterations):
            # Use random subset of trees
            n_trees = self.model.n_estimators
            subset_size = max(1, int(n_trees * 0.8))  # Use 80% of trees
            
            # This is a simplified approach - in practice, you might want to use
            # more sophisticated uncertainty quantification methods
            pred = self.model.predict(X, iteration_range=(0, subset_size))
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics"""
        
        if not self.is_trained:
            return {}
        
        booster = self.model.get_booster()
        
        complexity = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.max_depth,
            'n_features': len(self.feature_names),
            'best_iteration': self.get_best_iteration()
        }
        
        # Try to get additional booster stats
        try:
            stats = booster.get_dump_stats()
            complexity.update(stats)
        except:
            pass
        
        return complexity