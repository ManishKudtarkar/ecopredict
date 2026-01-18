"""Linear regression model implementation for EcoPredict"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from .base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LinearRegressionModel(BaseModel):
    """Linear regression model for ecological risk prediction"""
    
    def __init__(self,
                 model_type: str = 'linear',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 polynomial_degree: Optional[int] = None,
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 **kwargs):
        
        super().__init__(f"{model_type.title()}Regression", **kwargs)
        
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.polynomial_degree = polynomial_degree
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        
        # Validate model type
        valid_types = ['linear', 'ridge', 'lasso', 'elastic_net']
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
    
    def _create_model(self) -> BaseEstimator:
        """Create linear regression model"""
        
        # Choose base model
        if self.model_type == 'linear':
            base_model = LinearRegression(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize
            )
        elif self.model_type == 'ridge':
            base_model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize
            )
        elif self.model_type == 'lasso':
            base_model = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=1000
            )
        elif self.model_type == 'elastic_net':
            base_model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                max_iter=1000
            )
        
        # Add polynomial features if specified
        if self.polynomial_degree and self.polynomial_degree > 1:
            poly_features = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False
            )
            model = Pipeline([
                ('poly', poly_features),
                ('regressor', base_model)
            ])
        else:
            model = base_model
        
        return model
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Get model coefficients
        
        Returns:
            Dictionary mapping feature names to coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get coefficients")
        
        # Handle pipeline vs direct model
        if hasattr(self.model, 'named_steps'):
            # Pipeline with polynomial features
            regressor = self.model.named_steps['regressor']
            poly_transformer = self.model.named_steps['poly']
            
            # Get polynomial feature names
            poly_feature_names = poly_transformer.get_feature_names_out(self.feature_names)
            coefficients = regressor.coef_
            
            coef_dict = dict(zip(poly_feature_names, coefficients))
        else:
            # Direct model
            coefficients = self.model.coef_
            coef_dict = dict(zip(self.feature_names, coefficients))
        
        return coef_dict
    
    def get_intercept(self) -> float:
        """Get model intercept"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained to get intercept")
        
        if hasattr(self.model, 'named_steps'):
            return float(self.model.named_steps['regressor'].intercept_)
        else:
            return float(self.model.intercept_)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on absolute coefficient values"""
        
        coefficients = self.get_coefficients()
        
        # For polynomial features, aggregate by original feature
        if self.polynomial_degree and self.polynomial_degree > 1:
            importance_dict = {}
            
            for feature in self.feature_names:
                # Sum absolute coefficients for all polynomial terms involving this feature
                total_importance = 0
                for poly_feature, coef in coefficients.items():
                    if feature in poly_feature:
                        total_importance += abs(coef)
                importance_dict[feature] = total_importance
        else:
            # Direct mapping for linear features
            importance_dict = {name: abs(coef) for name, coef in coefficients.items()}
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def get_equation(self, precision: int = 3) -> str:
        """
        Get the regression equation as a string
        
        Args:
            precision: Number of decimal places
            
        Returns:
            String representation of the equation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get equation")
        
        coefficients = self.get_coefficients()
        intercept = self.get_intercept()
        
        # Build equation string
        terms = []
        
        for feature, coef in coefficients.items():
            if abs(coef) > 10**(-precision):  # Skip very small coefficients
                coef_str = f"{coef:.{precision}f}"
                if coef > 0 and terms:  # Add + for positive coefficients (except first)
                    terms.append(f"+ {coef_str} * {feature}")
                else:
                    terms.append(f"{coef_str} * {feature}")
        
        # Add intercept
        if abs(intercept) > 10**(-precision):
            intercept_str = f"{intercept:.{precision}f}"
            if intercept > 0 and terms:
                terms.append(f"+ {intercept_str}")
            else:
                terms.append(intercept_str)
        
        equation = f"y = {' '.join(terms)}"
        return equation
    
    def predict_with_confidence(self, 
                              X: pd.DataFrame,
                              confidence_level: float = 0.95) -> tuple:
        """
        Make predictions with confidence intervals
        
        Args:
            X: Features for prediction
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.predict(X)
        
        # For linear regression, we can calculate confidence intervals
        # This is a simplified implementation
        try:
            from scipy import stats
            
            # Calculate residual standard error from training data
            if hasattr(self, '_training_residuals'):
                residual_std = np.std(self._training_residuals)
            else:
                # Estimate from model if training residuals not available
                residual_std = np.sqrt(self.training_metrics.get('mse', 1.0))
            
            # Calculate confidence intervals (simplified)
            alpha = 1 - confidence_level
            t_value = stats.t.ppf(1 - alpha/2, df=len(X) - len(self.feature_names) - 1)
            
            margin_of_error = t_value * residual_std
            lower_bounds = predictions - margin_of_error
            upper_bounds = predictions + margin_of_error
            
            return predictions, lower_bounds, upper_bounds
            
        except ImportError:
            logger.warning("scipy not available for confidence intervals")
            return predictions, predictions, predictions
        except Exception as e:
            logger.warning(f"Error calculating confidence intervals: {e}")
            return predictions, predictions, predictions
    
    def get_residuals_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze model residuals
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Dictionary with residual analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for residual analysis")
        
        predictions = self.predict(X)
        residuals = y - predictions
        
        analysis = {
            'residuals': residuals,
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'min_residual': float(np.min(residuals)),
            'max_residual': float(np.max(residuals)),
            'residual_sum_squares': float(np.sum(residuals**2))
        }
        
        # Test for normality of residuals
        try:
            from scipy import stats
            _, p_value = stats.normaltest(residuals)
            analysis['normality_test_p_value'] = float(p_value)
            analysis['residuals_normal'] = p_value > 0.05
        except ImportError:
            pass
        
        return analysis
    
    def regularization_path(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          alphas: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute regularization path for Ridge/Lasso models
        
        Args:
            X: Training features
            y: Training target
            alphas: List of alpha values to try
            
        Returns:
            Dictionary with regularization path data
        """
        if self.model_type not in ['ridge', 'lasso', 'elastic_net']:
            raise ValueError("Regularization path only available for regularized models")
        
        if alphas is None:
            alphas = np.logspace(-4, 2, 50)
        
        coefficients = []
        scores = []
        
        for alpha in alphas:
            # Create temporary model with this alpha
            temp_model = self._create_model()
            
            # Update alpha
            if hasattr(temp_model, 'named_steps'):
                temp_model.named_steps['regressor'].alpha = alpha
            else:
                temp_model.alpha = alpha
            
            # Fit and evaluate
            temp_model.fit(X, y)
            
            if hasattr(temp_model, 'named_steps'):
                coef = temp_model.named_steps['regressor'].coef_
            else:
                coef = temp_model.coef_
            
            coefficients.append(coef)
            scores.append(temp_model.score(X, y))
        
        return {
            'alphas': alphas,
            'coefficients': np.array(coefficients),
            'scores': scores,
            'feature_names': self.feature_names
        }