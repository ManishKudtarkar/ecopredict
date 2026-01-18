"""Model evaluation utilities for EcoPredict"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils.logger import get_logger
from ..models.base_model import BaseModel

logger = get_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, output_dir: str = "outputs/evaluation/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_regression_model(self, 
                                model: BaseModel,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                model_name: str = "model") -> Dict[str, float]:
        """Evaluate regression model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred)
        }
        
        # Additional metrics
        metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        metrics['explained_variance'] = 1 - np.var(y_test - y_pred) / np.var(y_test)
        
        logger.info(f"Model {model_name} - R²: {metrics['r2_score']:.4f}, "
                   f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def evaluate_classification_model(self,
                                    model: BaseModel,
                                    X_test: pd.DataFrame,
                                    y_test: pd.Series,
                                    model_name: str = "model") -> Dict[str, Any]:
        """Evaluate classification model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            logger.warning("Model does not support probability predictions")
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add AUC if probabilities available
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        logger.info(f"Model {model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def compare_models(self, 
                      models: Dict[str, BaseModel],
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      task_type: str = "regression") -> pd.DataFrame:
        """Compare multiple models
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test targets
            task_type: "regression" or "classification"
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models for {task_type}")
        
        results = []
        
        for model_name, model in models.items():
            try:
                if task_type == "regression":
                    metrics = self.evaluate_regression_model(model, X_test, y_test, model_name)
                else:
                    metrics = self.evaluate_classification_model(model, X_test, y_test, model_name)
                    # Flatten classification metrics for comparison
                    if 'classification_report' in metrics:
                        metrics.update(metrics['classification_report']['weighted avg'])
                        del metrics['classification_report']
                        del metrics['confusion_matrix']
                
                metrics['model_name'] = model_name
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
        
        comparison_df = pd.DataFrame(results)
        
        # Save comparison results
        output_path = self.output_dir / f"model_comparison_{task_type}.csv"
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved model comparison to {output_path}")
        
        return comparison_df
    
    def plot_regression_results(self,
                              model: BaseModel,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              model_name: str = "model") -> None:
        """Create regression evaluation plots
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
        """
        y_pred = model.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Regression Evaluation: {model_name}', fontsize=16)
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"regression_evaluation_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved regression plots to {plot_path}")
    
    def plot_feature_importance(self,
                              model: BaseModel,
                              feature_names: List[str],
                              model_name: str = "model",
                              top_n: int = 20) -> None:
        """Plot feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name for the model
            top_n: Number of top features to show
        """
        try:
            # Get feature importance
            if hasattr(model.model, 'feature_importances_'):
                importance = model.model.feature_importances_
            elif hasattr(model.model, 'coef_'):
                importance = np.abs(model.model.coef_)
            else:
                logger.warning(f"Model {model_name} does not support feature importance")
                return
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Feature Importance: {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"feature_importance_{model_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save data
            csv_path = self.output_dir / f"feature_importance_{model_name}.csv"
            importance_df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved feature importance to {plot_path} and {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to plot feature importance for {model_name}: {str(e)}")
    
    def generate_evaluation_report(self,
                                 models: Dict[str, BaseModel],
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 task_type: str = "regression") -> Dict[str, Any]:
        """Generate comprehensive evaluation report
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test targets
            task_type: "regression" or "classification"
            
        Returns:
            Dictionary with complete evaluation results
        """
        logger.info("Generating comprehensive evaluation report")
        
        # Compare models
        comparison_df = self.compare_models(models, X_test, y_test, task_type)
        
        # Generate plots for each model
        for model_name, model in models.items():
            try:
                if task_type == "regression":
                    self.plot_regression_results(model, X_test, y_test, model_name)
                
                self.plot_feature_importance(model, X_test.columns.tolist(), model_name)
                
            except Exception as e:
                logger.error(f"Failed to generate plots for {model_name}: {str(e)}")
        
        # Determine best model
        if task_type == "regression":
            best_model_name = comparison_df.loc[comparison_df['r2_score'].idxmax(), 'model_name']
            best_metric = comparison_df['r2_score'].max()
            metric_name = "R² Score"
        else:
            best_model_name = comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model_name']
            best_metric = comparison_df['accuracy'].max()
            metric_name = "Accuracy"
        
        report = {
            'best_model': best_model_name,
            'best_metric_value': best_metric,
            'metric_name': metric_name,
            'comparison_results': comparison_df.to_dict('records'),
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'test_set_size': len(X_test),
            'num_features': len(X_test.columns)
        }
        
        # Save report
        import json
        report_path = self.output_dir / f"evaluation_report_{task_type}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated evaluation report: Best model is {best_model_name} "
                   f"with {metric_name}: {best_metric:.4f}")
        logger.info(f"Report saved to {report_path}")
        
        return report