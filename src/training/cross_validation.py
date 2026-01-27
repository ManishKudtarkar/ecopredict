"""Cross-validation utilities for EcoPredict"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils.logger import get_logger
from ..models.base_model import BaseModel

logger = get_logger(__name__)


class CrossValidator:
    """Cross-validation for model selection and evaluation"""
    
    def __init__(self, 
                 cv_folds: int = 5,
                 random_state: int = 42,
                 output_dir: str = "outputs/cross_validation/"):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def regression_cv(self,
                     model: BaseModel,
                     X: pd.DataFrame,
                     y: pd.Series,
                     cv_type: str = "kfold") -> Dict[str, Any]:
        """Perform cross-validation for regression models
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target variable
            cv_type: Type of CV ("kfold", "stratified", "timeseries")
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_type} cross-validation for regression")
        
        # Select CV strategy
        if cv_type == "kfold":
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        elif cv_type == "timeseries":
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            raise ValueError(f"Unsupported CV type for regression: {cv_type}")
        
        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model.model, X, y, 
            cv=cv, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        results = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            # Convert negative scores back to positive
            if metric.startswith('neg_'):
                test_scores = -test_scores
                train_scores = -train_scores
                metric_name = metric[4:]  # Remove 'neg_' prefix
            else:
                metric_name = metric
            
            results[f'{metric_name}_test_mean'] = np.mean(test_scores)
            results[f'{metric_name}_test_std'] = np.std(test_scores)
            results[f'{metric_name}_train_mean'] = np.mean(train_scores)
            results[f'{metric_name}_train_std'] = np.std(train_scores)
            results[f'{metric_name}_test_scores'] = test_scores.tolist()
            results[f'{metric_name}_train_scores'] = train_scores.tolist()
        
        # Add metadata
        results['cv_folds'] = self.cv_folds
        results['cv_type'] = cv_type
        results['sample_size'] = len(X)
        
        logger.info(f"CV Results - R² Test: {results['r2_test_mean']:.4f} ± {results['r2_test_std']:.4f}")
        
        return results
    
    def classification_cv(self,
                         model: BaseModel,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv_type: str = "stratified") -> Dict[str, Any]:
        """Perform cross-validation for classification models
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target variable
            cv_type: Type of CV ("kfold", "stratified")
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_type} cross-validation for classification")
        
        # Select CV strategy
        if cv_type == "kfold":
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        elif cv_type == "stratified":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported CV type for classification: {cv_type}")
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y)) == 2:
            scoring['roc_auc'] = 'roc_auc'
        
        # Perform cross-validation
        cv_results = cross_validate(
            model.model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        results = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[f'{metric}_test_mean'] = np.mean(test_scores)
            results[f'{metric}_test_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_train_std'] = np.std(train_scores)
            results[f'{metric}_test_scores'] = test_scores.tolist()
            results[f'{metric}_train_scores'] = train_scores.tolist()
        
        # Add metadata
        results['cv_folds'] = self.cv_folds
        results['cv_type'] = cv_type
        results['sample_size'] = len(X)
        results['num_classes'] = len(np.unique(y))
        
        logger.info(f"CV Results - Accuracy Test: {results['accuracy_test_mean']:.4f} ± {results['accuracy_test_std']:.4f}")
        
        return results
    
    def compare_models_cv(self,
                         models: Dict[str, BaseModel],
                         X: pd.DataFrame,
                         y: pd.Series,
                         task_type: str = "regression",
                         cv_type: str = "auto") -> pd.DataFrame:
        """Compare multiple models using cross-validation
        
        Args:
            models: Dictionary of model_name -> model
            X: Features
            y: Target variable
            task_type: "regression" or "classification"
            cv_type: CV type ("auto" for automatic selection)
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models using cross-validation")
        
        # Auto-select CV type
        if cv_type == "auto":
            if task_type == "regression":
                cv_type = "kfold"
            else:
                cv_type = "stratified"
        
        results = []
        
        for model_name, model in models.items():
            try:
                logger.info(f"Cross-validating {model_name}")
                
                if task_type == "regression":
                    cv_results = self.regression_cv(model, X, y, cv_type)
                    primary_metric = 'r2_test_mean'
                else:
                    cv_results = self.classification_cv(model, X, y, cv_type)
                    primary_metric = 'accuracy_test_mean'
                
                cv_results['model_name'] = model_name
                cv_results['primary_metric'] = cv_results[primary_metric]
                results.append(cv_results)
                
            except Exception as e:
                logger.error(f"Failed to cross-validate {model_name}: {str(e)}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by primary metric
        comparison_df = comparison_df.sort_values('primary_metric', ascending=False)
        
        # Save results
        output_path = self.output_dir / f"cv_comparison_{task_type}.csv"
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Saved CV comparison to {output_path}")
        
        return comparison_df
    
    def plot_cv_results(self,
                       cv_results: Dict[str, Any],
                       model_name: str,
                       task_type: str = "regression") -> None:
        """Plot cross-validation results
        
        Args:
            cv_results: Results from CV
            model_name: Name of the model
            task_type: "regression" or "classification"
        """
        if task_type == "regression":
            metrics = ['r2', 'mse', 'mae']
        else:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            if 'roc_auc_test_scores' in cv_results:
                metrics.append('roc_auc')
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        fig.suptitle(f'Cross-Validation Results: {model_name}', fontsize=16)
        
        for i, metric in enumerate(metrics):
            test_key = f'{metric}_test_scores'
            train_key = f'{metric}_train_scores'
            
            if test_key in cv_results and train_key in cv_results:
                test_scores = cv_results[test_key]
                train_scores = cv_results[train_key]
                
                # Box plot
                data_to_plot = [train_scores, test_scores]
                axes[i].boxplot(data_to_plot, labels=['Train', 'Test'])
                axes[i].set_title(f'{metric.upper()}')
                axes[i].set_ylabel('Score')
                
                # Add mean lines
                axes[i].axhline(y=np.mean(train_scores), color='blue', linestyle='--', alpha=0.7)
                axes[i].axhline(y=np.mean(test_scores), color='orange', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"cv_results_{model_name}_{task_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved CV plots to {plot_path}")
    
    def learning_curve(self,
                      model: BaseModel,
                      X: pd.DataFrame,
                      y: pd.Series,
                      model_name: str,
                      task_type: str = "regression",
                      train_sizes: Optional[List[float]] = None) -> Dict[str, Any]:
        """Generate learning curves
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target variable
            model_name: Name of the model
            task_type: "regression" or "classification"
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Dictionary with learning curve results
        """
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        logger.info(f"Generating learning curve for {model_name}")
        
        # Select scoring metric
        if task_type == "regression":
            scoring = 'r2'
        else:
            scoring = 'accuracy'
        
        # Generate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model.model, X, y,
            train_sizes=train_sizes,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring.upper()} Score')
        plt.title(f'Learning Curve: {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.output_dir / f"learning_curve_{model_name}_{task_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist(),
            'scoring_metric': scoring
        }
        
        logger.info(f"Generated learning curve for {model_name}, saved to {plot_path}")
        
        return results