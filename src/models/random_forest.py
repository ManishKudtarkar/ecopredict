"""Random Forest model implementation for EcoPredict"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

from .base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest model for ecological risk prediction"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        
        super().__init__("RandomForest", **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def _create_model(self) -> BaseEstimator:
        """Create Random Forest model"""
        
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the trees in the forest"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained to get tree info")
        
        trees = self.model.estimators_
        
        tree_info = {
            'n_trees': len(trees),
            'tree_depths': [tree.tree_.max_depth for tree in trees],
            'tree_nodes': [tree.tree_.node_count for tree in trees],
            'tree_leaves': [tree.tree_.n_leaves for tree in trees]
        }
        
        # Calculate statistics
        tree_info['avg_depth'] = np.mean(tree_info['tree_depths'])
        tree_info['avg_nodes'] = np.mean(tree_info['tree_nodes'])
        tree_info['avg_leaves'] = np.mean(tree_info['tree_leaves'])
        
        return tree_info
    
    def partial_dependence(self, 
                          X: pd.DataFrame, 
                          feature_names: list,
                          grid_resolution: int = 100) -> Dict[str, Any]:
        """
        Calculate partial dependence for specified features
        
        Args:
            X: Input features
            feature_names: Features to calculate partial dependence for
            grid_resolution: Number of points in the grid
            
        Returns:
            Partial dependence data
        """
        from sklearn.inspection import partial_dependence
        
        if not self.is_trained:
            raise ValueError("Model must be trained for partial dependence")
        
        # Get feature indices
        feature_indices = [self.feature_names.index(name) for name in feature_names]
        
        # Calculate partial dependence
        pd_results = partial_dependence(
            self.model, X, feature_indices, 
            grid_resolution=grid_resolution,
            kind='average'
        )
        
        results = {
            'partial_dependence': pd_results['average'],
            'grid_values': pd_results['grid_values'],
            'feature_names': feature_names
        }
        
        return results
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available"""
        
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'oob_score_'):
            return float(self.model.oob_score_)
        else:
            logger.warning("OOB score not available. Set oob_score=True when creating model.")
            return None