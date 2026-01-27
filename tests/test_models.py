"""Tests for EcoPredict models"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.base_model import BaseModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.regression import LinearRegressionModel
from training.train import ModelTrainer
from training.evaluate import ModelEvaluator
from training.cross_validation import CrossValidator


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data"""
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data"""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_informative=8,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    return X_df, y_series


class TestBaseModel:
    """Test BaseModel functionality"""
    
    def test_base_model_abstract(self):
        """Test that BaseModel cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseModel()


class TestRandomForestModel:
    """Test RandomForestModel"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = RandomForestModel(n_estimators=10, random_state=42)
        assert model.n_estimators == 10
        assert model.random_state == 42
    
    def test_fit_predict_regression(self, sample_regression_data):
        """Test fitting and prediction for regression"""
        X, y = sample_regression_data
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_evaluate_regression(self, sample_regression_data):
        """Test model evaluation for regression"""
        X, y = sample_regression_data
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert 0 <= metrics["r2"] <= 1
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
    
    def test_feature_importance(self, sample_regression_data):
        """Test feature importance extraction"""
        X, y = sample_regression_data
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert all(imp >= 0 for imp in importance)
        assert abs(sum(importance) - 1.0) < 1e-6  # Should sum to 1


class TestXGBoostModel:
    """Test XGBoostModel"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = XGBoostModel(n_estimators=10, max_depth=3, random_state=42)
        assert model.n_estimators == 10
        assert model.max_depth == 3
    
    def test_fit_predict_regression(self, sample_regression_data):
        """Test fitting and prediction for regression"""
        X, y = sample_regression_data
        
        model = XGBoostModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_evaluate_regression(self, sample_regression_data):
        """Test model evaluation"""
        X, y = sample_regression_data
        
        model = XGBoostModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics


class TestLinearRegressionModel:
    """Test LinearRegressionModel"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = LinearRegressionModel(fit_intercept=True)
        assert model.fit_intercept == True
    
    def test_fit_predict_regression(self, sample_regression_data):
        """Test fitting and prediction"""
        X, y = sample_regression_data
        
        model = LinearRegressionModel()
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_coefficients(self, sample_regression_data):
        """Test coefficient extraction"""
        X, y = sample_regression_data
        
        model = LinearRegressionModel()
        model.fit(X, y)
        
        coefficients = model.get_coefficients()
        assert len(coefficients) == X.shape[1]


class TestModelTrainer:
    """Test ModelTrainer functionality"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        trainer = ModelTrainer()
        assert trainer.config is not None
        assert isinstance(trainer.models, dict)
    
    def test_prepare_data(self, sample_regression_data):
        """Test data preparation"""
        X, y = sample_regression_data
        data = X.copy()
        data["target"] = y
        
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(data, "target")
        
        assert len(X_train) + len(X_test) == len(data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_initialize_models(self):
        """Test model initialization"""
        trainer = ModelTrainer()
        models = trainer.initialize_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        for model_name, model in models.items():
            assert isinstance(model, BaseModel)
    
    def test_train_single_model(self, sample_regression_data):
        """Test training a single model"""
        X, y = sample_regression_data
        
        trainer = ModelTrainer()
        trainer.initialize_models()
        
        # Train random forest
        model = trainer.train_model("random_forest", X, y)
        assert model is not None
        
        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5


class TestModelEvaluator:
    """Test ModelEvaluator functionality"""
    
    def test_initialization(self):
        """Test evaluator initialization"""
        evaluator = ModelEvaluator()
        assert evaluator.output_dir is not None
    
    def test_evaluate_regression_model(self, sample_regression_data):
        """Test regression model evaluation"""
        X, y = sample_regression_data
        
        # Train a simple model
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_regression_model(model, X, y)
        
        assert "r2_score" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
    
    def test_compare_models(self, sample_regression_data):
        """Test model comparison"""
        X, y = sample_regression_data
        
        # Train multiple models
        models = {
            "rf": RandomForestModel(n_estimators=10, random_state=42),
            "lr": LinearRegressionModel()
        }
        
        for model in models.values():
            model.fit(X, y)
        
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(models, X, y, task_type="regression")
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(models)
        assert "model_name" in comparison.columns


class TestCrossValidator:
    """Test CrossValidator functionality"""
    
    def test_initialization(self):
        """Test cross-validator initialization"""
        cv = CrossValidator(cv_folds=3)
        assert cv.cv_folds == 3
    
    def test_regression_cv(self, sample_regression_data):
        """Test regression cross-validation"""
        X, y = sample_regression_data
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        cv = CrossValidator(cv_folds=3)
        
        results = cv.regression_cv(model, X, y)
        
        assert "r2_test_mean" in results
        assert "r2_test_std" in results
        assert "mse_test_mean" in results
        assert "cv_folds" in results
        assert results["cv_folds"] == 3
    
    def test_compare_models_cv(self, sample_regression_data):
        """Test cross-validation model comparison"""
        X, y = sample_regression_data
        
        models = {
            "rf": RandomForestModel(n_estimators=10, random_state=42),
            "lr": LinearRegressionModel()
        }
        
        cv = CrossValidator(cv_folds=3)
        comparison = cv.compare_models_cv(models, X, y, task_type="regression")
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(models)


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_save_load(self, sample_regression_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_regression_data
        
        # Train model
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make predictions before saving
        predictions_before = model.predict(X[:5])
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        model.save(str(model_path))
        
        # Load model
        loaded_model = RandomForestModel.load(str(model_path))
        
        # Make predictions after loading
        predictions_after = loaded_model.predict(X[:5])
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(predictions_before, predictions_after)


def test_model_integration_workflow(sample_regression_data):
    """Integration test for complete model workflow"""
    X, y = sample_regression_data
    data = X.copy()
    data["target"] = y
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(data, "target")
    
    # Train models
    trainer.initialize_models()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model(evaluation_results)
    
    assert best_model_name in trained_models
    assert best_model is not None
    
    # Test prediction with best model
    predictions = best_model.predict(X_test[:5])
    assert len(predictions) == 5


if __name__ == "__main__":
    pytest.main([__file__])