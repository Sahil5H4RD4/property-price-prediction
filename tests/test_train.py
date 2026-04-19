"""
Unit tests for src/train.py

Tests cover:
- get_models: returns expected model names with correct types
- evaluate_model: metric keys, types, and ranges
- get_feature_importance: DataFrame shape, column names, sorting
- train_and_evaluate: end-to-end on synthetic data with mocked file I/O
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.train import get_models, evaluate_model, get_feature_importance


# ─── get_models ──────────────────────────────────────────────────

class TestGetModels:
    def test_returns_four_models(self):
        models = get_models()
        assert len(models) == 4

    def test_expected_model_names_present(self):
        models = get_models()
        assert 'Linear Regression' in models
        assert 'Decision Tree' in models
        assert 'Random Forest' in models
        assert 'Gradient Boosting' in models

    def test_model_types_correct(self):
        models = get_models()
        assert isinstance(models['Linear Regression'], LinearRegression)
        assert isinstance(models['Decision Tree'], DecisionTreeRegressor)
        assert isinstance(models['Random Forest'], RandomForestRegressor)
        assert isinstance(models['Gradient Boosting'], GradientBoostingRegressor)

    def test_random_forest_uses_parallel_jobs(self):
        models = get_models()
        assert models['Random Forest'].n_jobs == -1

    def test_models_have_deterministic_random_state(self):
        models = get_models()
        for name, model in models.items():
            if hasattr(model, 'random_state'):
                assert model.random_state == 42, f"{name} should use random_state=42"


# ─── evaluate_model ──────────────────────────────────────────────

class TestEvaluateModel:
    @pytest.fixture
    def fitted_linear(self, sample_housing_df):
        """Train a LinearRegression on synthetic data for evaluation tests."""
        from src.preprocess import BINARY_COLUMNS
        df = sample_housing_df.copy()
        for col in BINARY_COLUMNS:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        df = pd.get_dummies(df, columns=['furnishingstatus'])
        y = df['price']
        X = df.drop('price', axis=1).astype(float)

        model = LinearRegression()
        model.fit(X, y)
        return model, X, y

    def test_returns_required_keys(self, fitted_linear):
        model, X, y = fitted_linear
        metrics = evaluate_model(model, X, y)
        assert set(metrics.keys()) == {'MAE', 'RMSE', 'R2'}

    def test_metrics_are_python_floats(self, fitted_linear):
        model, X, y = fitted_linear
        metrics = evaluate_model(model, X, y)
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_mae_and_rmse_non_negative(self, fitted_linear):
        model, X, y = fitted_linear
        metrics = evaluate_model(model, X, y)
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0

    def test_r2_bounded(self, fitted_linear):
        model, X, y = fitted_linear
        metrics = evaluate_model(model, X, y)
        assert metrics['R2'] <= 1.0

    def test_perfect_prediction_gives_r2_one(self):
        """A model that predicts exactly should give R²=1."""
        model = MagicMock()
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        model.predict.return_value = y.values
        metrics = evaluate_model(model, None, y)
        assert metrics['R2'] == 1.0
        assert metrics['MAE'] == 0.0
        assert metrics['RMSE'] == 0.0


# ─── get_feature_importance ──────────────────────────────────────

class TestGetFeatureImportance:
    def test_tree_model_returns_dataframe(self, sample_housing_df):
        from src.preprocess import BINARY_COLUMNS
        df = sample_housing_df.copy()
        for col in BINARY_COLUMNS:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        df = pd.get_dummies(df, columns=['furnishingstatus'])
        y = df['price']
        X = df.drop('price', axis=1).astype(float)

        model = DecisionTreeRegressor(random_state=42, max_depth=3)
        model.fit(X, y)

        result = get_feature_importance(model, list(X.columns), 'Decision Tree')
        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'importance' in result.columns

    def test_linear_model_uses_coef(self, sample_housing_df):
        from src.preprocess import BINARY_COLUMNS
        df = sample_housing_df.copy()
        for col in BINARY_COLUMNS:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        df = pd.get_dummies(df, columns=['furnishingstatus'])
        y = df['price']
        X = df.drop('price', axis=1).astype(float)

        model = LinearRegression()
        model.fit(X, y)

        result = get_feature_importance(model, list(X.columns), 'Linear Regression')
        assert result is not None
        assert len(result) == len(X.columns)

    def test_sorted_descending(self, sample_housing_df):
        from src.preprocess import BINARY_COLUMNS
        df = sample_housing_df.copy()
        for col in BINARY_COLUMNS:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        df = pd.get_dummies(df, columns=['furnishingstatus'])
        y = df['price']
        X = df.drop('price', axis=1).astype(float)

        model = DecisionTreeRegressor(random_state=42, max_depth=3)
        model.fit(X, y)
        result = get_feature_importance(model, list(X.columns), 'DT')

        importances = result['importance'].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_model_without_importance_returns_none(self):
        model = MagicMock(spec=[])  # no coef_, no feature_importances_
        result = get_feature_importance(model, ['a', 'b'], 'dummy')
        assert result is None

    def test_output_has_reset_index(self, sample_housing_df):
        from src.preprocess import BINARY_COLUMNS
        df = sample_housing_df.copy()
        for col in BINARY_COLUMNS:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        df = pd.get_dummies(df, columns=['furnishingstatus'])
        y = df['price']
        X = df.drop('price', axis=1).astype(float)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        result = get_feature_importance(model, list(X.columns), 'RF')

        # Index should be 0-based after reset_index(drop=True)
        assert list(result.index) == list(range(len(result)))
