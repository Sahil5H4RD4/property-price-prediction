"""
Unit tests for src/predict.py

Tests cover:
- validate_input: required fields, numeric bounds, binary values, furnishing
- load_model: FileNotFoundError when artifacts missing
- get_model_info: caching behaviour and FileNotFoundError
- predict_price: end-to-end with mocked model/scaler/features
- predict_batch: multi-row prediction and validation error surfacing
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open

from src.predict import validate_input, REQUIRED_INPUT_FIELDS


# ─── validate_input ──────────────────────────────────────────────

class TestValidateInput:
    def test_valid_input_passes(self, valid_property):
        validate_input(valid_property)  # should not raise

    def test_missing_field_raises(self, valid_property):
        del valid_property['area']
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_input(valid_property)

    def test_all_required_fields_checked(self):
        for field in REQUIRED_INPUT_FIELDS:
            prop = {f: 'dummy' for f in REQUIRED_INPUT_FIELDS}
            del prop[field]
            with pytest.raises(ValueError, match="Missing required fields"):
                validate_input(prop)

    def test_negative_area_raises(self, valid_property):
        valid_property['area'] = -100
        with pytest.raises(ValueError, match="area"):
            validate_input(valid_property)

    def test_zero_area_raises(self, valid_property):
        valid_property['area'] = 0
        with pytest.raises(ValueError, match="area"):
            validate_input(valid_property)

    def test_negative_bedrooms_raises(self, valid_property):
        valid_property['bedrooms'] = -1
        with pytest.raises(ValueError, match="bedrooms"):
            validate_input(valid_property)

    def test_zero_bedrooms_allowed(self, valid_property):
        valid_property['bedrooms'] = 0
        validate_input(valid_property)  # should not raise

    def test_invalid_binary_value_raises(self, valid_property):
        valid_property['mainroad'] = 'maybe'
        with pytest.raises(ValueError, match="mainroad"):
            validate_input(valid_property)

    def test_numeric_binary_value_allowed(self, valid_property):
        valid_property['mainroad'] = 1
        validate_input(valid_property)  # should not raise

    def test_invalid_furnishing_raises(self, valid_property):
        valid_property['furnishingstatus'] = 'partially-furnished'
        with pytest.raises(ValueError, match="furnishingstatus"):
            validate_input(valid_property)

    @pytest.mark.parametrize("status", ["furnished", "semi-furnished", "unfurnished"])
    def test_all_valid_furnishing_statuses_accepted(self, valid_property, status):
        valid_property['furnishingstatus'] = status
        validate_input(valid_property)  # should not raise


# ─── get_model_info ──────────────────────────────────────────────

class TestGetModelInfo:
    def test_raises_when_file_missing(self, tmp_path, monkeypatch):
        import src.predict as predict_module
        monkeypatch.setattr(predict_module, 'MODELS_DIR', str(tmp_path))
        monkeypatch.setattr(predict_module, '_model_info_cache', None)

        with pytest.raises(FileNotFoundError, match="model_info.json"):
            predict_module.get_model_info()

    def test_returns_dict_and_caches(self, tmp_path, monkeypatch):
        import src.predict as predict_module

        fake_info = {
            'best_model': 'Random Forest',
            'metrics': {'R2': 0.85, 'MAE': 500000, 'RMSE': 700000},
            'all_results': {
                'Random Forest': {'R2': 0.85, 'MAE': 500000, 'RMSE': 700000}
            },
        }
        info_path = tmp_path / 'model_info.json'
        info_path.write_text(json.dumps(fake_info))

        monkeypatch.setattr(predict_module, 'MODELS_DIR', str(tmp_path))
        monkeypatch.setattr(predict_module, '_model_info_cache', None)

        result = predict_module.get_model_info()
        assert result['best_model'] == 'Random Forest'

        # Second call should return the same cached object
        cached = predict_module.get_model_info()
        assert result is cached


# ─── load_model ──────────────────────────────────────────────────

class TestLoadModel:
    def test_raises_when_model_pkl_missing(self, tmp_path, monkeypatch):
        import src.predict as predict_module
        monkeypatch.setattr(predict_module, 'MODELS_DIR', str(tmp_path))
        monkeypatch.setattr(predict_module, '_model_info_cache', None)

        with pytest.raises(FileNotFoundError):
            predict_module.load_model()


# ─── predict_price ───────────────────────────────────────────────

class TestPredictPrice:
    """End-to-end predict_price tests using mocked ML artifacts."""

    def _make_mock_model(self, return_value=5_000_000.0):
        model = MagicMock()
        model.predict.return_value = np.array([return_value])
        return model

    def _make_mock_scaler(self, n=18):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda X: X.values
        return scaler

    def _feature_names(self):
        return [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'prefarea',
            'furnishingstatus_furnished',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished',
            'total_rooms', 'area_per_room', 'amenity_score', 'area_tier',
        ]

    def _fake_model_info(self):
        return {
            'best_model': 'Random Forest',
            'metrics': {'R2': 0.85, 'MAE': 500_000, 'RMSE': 700_000},
            'all_results': {
                'Random Forest': {'R2': 0.85, 'MAE': 500_000, 'RMSE': 700_000}
            },
        }

    def test_returns_expected_keys(self, valid_property):
        import src.predict as predict_module

        model = self._make_mock_model(5_000_000.0)
        scaler = self._make_mock_scaler()
        feature_names = self._feature_names()

        with patch.object(predict_module, 'get_model_info', return_value=self._fake_model_info()):
            result = predict_price_helper(valid_property, model, scaler, feature_names)

        assert 'predicted_price' in result
        assert 'model_name' in result
        assert 'model_r2' in result
        assert 'model_mae' in result
        assert 'model_rmse' in result

    def test_predicted_price_is_float(self, valid_property):
        import src.predict as predict_module

        model = self._make_mock_model(7_500_000.0)
        scaler = self._make_mock_scaler()
        feature_names = self._feature_names()

        with patch.object(predict_module, 'get_model_info', return_value=self._fake_model_info()):
            result = predict_price_helper(valid_property, model, scaler, feature_names)

        assert isinstance(result['predicted_price'], float)
        assert result['predicted_price'] == 7_500_000.0

    def test_invalid_input_raises_before_model_call(self, valid_property):
        from src.predict import predict_price

        valid_property['area'] = -1
        model = self._make_mock_model()

        with pytest.raises(ValueError, match="area"):
            predict_price(valid_property, model=model, scaler=self._make_mock_scaler(),
                          feature_names=self._feature_names())
        model.predict.assert_not_called()


def predict_price_helper(input_dict, model, scaler, feature_names):
    """Call predict_price with pre-built artifacts."""
    from src.predict import predict_price
    return predict_price(input_dict, model=model, scaler=scaler, feature_names=feature_names)


# ─── predict_batch ───────────────────────────────────────────────

class TestPredictBatch:
    def _make_valid_df(self, n=5) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'area': 5000 + i * 100,
                'bedrooms': 3,
                'bathrooms': 2,
                'stories': 2,
                'mainroad': 'yes',
                'guestroom': 'no',
                'basement': 'no',
                'hotwaterheating': 'no',
                'airconditioning': 'yes',
                'parking': 1,
                'prefarea': 'yes',
                'furnishingstatus': 'semi-furnished',
            }
            for i in range(n)
        ])

    def test_output_has_predicted_price_column(self):
        from src.predict import predict_batch

        df = self._make_valid_df(3)
        model = MagicMock()
        model.predict.return_value = np.array([5_000_000.0])
        scaler = MagicMock()
        scaler.transform.side_effect = lambda X: X.values

        feature_names = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'prefarea',
            'furnishingstatus_furnished',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished',
            'total_rooms', 'area_per_room', 'amenity_score', 'area_tier',
        ]

        result = predict_batch(df, model=model, scaler=scaler, feature_names=feature_names)
        assert 'predicted_price' in result.columns
        assert len(result) == len(df)

    def test_invalid_row_surfaces_error(self):
        from src.predict import predict_batch

        df = self._make_valid_df(2)
        df.at[0, 'area'] = -500  # invalid

        model = MagicMock()
        model.predict.return_value = np.array([5_000_000.0])
        scaler = MagicMock()
        scaler.transform.side_effect = lambda X: X.values

        with pytest.raises(ValueError, match="Batch validation failed"):
            predict_batch(df, model=model, scaler=scaler, feature_names=[])
