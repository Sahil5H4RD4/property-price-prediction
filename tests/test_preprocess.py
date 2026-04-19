"""
Unit tests for src/preprocess.py

Tests cover:
- Binary feature encoding
- Categorical one-hot encoding
- Feature engineering (derived columns)
- area_tier inference consistency
- preprocess_single_input shape and column order
- Missing value handling
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.preprocess import (
    BINARY_COLUMNS,
    AREA_TIER_LOW,
    AREA_TIER_HIGH,
    encode_binary_features,
    encode_categorical_features,
    engineer_features,
    handle_missing_values,
    _infer_area_tier,
)


# ─── encode_binary_features ──────────────────────────────────────

class TestEncodeBinaryFeatures:
    def _make_df(self, values: dict) -> pd.DataFrame:
        return pd.DataFrame([values])

    def test_yes_maps_to_one(self):
        df = self._make_df({'mainroad': 'yes'})
        result = encode_binary_features(df)
        assert result['mainroad'].iloc[0] == 1

    def test_no_maps_to_zero(self):
        df = self._make_df({'mainroad': 'no'})
        result = encode_binary_features(df)
        assert result['mainroad'].iloc[0] == 0

    def test_already_numeric_untouched(self):
        df = self._make_df({'mainroad': 1})
        result = encode_binary_features(df)
        assert result['mainroad'].iloc[0] == 1

    def test_all_binary_columns_encoded(self):
        row = {col: 'yes' for col in BINARY_COLUMNS}
        df = pd.DataFrame([row])
        result = encode_binary_features(df)
        for col in BINARY_COLUMNS:
            assert result[col].dtype in (int, np.int64, np.int32), \
                f"{col} should be integer after encoding"
            assert result[col].iloc[0] == 1

    def test_missing_column_skipped_without_error(self):
        df = pd.DataFrame([{'area': 5000}])
        result = encode_binary_features(df)
        assert 'area' in result.columns


# ─── encode_categorical_features ─────────────────────────────────

class TestEncodeCategoricalFeatures:
    def test_furnished_creates_three_dummies(self):
        df = pd.DataFrame([{'furnishingstatus': 'furnished'}])
        result = encode_categorical_features(df)
        expected = {
            'furnishingstatus_furnished',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished',
        }
        assert expected.issubset(result.columns)

    def test_correct_dummy_is_one(self):
        for status in ('furnished', 'semi-furnished', 'unfurnished'):
            df = pd.DataFrame([{'furnishingstatus': status}])
            result = encode_categorical_features(df)
            assert result[f'furnishingstatus_{status}'].iloc[0] == 1

    def test_other_dummies_are_zero(self):
        df = pd.DataFrame([{'furnishingstatus': 'furnished'}])
        result = encode_categorical_features(df)
        assert result['furnishingstatus_unfurnished'].iloc[0] == 0


# ─── engineer_features ───────────────────────────────────────────

class TestEngineerFeatures:
    def _base_df(self, area=6000, beds=3, baths=2) -> pd.DataFrame:
        row = {col: 0 for col in BINARY_COLUMNS}
        row.update({'area': area, 'bedrooms': beds, 'bathrooms': baths})
        return pd.DataFrame([row])

    def test_total_rooms_is_beds_plus_baths(self):
        df = self._base_df(beds=3, baths=2)
        result = engineer_features(df)
        assert result['total_rooms'].iloc[0] == 5

    def test_area_per_room_calculation(self):
        df = self._base_df(area=6000, beds=3, baths=2)
        result = engineer_features(df)
        assert abs(result['area_per_room'].iloc[0] - 6000 / 5) < 0.01

    def test_area_per_room_no_division_by_zero(self):
        df = self._base_df(beds=0, baths=0)
        result = engineer_features(df)
        assert np.isfinite(result['area_per_room'].iloc[0])

    def test_amenity_score_sums_binary_cols(self):
        row = {col: 'yes' for col in BINARY_COLUMNS}
        row.update({'area': 5000, 'bedrooms': 3, 'bathrooms': 2})
        df = pd.DataFrame([row])
        df = encode_binary_features(df)
        result = engineer_features(df)
        assert result['amenity_score'].iloc[0] == len(BINARY_COLUMNS)

    def test_area_tier_produced(self):
        df = self._base_df()
        result = engineer_features(df)
        assert 'area_tier' in result.columns
        assert result['area_tier'].iloc[0] in (0, 1, 2)


# ─── _infer_area_tier ────────────────────────────────────────────

class TestInferAreaTier:
    def test_small_area_is_tier_0(self):
        assert _infer_area_tier(AREA_TIER_LOW - 1) == 0

    def test_mid_area_is_tier_1(self):
        mid = (AREA_TIER_LOW + AREA_TIER_HIGH) // 2
        assert _infer_area_tier(mid) == 1

    def test_large_area_is_tier_2(self):
        assert _infer_area_tier(AREA_TIER_HIGH + 1) == 2

    def test_boundary_low_is_tier_1(self):
        assert _infer_area_tier(AREA_TIER_LOW) == 1

    def test_boundary_high_is_tier_2(self):
        assert _infer_area_tier(AREA_TIER_HIGH) == 2


# ─── handle_missing_values ───────────────────────────────────────

class TestHandleMissingValues:
    def test_numeric_filled_with_median(self):
        df = pd.DataFrame({'price': [100, 200, np.nan, 400]})
        result = handle_missing_values(df)
        assert result['price'].isnull().sum() == 0
        # median of [100, 200, 400] = 200
        assert result['price'].iloc[2] == 200.0

    def test_categorical_filled_with_mode(self):
        df = pd.DataFrame({'status': ['yes', 'yes', 'no', None]})
        result = handle_missing_values(df)
        assert result['status'].isnull().sum() == 0
        assert result['status'].iloc[3] == 'yes'

    def test_no_missing_unchanged(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(result, df)


# ─── preprocess_single_input ─────────────────────────────────────

class TestPreprocessSingleInput:
    """Integration-level tests that mock the scaler to avoid needing
    trained artifacts on disk."""

    def _mock_scaler(self, n_features: int):
        from sklearn.preprocessing import StandardScaler
        scaler = MagicMock(spec=StandardScaler)
        scaler.transform.side_effect = lambda X: X.values
        return scaler

    def test_output_is_dataframe(self, valid_property):
        from src.preprocess import preprocess_single_input

        feature_names = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'prefarea',
            'furnishingstatus_furnished',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished',
            'total_rooms', 'area_per_room', 'amenity_score', 'area_tier',
        ]
        scaler = self._mock_scaler(len(feature_names))

        result = preprocess_single_input(valid_property, scaler=scaler, feature_names=feature_names)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, len(feature_names))

    def test_output_columns_match_feature_names(self, valid_property):
        from src.preprocess import preprocess_single_input

        feature_names = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'prefarea',
            'furnishingstatus_furnished',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished',
            'total_rooms', 'area_per_room', 'amenity_score', 'area_tier',
        ]
        scaler = self._mock_scaler(len(feature_names))

        result = preprocess_single_input(valid_property, scaler=scaler, feature_names=feature_names)
        assert list(result.columns) == feature_names

    def test_furnishing_one_hot_correct(self, valid_property):
        from src.preprocess import preprocess_single_input

        feature_names = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'prefarea',
            'furnishingstatus_furnished',
            'furnishingstatus_semi-furnished',
            'furnishingstatus_unfurnished',
            'total_rooms', 'area_per_room', 'amenity_score', 'area_tier',
        ]
        scaler = self._mock_scaler(len(feature_names))
        valid_property['furnishingstatus'] = 'furnished'

        result = preprocess_single_input(valid_property, scaler=scaler, feature_names=feature_names)
        assert result['furnishingstatus_furnished'].iloc[0] == 1.0
        assert result['furnishingstatus_unfurnished'].iloc[0] == 0.0
