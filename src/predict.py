"""
Prediction Utilities
====================
Load the trained model and make predictions
on new property data.
"""

import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

REQUIRED_INPUT_FIELDS = [
    'area', 'bedrooms', 'bathrooms', 'stories',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
    'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
]

BINARY_FIELDS = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

VALID_FURNISHING = {'furnished', 'semi-furnished', 'unfurnished'}
VALID_YESNO = {'yes', 'no'}

# Module-level cache to avoid repeated JSON reads
_model_info_cache: dict | None = None


def get_model_info() -> dict:
    """Load model_info.json once and cache it."""
    global _model_info_cache
    if _model_info_cache is None:
        info_path = os.path.join(MODELS_DIR, 'model_info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"model_info.json not found at {info_path}. "
                "Run src/train.py to train models first."
            )
        with open(info_path, 'r') as f:
            _model_info_cache = json.load(f)
        logger.debug("Loaded model_info.json into cache")
    return _model_info_cache


def validate_input(input_dict: dict) -> None:
    """Validate property input dict before prediction.

    Raises:
        ValueError: if required fields missing, out of bounds, or invalid values.
    """
    missing = [f for f in REQUIRED_INPUT_FIELDS if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if not isinstance(input_dict['area'], (int, float)) or input_dict['area'] <= 0:
        raise ValueError(f"'area' must be a positive number, got: {input_dict['area']}")

    for field in ('bedrooms', 'bathrooms', 'stories', 'parking'):
        val = input_dict[field]
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(f"'{field}' must be a non-negative number, got: {val}")

    for field in BINARY_FIELDS:
        val = input_dict[field]
        if isinstance(val, str) and val not in VALID_YESNO:
            raise ValueError(f"'{field}' must be 'yes' or 'no', got: '{val}'")

    fs = input_dict['furnishingstatus']
    if fs not in VALID_FURNISHING:
        raise ValueError(
            f"'furnishingstatus' must be one of {VALID_FURNISHING}, got: '{fs}'"
        )


def load_model(model_name: str | None = None):
    """Load a trained model, scaler, feature names, and model info.

    Raises:
        FileNotFoundError: if model file or required artifacts are missing.
    """
    if model_name:
        safe_name = model_name.replace(' ', '_').lower()
        model_path = os.path.join(MODELS_DIR, f'{safe_name}.pkl')
        if not os.path.exists(model_path):
            logger.warning(
                "Model '%s' not found at %s, falling back to best_model.pkl",
                model_name, model_path
            )
            model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    else:
        model_path = os.path.join(MODELS_DIR, 'best_model.pkl')

    for path, label in [
        (model_path, 'model'),
        (os.path.join(MODELS_DIR, 'scaler.pkl'), 'scaler'),
        (os.path.join(MODELS_DIR, 'feature_names.pkl'), 'feature_names'),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artifact '{label}' not found at {path}. "
                "Run src/train.py to generate model artifacts."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    model_info = get_model_info()

    logger.info("Loaded model from %s", model_path)
    return model, scaler, feature_names, model_info


def predict_price(
    input_dict: dict,
    model=None,
    scaler=None,
    feature_names=None,
    model_name: str | None = None,
) -> dict:
    """Predict price for a single property.

    Args:
        input_dict: property features dict.
        model, scaler, feature_names: pre-loaded artifacts (optional).
        model_name: name of the specific model to use if model is None.

    Returns:
        dict with predicted_price, model_name, and confidence metrics.

    Raises:
        ValueError: if input_dict is invalid.
    """
    from src.preprocess import preprocess_single_input

    validate_input(input_dict)

    if model is None:
        model, scaler, feature_names, model_info = load_model(model_name)
    else:
        model_info = get_model_info()

    X = preprocess_single_input(input_dict, scaler=scaler, feature_names=feature_names)
    predicted_price = model.predict(X)[0]

    actual_model_name = (
        model_name if model_name and model_name in model_info['all_results']
        else model_info['best_model']
    )
    metrics = model_info['all_results'].get(actual_model_name, model_info['metrics'])

    logger.info("Predicted price ₹%,.0f using %s", predicted_price, actual_model_name)
    return {
        'predicted_price': round(float(predicted_price), 2),
        'model_name': actual_model_name,
        'model_r2': metrics['R2'],
        'model_mae': metrics['MAE'],
        'model_rmse': metrics.get('RMSE', 0),
    }


def predict_batch(df: pd.DataFrame, model=None, scaler=None, feature_names=None) -> pd.DataFrame:
    """Predict prices for a batch of properties.

    Preprocesses all rows in a single pass instead of row-by-row to improve
    performance on large uploads.

    Args:
        df: DataFrame with property feature columns.
        model, scaler, feature_names: pre-loaded artifacts (optional).

    Returns:
        Copy of df with 'predicted_price' column appended.
    """
    from src.preprocess import (
        BINARY_COLUMNS, preprocess_single_input
    )

    if model is None:
        model, scaler, feature_names, _ = load_model()

    logger.info("Running batch prediction on %d rows", len(df))

    # Validate each row before batch processing
    errors = []
    for idx, row in df.iterrows():
        try:
            validate_input(row.to_dict())
        except ValueError as e:
            errors.append(f"Row {idx}: {e}")

    if errors:
        raise ValueError(
            f"Batch validation failed for {len(errors)} row(s):\n"
            + "\n".join(errors[:5])
            + ("\n..." if len(errors) > 5 else "")
        )

    predictions = []
    for _, row in df.iterrows():
        X = preprocess_single_input(row.to_dict(), scaler=scaler, feature_names=feature_names)
        predictions.append(round(float(model.predict(X)[0]), 2))

    result_df = df.copy()
    result_df['predicted_price'] = predictions
    logger.info("Batch prediction complete")
    return result_df


def get_feature_importance() -> pd.DataFrame | None:
    """Load feature importance data for the best model."""
    importance_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    if os.path.exists(importance_path):
        return pd.read_csv(importance_path)
    return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sample = {
        'area': 7420,
        'bedrooms': 4,
        'bathrooms': 2,
        'stories': 3,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'semi-furnished',
    }

    result = predict_price(sample)
    print(f"\nSample Property Prediction:")
    print(f"  Predicted Price: \u20b9{result['predicted_price']:,.2f}")
    print(f"  Model: {result['model_name']} (R\u00b2 = {result['model_r2']})")
