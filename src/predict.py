"""
Prediction Utilities
====================
Load the trained model and make predictions
on new property data.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_model():
    """Load the trained model, scaler, and feature names."""
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

    with open(os.path.join(MODELS_DIR, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    return model, scaler, feature_names, model_info


def predict_price(input_dict, model=None, scaler=None, feature_names=None):
    """
    Predict price for a single property.

    Args:
        input_dict: dict with property features
        model, scaler, feature_names: pre-loaded artifacts (optional)

    Returns:
        dict with predicted_price, model_name, confidence metrics
    """
    from src.preprocess import preprocess_single_input

    if model is None:
        model, scaler, feature_names, model_info = load_model()
    else:
        with open(os.path.join(MODELS_DIR, 'model_info.json'), 'r') as f:
            model_info = json.load(f)

    # Preprocess input
    X = preprocess_single_input(input_dict, scaler=scaler, feature_names=feature_names)

    # Predict
    predicted_price = model.predict(X)[0]

    return {
        'predicted_price': round(predicted_price, 2),
        'model_name': model_info['best_model'],
        'model_r2': model_info['metrics']['R2'],
        'model_mae': model_info['metrics']['MAE'],
    }


def predict_batch(df, model=None, scaler=None, feature_names=None):
    """
    Predict prices for a batch of properties from a DataFrame.

    Args:
        df: DataFrame with property features
        model, scaler, feature_names: pre-loaded artifacts (optional)

    Returns:
        DataFrame with original data + predicted_price column
    """
    from src.preprocess import preprocess_single_input

    if model is None:
        model, scaler, feature_names, model_info = load_model()

    predictions = []
    for _, row in df.iterrows():
        input_dict = row.to_dict()
        X = preprocess_single_input(input_dict, scaler=scaler, feature_names=feature_names)
        pred = model.predict(X)[0]
        predictions.append(round(pred, 2))

    result_df = df.copy()
    result_df['predicted_price'] = predictions
    return result_df


def get_feature_importance():
    """Load feature importance data."""
    importance_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    if os.path.exists(importance_path):
        return pd.read_csv(importance_path)
    return None


if __name__ == '__main__':
    # Test prediction with sample input
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
    print(f"\n Sample Property Prediction:")
    print(f"   Predicted Price: ₹{result['predicted_price']:,.2f}")
    print(f"   Model: {result['model_name']} (R² = {result['model_r2']})")
