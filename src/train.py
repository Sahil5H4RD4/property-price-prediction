"""
Model Training & Evaluation Pipeline
=====================================
Trains multiple regression models, evaluates them,
extracts feature importances, and saves the best model.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import preprocess_pipeline

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models():
    """Return dictionary of models to train."""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, random_state=42, learning_rate=0.1, max_depth=5
        ),
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
    }


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importances from a trained model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


def train_and_evaluate():
    """
    Full training pipeline:
    1. Preprocess data
    2. Train all models
    3. Evaluate each
    4. Save the best one
    5. Extract feature importances

    Returns:
        results: dict with model metrics
        best_model_name: name of the best model
        feature_importance: DataFrame of feature importances
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline()

    # Train and evaluate all models
    models = get_models()
    results = {}
    trained_models = {}

    print("/n Training models...\n")
    print(f"{'Model':<25} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"{name:<25} {metrics['MAE']:>12,.2f} {metrics['RMSE']:>12,.2f} {metrics['R2']:>10.4f}")

    # Find best model (highest R²)
    best_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_models[best_name]
    print(f"\n Best Model: {best_name} (R² = {results[best_name]['R2']:.4f})")

    # Save best model
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

    # Save model name
    model_info = {
        'best_model': best_name,
        'metrics': results[best_name],
        'all_results': results,
    }
    info_path = os.path.join(MODELS_DIR, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to {info_path}")

    # Feature importance from best model
    importance_df = get_feature_importance(best_model, feature_names, best_name)
    if importance_df is not None:
        importance_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"\n Top Feature Importances ({best_name}):")
        print(importance_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)

    return results, best_name, importance_df


if __name__ == '__main__':
    results, best_name, importance_df = train_and_evaluate()
