"""
Model Training & Evaluation Pipeline
=====================================
Trains multiple regression models, evaluates them,
extracts feature importances, and saves the best model.
"""

import logging
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import preprocess_pipeline

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
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


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute MAE, RMSE, and R² for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        'MAE': round(float(mean_absolute_error(y_test, y_pred)), 2),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        'R2': round(float(r2_score(y_test, y_pred)), 4),
    }


def get_feature_importance(
    model, feature_names: list, model_name: str
) -> pd.DataFrame | None:
    """Extract feature importances from a fitted model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None

    return (
        pd.DataFrame({'feature': feature_names, 'importance': importances})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


def train_and_evaluate() -> tuple[dict, str, pd.DataFrame]:
    """Full training pipeline: preprocess → train → evaluate → save.

    Returns:
        results: dict mapping model name → metrics
        best_model_name: name of the highest-R² model
        importance_df: feature importances for the best model
    """
    logger.info("Starting model training pipeline")

    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline()

    models = get_models()
    results: dict = {}
    trained_models: dict = {}

    logger.info("Training %d models", len(models))
    print(f"\n{'Model':<25} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
    print("-" * 60)

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model

        print(
            f"{name:<25} {metrics['MAE']:>12,.2f} "
            f"{metrics['RMSE']:>12,.2f} {metrics['R2']:>10.4f}"
        )

        model_filename = f"{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
        logger.debug("Saved %s to %s", name, model_filename)

    best_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_models[best_name]
    logger.info("Best model: %s (R²=%.4f)", best_name, results[best_name]['R2'])

    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)

    model_info = {
        'best_model': best_name,
        'metrics': results[best_name],
        'all_results': results,
    }
    info_path = os.path.join(MODELS_DIR, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info("Model info written to %s", info_path)

    best_importance_df = None
    for name, model in trained_models.items():
        importance_df = get_feature_importance(model, feature_names, name)
        if importance_df is None:
            continue

        safe_name = name.replace(' ', '_').lower()
        importance_path = os.path.join(MODELS_DIR, f'importance_{safe_name}.csv')
        importance_df.to_csv(importance_path, index=False)

        if name == best_name:
            best_importance_df = importance_df
            importance_df.to_csv(
                os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False
            )
            logger.info("Top features (%s):\n%s", best_name, importance_df.head(5))

    logger.info("Training pipeline complete")
    return results, best_name, best_importance_df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    results, best_name, importance_df = train_and_evaluate()
    print(f"\nBest Model: {best_name}")
