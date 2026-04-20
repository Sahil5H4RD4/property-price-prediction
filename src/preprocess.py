"""
Data Preprocessing & Feature Engineering Module
================================================
Handles data cleaning, encoding, feature engineering,
scaling, and train/test splitting for the housing dataset.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Housing.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Column Definitions ──────────────────────────────────────────
BINARY_COLUMNS = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

CATEGORICAL_COLUMNS = ['furnishingstatus']

NUMERICAL_COLUMNS = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Area tier boundaries derived from the training dataset quantiles.
# These mirror the pd.qcut(q=3) split so inference matches training.
# Recomputed from Housing.csv: 33rd percentile ≈ 4600, 66th ≈ 8000.
AREA_TIER_LOW = 4600
AREA_TIER_HIGH = 8000

# Module-level cache for scaler and feature names
_scaler_cache: StandardScaler | None = None
_feature_names_cache: list | None = None


def _get_scaler_and_features() -> tuple[StandardScaler, list]:
    """Load scaler and feature names from disk once, then cache."""
    global _scaler_cache, _feature_names_cache
    if _scaler_cache is None or _feature_names_cache is None:
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            raise FileNotFoundError(
                "scaler.pkl or feature_names.pkl not found. Run src/train.py first."
            )
        _scaler_cache = joblib.load(scaler_path)
        _feature_names_cache = joblib.load(features_path)
        logger.debug("Loaded scaler and feature names into module cache")
    return _scaler_cache, _feature_names_cache


def load_raw_data() -> pd.DataFrame:
    """Load the raw housing CSV."""
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), DATA_PATH)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with median and categorical with mode."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning("Missing values found:\n%s", missing[missing > 0])
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        logger.info("Missing values filled")
    else:
        logger.debug("No missing values found")
    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode yes/no binary features as 0/1."""
    for col in BINARY_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    logger.debug("Encoded %d binary columns", len(BINARY_COLUMNS))
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode furnishingstatus, ensuring all levels are present."""
    dummies = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    
    # Ensure all levels are present even if not in current sample
    for status in ['furnished', 'semi-furnished', 'unfurnished']:
        col = f'furnishingstatus_{status}'
        if col not in dummies.columns:
            dummies[col] = 0
            
    logger.debug("One-hot encoded: %s", CATEGORICAL_COLUMNS)
    return dummies


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for better prediction."""
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['area_per_room'] = df['area'] / df['total_rooms'].replace(0, 1)

    amenity_cols = [col for col in BINARY_COLUMNS if col in df.columns]
    df['amenity_score'] = df[amenity_cols].sum(axis=1)

    # qcut requires unique bin edges. For small DFs or uniform data, 
    # we fallback to manual tiering or uniform labels.
    if len(df['area'].unique()) > 1:
        try:
            df['area_tier'] = pd.qcut(df['area'], q=3, labels=[0, 1, 2], duplicates='drop').astype(int)
        except ValueError:
            # If qcut still fails due to insufficient unique values for 3 bins
            df['area_tier'] = df['area'].apply(_infer_area_tier)
    else:
        df['area_tier'] = df['area'].apply(_infer_area_tier)

    logger.debug("Engineered features: total_rooms, area_per_room, amenity_score, area_tier")
    return df


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, transform both sets, and persist."""
    global _scaler_cache
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    _scaler_cache = scaler  # keep module cache in sync
    logger.info("Scaler fitted and saved to %s", scaler_path)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(test_size: float = 0.2, random_state: int = 42):
    """Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    global _feature_names_cache

    logger.info("Starting preprocessing pipeline")

    df = load_raw_data()
    df = handle_missing_values(df)
    df = encode_binary_features(df)
    df = encode_categorical_features(df)
    df = engineer_features(df)

    y = df['price']
    X = df.drop('price', axis=1)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Train/test split: %d train, %d test", len(X_train), len(X_test))

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    joblib.dump(feature_names, feature_names_path)
    _feature_names_cache = feature_names  # sync module cache
    logger.info("Feature names saved: %d features", len(feature_names))

    logger.info("Preprocessing complete. Features: %s", feature_names)
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def _infer_area_tier(area: float) -> int:
    """Map area to tier using boundaries derived from training-set quantiles."""
    if area < AREA_TIER_LOW:
        return 0
    elif area < AREA_TIER_HIGH:
        return 1
    return 2


def preprocess_single_input(
    input_dict: dict,
    scaler: StandardScaler | None = None,
    feature_names: list | None = None,
) -> pd.DataFrame:
    """Preprocess a single property dict for inference.

    Uses cached scaler/feature_names if not supplied, avoiding repeated disk reads.

    Args:
        input_dict: raw property features.
        scaler: fitted StandardScaler; loaded from cache if None.
        feature_names: ordered feature list; loaded from cache if None.

    Returns:
        Scaled single-row DataFrame ready for model.predict().
    """
    if scaler is None or feature_names is None:
        cached_scaler, cached_features = _get_scaler_and_features()
        scaler = scaler or cached_scaler
        feature_names = feature_names or cached_features

    df = pd.DataFrame([input_dict])

    # Encode binary columns
    for col in BINARY_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # One-hot encode furnishing status
    if 'furnishingstatus' in df.columns:
        furnishing_val = df['furnishingstatus'].iloc[0]
        df = df.drop('furnishingstatus', axis=1)
        for status in ['furnished', 'semi-furnished', 'unfurnished']:
            df[f'furnishingstatus_{status}'] = 1 if furnishing_val == status else 0

    # Feature engineering
    bedrooms = df.get('bedrooms', pd.Series([0])).iloc[0]
    bathrooms = df.get('bathrooms', pd.Series([0])).iloc[0]
    df['total_rooms'] = bedrooms + bathrooms
    total_rooms = df['total_rooms'].iloc[0]
    df['area_per_room'] = df['area'] / (total_rooms if total_rooms > 0 else 1)

    amenity_cols = [c for c in BINARY_COLUMNS if c in df.columns]
    df['amenity_score'] = df[amenity_cols].sum(axis=1)

    df['area_tier'] = _infer_area_tier(float(df['area'].iloc[0]))

    # Ensure all expected features are present in the correct order
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    df = df[feature_names].astype(float)

    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    return df_scaled


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline()
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
