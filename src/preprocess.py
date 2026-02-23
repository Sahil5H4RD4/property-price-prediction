"""
Data Preprocessing & Feature Engineering Module
================================================
Handles data cleaning, encoding, feature engineering,
scaling, and train/test splitting for the housing dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Housing.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Binary Columns ─────────────────────────────────────────────
BINARY_COLUMNS = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

CATEGORICAL_COLUMNS = ['furnishingstatus']

NUMERICAL_COLUMNS = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']


def load_raw_data():
    """Load the raw housing CSV."""
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"⚠️  Found missing values:\n{missing[missing > 0]}")
        # Fill numeric with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        # Fill categorical with mode
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        print("✅ Missing values handled")
    else:
        print("✅ No missing values found")
    return df


def encode_binary_features(df):
    """Encode yes/no binary features as 0/1."""
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})
    print(f"✅ Encoded {len(BINARY_COLUMNS)} binary columns")
    return df


def encode_categorical_features(df):
    """One-hot encode categorical features."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=False)
    print(f"✅ One-hot encoded: {CATEGORICAL_COLUMNS}")
    return df


def engineer_features(df):
    """Create derived features for better prediction."""
    # Total rooms = bedrooms + bathrooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Area per room
    df['area_per_room'] = df['area'] / df['total_rooms'].replace(0, 1)

    # Amenity score (sum of all binary amenities)
    amenity_cols = [col for col in BINARY_COLUMNS if col in df.columns]
    df['amenity_score'] = df[amenity_cols].sum(axis=1)

    # Property tier based on area
    df['area_tier'] = pd.qcut(df['area'], q=3, labels=[0, 1, 2]).astype(int)

    print(f"✅ Engineered features: total_rooms, area_per_room, amenity_score, area_tier")
    return df


def scale_features(X_train, X_test):
    """Apply StandardScaler to features, fit on train, transform both."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Save the scaler for inference
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    df = load_raw_data()

    # Clean
    df = handle_missing_values(df)

    # Encode
    df = encode_binary_features(df)
    df = encode_categorical_features(df)

    # Feature Engineering
    df = engineer_features(df)

    # Separate target
    y = df['price']
    X = df.drop('price', axis=1)

    # Store feature names
    feature_names = X.columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Train/test split: {len(X_train)} train, {len(X_test)} test")

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save feature names
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    joblib.dump(feature_names, feature_names_path)
    print(f"✅ Feature names saved ({len(feature_names)} features)")

    print("\n✅ Preprocessing complete!")
    print(f"   Features: {feature_names}")
    print("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def preprocess_single_input(input_dict, scaler=None, feature_names=None):
    """
    Preprocess a single property input for prediction.

    Args:
        input_dict: dict with property features
        scaler: fitted StandardScaler (loaded from file if None)
        feature_names: list of feature names (loaded from file if None)

    Returns:
        Scaled DataFrame ready for prediction
    """
    if scaler is None:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    if feature_names is None:
        feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

    # Create a single-row DataFrame
    df = pd.DataFrame([input_dict])

    # Encode binary features
    for col in BINARY_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({'yes': 1, 'no': 0})

    # One-hot encode furnishing status
    if 'furnishingstatus' in df.columns:
        furnishing_val = df['furnishingstatus'].iloc[0]
        df = df.drop('furnishingstatus', axis=1)
        for status in ['furnished', 'semi-furnished', 'unfurnished']:
            col_name = f'furnishingstatus_{status}'
            df[col_name] = 1 if furnishing_val == status else 0

    # Engineer features
    df['total_rooms'] = df.get('bedrooms', 0) + df.get('bathrooms', 0)
    total_rooms = df['total_rooms'].iloc[0]
    df['area_per_room'] = df['area'] / (total_rooms if total_rooms > 0 else 1)

    amenity_cols = [c for c in BINARY_COLUMNS if c in df.columns]
    df['amenity_score'] = df[amenity_cols].sum(axis=1)

    # For area_tier, use simple rules (matching the quartile logic)
    area = df['area'].iloc[0]
    if area < 6000:
        df['area_tier'] = 0
    elif area < 12000:
        df['area_tier'] = 1
    else:
        df['area_tier'] = 2

    # Ensure all features present and in correct order
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    df = df[feature_names]

    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_names
    )

    return df_scaled


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_pipeline()
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
