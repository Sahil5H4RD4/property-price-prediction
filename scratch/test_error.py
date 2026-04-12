
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess_single_input

input_data = {
    'area': 5000, 
    'bedrooms': 3, 
    'bathrooms': 2,
    'stories': 2, 
    'mainroad': pd.Series(['yes'], dtype='string')[0], 
    'guestroom': pd.Series(['no'], dtype='string')[0],
    'basement': pd.Series(['no'], dtype='string')[0], 
    'hotwaterheating': pd.Series(['no'], dtype='string')[0],
    'airconditioning': pd.Series(['yes'], dtype='string')[0], 
    'parking': 1,
    'prefarea': pd.Series(['yes'], dtype='string')[0], 
    'furnishingstatus': pd.Series(['semi-furnished'], dtype='string')[0]
}

try:
    import sys
    print(f"Python version: {sys.version}")
    
    # We need to manually do what preprocess_single_input does to inspect dtypes
    from src.preprocess import MODELS_DIR, BINARY_COLUMNS, CATEGORICAL_COLUMNS
    import joblib
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    
    df = pd.DataFrame([input_data])
    
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
    
    area = df['area'].iloc[0]
    if area < 6000:
        df['area_tier'] = 0
    elif area < 12000:
        df['area_tier'] = 1
    else:
        df['area_tier'] = 2
        
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
            
    df = df[feature_names]
    
    print("\nDataFrame Dtypes before scaling:")
    print(df.dtypes)
    
    print("\nAttempting scaling...")
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
