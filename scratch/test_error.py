"""
Manual integration smoke test for the preprocessing pipeline.

Run from project root after training models:
    python scratch/test_error.py

This script was originally used during development to debug the pandas
StringDtype compatibility issue (pd.Series(['yes'], dtype='string')[0]
produces a StringDtype scalar that was not handled by the 'yes'/'no'
mapping). That bug is fixed in src/preprocess.py.

The script is kept as a quick sanity check that the full pipeline works
end-to-end on a known input without requiring pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_price

INPUT = {
    'area': 5000,
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

if __name__ == '__main__':
    print(f"Python {sys.version}")
    print("Running smoke test on sample property...\n")
    try:
        result = predict_price(INPUT)
        print(f"Predicted Price : \u20b9{result['predicted_price']:>15,.2f}")
        print(f"Model           : {result['model_name']}")
        print(f"R\u00b2 Score         : {result['model_r2']:.4f}")
        print(f"MAE             : \u20b9{result['model_mae']:>12,.0f}")
        print("\nSmoke test PASSED")
    except Exception as exc:
        import traceback
        print(f"Smoke test FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)
