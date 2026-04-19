"""
Shared pytest fixtures for the property-price-prediction test suite.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on the path regardless of how pytest is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─── Sample property dicts ────────────────────────────────────────

@pytest.fixture
def valid_property():
    """A fully-specified, valid property input dict."""
    return {
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


@pytest.fixture
def minimal_property():
    """Smallest-sensible property input."""
    return {
        'area': 1000,
        'bedrooms': 1,
        'bathrooms': 1,
        'stories': 1,
        'mainroad': 'no',
        'guestroom': 'no',
        'basement': 'no',
        'hotwaterheating': 'no',
        'airconditioning': 'no',
        'parking': 0,
        'prefarea': 'no',
        'furnishingstatus': 'unfurnished',
    }


@pytest.fixture
def sample_housing_df():
    """Small synthetic DataFrame matching the Housing.csv schema."""
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame({
        'price':          rng.integers(2_000_000, 14_000_000, n),
        'area':           rng.integers(1650, 16_200, n),
        'bedrooms':       rng.integers(1, 7, n),
        'bathrooms':      rng.integers(1, 5, n),
        'stories':        rng.integers(1, 5, n),
        'mainroad':       rng.choice(['yes', 'no'], n),
        'guestroom':      rng.choice(['yes', 'no'], n),
        'basement':       rng.choice(['yes', 'no'], n),
        'hotwaterheating': rng.choice(['yes', 'no'], n),
        'airconditioning': rng.choice(['yes', 'no'], n),
        'parking':        rng.integers(0, 4, n),
        'prefarea':       rng.choice(['yes', 'no'], n),
        'furnishingstatus': rng.choice(
            ['furnished', 'semi-furnished', 'unfurnished'], n
        ),
    })
