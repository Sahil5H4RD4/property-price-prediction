"""
Unit tests for src/agent.py

Tests cover:
- analyze_input: correct analysis string construction
- retrieve_market_data: vectorstore absent path
- compare_properties: exact match, progressive fallback, no match
- run_advisory_agent: missing GROQ_API_KEY raises EnvironmentError
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.agent import analyze_input, retrieve_market_data, compare_properties, AgentState


def _make_state(area=6000, bedrooms=3, price=5_000_000.0) -> AgentState:
    return {
        'property_details': {
            'area': area,
            'bedrooms': bedrooms,
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
        },
        'predicted_price': price,
        'model_metrics': {'model_r2': 0.85},
        'market_context': '',
        'comparable_summary': '',
        'analysis': '',
        'report': '',
    }


# ─── analyze_input ───────────────────────────────────────────────

class TestAnalyzeInput:
    def test_analysis_contains_area(self):
        state = _make_state(area=7500)
        result = analyze_input(state)
        assert '7500' in result['analysis']

    def test_analysis_contains_bedrooms(self):
        state = _make_state(bedrooms=4)
        result = analyze_input(state)
        assert '4' in result['analysis']

    def test_analysis_contains_price(self):
        state = _make_state(price=4_500_000)
        result = analyze_input(state)
        assert '4,500,000' in result['analysis']

    def test_returns_dict_with_analysis_key(self):
        state = _make_state()
        result = analyze_input(state)
        assert 'analysis' in result
        assert isinstance(result['analysis'], str)


# ─── retrieve_market_data ────────────────────────────────────────

class TestRetrieveMarketData:
    def test_returns_fallback_message_when_vectorstore_absent(self):
        state = _make_state()
        with patch('src.agent._load_vectorstore', return_value=None):
            result = retrieve_market_data(state)
        assert 'market_context' in result
        assert len(result['market_context']) > 0

    def test_returns_context_from_vectorstore(self):
        state = _make_state()
        mock_doc = MagicMock()
        mock_doc.page_content = "Indian real estate market growing at 8% CAGR."
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]

        with patch('src.agent._load_vectorstore', return_value=mock_vs):
            result = retrieve_market_data(state)

        assert 'Indian real estate' in result['market_context']


# ─── compare_properties ──────────────────────────────────────────

class TestCompareProperties:
    def _make_housing_df(self, n=30):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            'price': rng.integers(2_000_000, 12_000_000, n),
            'area': rng.integers(4000, 9000, n),
            'bedrooms': rng.choice([2, 3, 4], n),
        })

    def test_finds_comparables_for_matching_area_and_beds(self):
        df = self._make_housing_df(50)
        state = _make_state(area=6000, bedrooms=3)

        with patch('src.agent.os.path.exists', return_value=True), \
             patch('src.agent.pd.read_csv', return_value=df):
            result = compare_properties(state)

        assert 'comparable_summary' in result
        assert 'Found' in result['comparable_summary']

    def test_no_data_file_returns_graceful_message(self):
        state = _make_state()
        with patch('src.agent.os.path.exists', return_value=False):
            result = compare_properties(state)
        assert 'unavailable' in result['comparable_summary'].lower()

    def test_zero_comparables_returns_not_found_message(self):
        # Use an area so far from the data that nothing matches even with
        # the widest (±50%) fallback band.
        df = self._make_housing_df(20)  # area range 4000–9000
        state = _make_state(area=100, bedrooms=3)

        with patch('src.agent.os.path.exists', return_value=True), \
             patch('src.agent.pd.read_csv', return_value=df):
            result = compare_properties(state)

        assert 'No comparable' in result['comparable_summary']

    def test_progressive_fallback_relaxes_to_wider_band(self):
        """When exact ±20% match fails, the fallback should still find properties."""
        # Only 2 properties match the tight filter; ±50% no-beds filter should work.
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            'price': [5_000_000] * 10,
            'area': [6000] * 10,
            'bedrooms': [99] * 10,  # bedrooms far from the query's 3
        })
        state = _make_state(area=6000, bedrooms=3)

        with patch('src.agent.os.path.exists', return_value=True), \
             patch('src.agent.pd.read_csv', return_value=df):
            result = compare_properties(state)

        # The widest band (any bedrooms) should find these
        assert 'Found' in result['comparable_summary']


# ─── run_advisory_agent ──────────────────────────────────────────

class TestRunAdvisoryAgent:
    def test_raises_environment_error_without_api_key(self):
        from src.agent import run_advisory_agent
        with patch('src.agent.os.environ', {}): # Empty environment
            with patch('dotenv.load_dotenv'):  # patch where it's defined
                with pytest.raises(EnvironmentError, match="GROQ_API_KEY is not set"):
                    run_advisory_agent({}, 5_000_000.0, {'model_r2': 0.85})
