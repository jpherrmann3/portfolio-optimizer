"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_optimizer.data_loader import (
    load_prices,
    compute_returns,
    load_factors,
    merge_factors,
)

class TestLoadPrices:
    """Tests for load_prices function"""

    def test_load_single_ticker(self):
        """Test loading a single ticker"""
        prices = load_prices('AAPL', start='2023-01-01', end='2023-12-31')

        assert isinstance(prices, pd.DataFrame)
        assert 'AAPL' in prices.columns
        assert len(prices) > 0
        assert prices.index.name == 'Date' or isinstance(prices.index, pd.DatetimeIndex)

    def test_load_multiple_tickers(self):
        """Test loading multiple tickers"""
        tickers = ['AAPL', 'MSFT', 'GOOG']
        prices = load_prices(tickers, start='2023-01-01', end='2023-12-31')

        assert isinstance(prices, pd.DataFrame)
        assert all(ticker in prices.columns for ticker in tickers)
        assert len(prices) > 0

    def test_load_with_default_end_date(self):
        """Test loading without specifying end date"""
        prices = load_prices('AAPL', start='2023-01-01')

        assert isinstance(prices, pd.DataFrame)
        assert len(prices) > 0

    def test_invalid_ticker(self):
        """Test handling of invalid ticker"""
        # This should either raise an error or return empty/NaN data
        # Depending on yfinance behavior
        try:
            prices = load_prices('INVALID_TICKER_XYZ', start='2023-01-01', end='2023-12-31')
            # If it doesn't raise an error, check for empty or all-NaN data
            assert len(prices) == 0 or prices.isna().all().all()
        except Exception:
            pass  # Expected behavior

    def test_date_range(self):
        """Test that returned data is within specified date range"""
        start = '2023-01-01'
        end = '2023-06-30'
        prices = load_prices('AAPL', start=start, end=end)

        assert prices.index.min() >= pd.to_datetime(start)
        assert prices.index.max() <= pd.to_datetime(end)


class TestComputeReturns:
    """Tests for compute_returns function"""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'AAPL': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            'MSFT': [200, 202, 204, 203, 205, 207, 206, 208, 210, 209]
        }, index=dates)
        return prices

    def test_simple_returns(self, sample_prices):
        """Test simple return calculation"""
        returns = compute_returns(sample_prices, method='simple')

        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] == sample_prices.shape[0] - 1
        assert all(col in returns.columns for col in sample_prices.columns)

        # Check first return is correct: (102-100)/100 = 0.02
        assert abs(returns.iloc[0]['AAPL'] - 0.02) < 1e-6

    def test_log_returns(self, sample_prices):
        """Test log return calculation"""
        returns = compute_returns(sample_prices, method='log')

        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] == sample_prices.shape[0] - 1

        # Log return should be close to simple return for small changes
        simple_returns = compute_returns(sample_prices, method='simple')
        assert np.allclose(returns.values, simple_returns.values, atol=0.01)

    def test_multi_period_returns(self, sample_prices):
        """Test multi-period return calculation"""
        returns = compute_returns(sample_prices, method='simple', periods=2)

        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] == sample_prices.shape[0] - 2

    def test_invalid_method(self, sample_prices):
        """Test error handling for invalid method"""
        with pytest.raises(ValueError):
            compute_returns(sample_prices, method='invalid')

    def test_dropna_parameter(self, sample_prices):
        """Test dropna parameter"""
        returns_dropped = compute_returns(sample_prices, dropna=True)
        returns_kept = compute_returns(sample_prices, dropna=False)

        assert len(returns_dropped) < len(returns_kept)
        assert returns_kept.isna().any().any()
        assert not returns_dropped.isna().any().any()


class TestLoadFactors:
    """Tests for load_factors function"""

    def test_load_daily_factors(self):
        """Test loading daily factor data"""
        factors = load_factors(start='2023-01-01', end='2023-12-31', frequency='daily')

        assert isinstance(factors, pd.DataFrame)
        assert len(factors) > 0
        assert isinstance(factors.index, pd.DatetimeIndex)

        # Check for common factor columns
        expected_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        assert any(factor in factors.columns for factor in expected_factors)

    def test_load_monthly_factors(self):
        """Test loading monthly factor data"""
        factors = load_factors(start='2023-01-01', end='2023-12-31', frequency='monthly')

        assert isinstance(factors, pd.DataFrame)
        assert len(factors) > 0

    def test_load_specific_factors(self):
        """Test loading specific factors"""
        requested_factors = ['Mkt-RF', 'SMB']
        factors = load_factors(
            start='2023-01-01',
            end='2023-12-31',
            factors=requested_factors
        )

        assert isinstance(factors, pd.DataFrame)
        # Should have at least some of the requested factors
        assert any(f in factors.columns for f in requested_factors)

    def test_factor_date_range(self):
        """Test that factors are within specified date range"""
        start = '2023-01-01'
        end = '2023-06-30'
        factors = load_factors(start=start, end=end)

        # Allow some flexibility for month-end dates
        assert factors.index.min() >= pd.to_datetime(start) - timedelta(days=5)
        assert factors.index.max() <= pd.to_datetime(end) + timedelta(days=5)


class TestMergeFactors:
    """Tests for merge_factors function"""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.randn(10) * 0.02,
            'MSFT': np.random.randn(10) * 0.02
        }, index=dates)
        return returns

    @pytest.fixture
    def sample_factors(self):
        """Create sample factor data"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        factors = pd.DataFrame({
            'Mkt-RF': np.random.randn(10) * 0.01,
            'SMB': np.random.randn(10) * 0.005,
            'RF': np.random.randn(10) * 0.0001
        }, index=dates)
        return factors

    def test_inner_merge(self, sample_returns, sample_factors):
        """Test inner merge of returns and factors"""
        merged = merge_factors(sample_returns, sample_factors, how='inner')

        assert isinstance(merged, pd.DataFrame)
        assert all(col in merged.columns for col in sample_returns.columns)
        assert all(col in merged.columns for col in sample_factors.columns)
        assert len(merged) == min(len(sample_returns), len(sample_factors))

    def test_outer_merge(self, sample_returns, sample_factors):
        """Test outer merge of returns and factors"""
        # Create data with different date ranges
        returns = sample_returns.iloc[:7]
        factors = sample_factors.iloc[3:]

        merged = merge_factors(returns, factors, how='outer')

        assert isinstance(merged, pd.DataFrame)
        assert len(merged) == 10  # Should have all dates

    def test_merge_preserves_data(self, sample_returns, sample_factors):
        """Test that merge preserves original data values"""
        merged = merge_factors(sample_returns, sample_factors, how='inner')

        # Check that values match for a specific date
        test_date = sample_returns.index[0]
        assert np.allclose(
            merged.loc[test_date, sample_returns.columns].values,
            sample_returns.loc[test_date].values
        )


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_pipeline(self):
        """Test complete data loading pipeline"""
        # Load prices
        prices = load_prices(['AAPL', 'MSFT'], start='2023-01-01', end='2023-12-31')
        assert len(prices) > 0

        # Compute returns
        returns = compute_returns(prices)
        assert len(returns) == len(prices) - 1

        # Load factors
        factors = load_factors(start='2023-01-01', end='2023-12-31')
        assert len(factors) > 0

        # Merge
        combined = merge_factors(returns, factors)
        assert len(combined) > 0
        assert 'AAPL' in combined.columns
        assert any('Mkt-RF' in col or 'Mkt-RF' in str(col) for col in combined.columns)

    def test_consistency_checks(self):
        """Test data consistency across operations"""
        prices = load_prices('AAPL', start='2023-01-01', end='2023-03-31')

        # Returns should have one less row than prices
        returns = compute_returns(prices)
        assert len(returns) == len(prices) - 1

        # All returns should be finite
        assert np.isfinite(returns).all().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
