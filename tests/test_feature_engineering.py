"""
Unit tests for feature_engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_optimizer.feature_engineering import (
    create_lagged_features,
    compute_rolling_stats,
    compute_rolling_volatility,
    compute_technical_indicators,
    detect_regimes,
    create_momentum_features,
    compute_rsi,
    compute_sma,
    compute_ema,
    compute_bollinger_bands,
)


class TestLaggedFeatures:
    """Tests for lagged features"""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.randn(50) * 0.02,
            'MSFT': np.random.randn(50) * 0.02
        }, index=dates)
        return returns

    def test_create_lagged_features_int(self, sample_returns):
        """Test creating lagged features with integer lag"""
        lagged = create_lagged_features(sample_returns, lags=3)

        # Should have original + lagged columns
        assert 'AAPL' in lagged.columns
        assert 'AAPL_lag1' in lagged.columns
        assert 'AAPL_lag2' in lagged.columns
        assert 'AAPL_lag3' in lagged.columns
        assert 'MSFT_lag1' in lagged.columns

        # Check lag values are correct
        assert lagged['AAPL_lag1'].iloc[1] == sample_returns['AAPL'].iloc[0]
        assert lagged['AAPL_lag2'].iloc[2] == sample_returns['AAPL'].iloc[0]

    def test_create_lagged_features_list(self, sample_returns):
        """Test creating specific lags"""
        lagged = create_lagged_features(sample_returns, lags=[1, 5, 10])

        assert 'AAPL_lag1' in lagged.columns
        assert 'AAPL_lag5' in lagged.columns
        assert 'AAPL_lag10' in lagged.columns
        assert 'AAPL_lag2' not in lagged.columns

    def test_create_lagged_features_specific_columns(self, sample_returns):
        """Test lagging only specific columns"""
        lagged = create_lagged_features(sample_returns, lags=2, columns=['AAPL'])

        assert 'AAPL_lag1' in lagged.columns
        assert 'MSFT_lag1' not in lagged.columns


class TestRollingStats:
    """Tests for rolling statistics"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'returns': np.random.randn(100) * 0.02
        }, index=dates)
        return data

    def test_compute_rolling_stats_mean(self, sample_data):
        """Test rolling mean calculation"""
        rolling = compute_rolling_stats(sample_data, window=10, stats=['mean'])

        assert 'returns_roll_mean_10' in rolling.columns
        # First 9 values should be NaN (min_periods=window)
        assert rolling['returns_roll_mean_10'].iloc[:9].isna().all()
        # 10th value should not be NaN
        assert not pd.isna(rolling['returns_roll_mean_10'].iloc[9])

    def test_compute_rolling_stats_all(self, sample_data):
        """Test all rolling statistics"""
        rolling = compute_rolling_stats(
            sample_data,
            window=21,
            stats=['mean', 'std', 'skew', 'kurt']
        )

        assert 'returns_roll_mean_21' in rolling.columns
        assert 'returns_roll_std_21' in rolling.columns
        assert 'returns_roll_skew_21' in rolling.columns
        assert 'returns_roll_kurt_21' in rolling.columns

    def test_compute_rolling_volatility(self, sample_data):
        """Test rolling volatility"""
        vol = compute_rolling_volatility(sample_data, window=21, annualize=False)

        assert vol.shape == sample_data.shape
        assert (vol.dropna() >= 0).all().all()

    def test_compute_rolling_volatility_annualized(self, sample_data):
        """Test annualized rolling volatility"""
        vol_annual = compute_rolling_volatility(sample_data, window=21, annualize=True)
        vol_daily = compute_rolling_volatility(sample_data, window=21, annualize=False)

        # Annualized should be higher (by sqrt(252))
        ratio = vol_annual.dropna() / vol_daily.dropna()
        assert np.allclose(ratio, np.sqrt(252))


class TestTechnicalIndicators:
    """Tests for technical indicators"""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create trending price data
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        prices = pd.DataFrame({
            'AAPL': trend + noise
        }, index=dates)
        return prices

    def test_compute_sma(self, sample_prices):
        """Test Simple Moving Average"""
        sma = compute_sma(sample_prices['AAPL'], window=20)

        assert len(sma) == len(sample_prices)
        assert sma.iloc[:19].isna().all()  # First 19 should be NaN
        assert not pd.isna(sma.iloc[19])  # 20th should have value

    def test_compute_ema(self, sample_prices):
        """Test Exponential Moving Average"""
        ema = compute_ema(sample_prices['AAPL'], span=20)

        assert len(ema) == len(sample_prices)
        # EMA responds faster than SMA
        assert not pd.isna(ema.iloc[0])

    def test_compute_rsi(self, sample_prices):
        """Test RSI calculation"""
        rsi = compute_rsi(sample_prices['AAPL'], window=14)

        assert len(rsi) == len(sample_prices)
        # RSI should be between 0 and 100
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

    def test_compute_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands"""
        mid, upper, lower = compute_bollinger_bands(sample_prices['AAPL'], window=20)

        assert len(mid) == len(sample_prices)
        assert len(upper) == len(sample_prices)
        assert len(lower) == len(sample_prices)

        # Upper should be >= middle >= lower
        valid_data = ~(mid.isna() | upper.isna() | lower.isna())
        assert (upper[valid_data] >= mid[valid_data]).all()
        assert (mid[valid_data] >= lower[valid_data]).all()

    def test_compute_technical_indicators(self, sample_prices):
        """Test computing multiple technical indicators"""
        tech = compute_technical_indicators(sample_prices, indicators=['sma', 'rsi'])

        assert 'AAPL_sma_20' in tech.columns
        assert 'AAPL_rsi_14' in tech.columns
        assert len(tech) == len(sample_prices)


class TestMomentumFeatures:
    """Tests for momentum features"""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 100,
            'MSFT': np.random.randn(100).cumsum() + 200
        }, index=dates)
        return prices

    def test_create_momentum_features(self, sample_prices):
        """Test momentum feature creation"""
        momentum = create_momentum_features(sample_prices, periods=[10, 21])

        assert 'AAPL_momentum_10' in momentum.columns
        assert 'AAPL_momentum_21' in momentum.columns
        assert 'AAPL_price_to_sma_10' in momentum.columns
        assert len(momentum) == len(sample_prices)

    def test_momentum_calculation(self, sample_prices):
        """Test momentum values are correct"""
        momentum = create_momentum_features(sample_prices, periods=[5])

        # Check momentum calculation
        expected = sample_prices['AAPL'].pct_change(periods=5)
        assert np.allclose(
            momentum['AAPL_momentum_5'].dropna(),
            expected.dropna(),
            rtol=1e-5
        )


class TestRegimeDetection:
    """Tests for regime detection"""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data with distinct regimes"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        # Create three regimes: bull, bear, sideways
        regime1 = np.random.normal(0.001, 0.01, 100)  # Bull
        regime2 = np.random.normal(-0.001, 0.02, 100)  # Bear (higher vol)
        regime3 = np.random.normal(0.0, 0.005, 100)  # Sideways (low vol)

        returns = pd.DataFrame({
            'market': np.concatenate([regime1, regime2, regime3])
        }, index=dates)
        return returns

    def test_detect_regimes_kmeans(self, sample_returns):
        """Test K-means regime detection"""
        regimes = detect_regimes(sample_returns, n_regimes=3, method='kmeans')

        assert 'regime' in regimes.columns
        assert len(regimes) == len(sample_returns)

        # Should have 3 unique regimes
        unique_regimes = regimes['regime'].dropna().unique()
        assert len(unique_regimes) <= 3

        # Check one-hot encoding
        assert 'regime_0' in regimes.columns
        assert 'regime_1' in regimes.columns
        assert 'regime_2' in regimes.columns

    def test_detect_regimes_features(self, sample_returns):
        """Test regime detection with specific features"""
        regimes = detect_regimes(
            sample_returns,
            n_regimes=2,
            method='kmeans',
            features=['return', 'volatility']
        )

        assert 'regime' in regimes.columns
        assert len(regimes) == len(sample_returns)


class TestIntegration:
    """Integration tests combining multiple features"""

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 100,
            'MSFT': np.random.randn(100).cumsum() + 200
        }, index=dates)
        returns = prices.pct_change().dropna()
        return prices, returns

    def test_full_feature_pipeline(self, sample_data):
        """Test creating a complete feature set"""
        prices, returns = sample_data

        # 1. Lagged features
        lagged = create_lagged_features(returns, lags=5)
        assert len(lagged.columns) > len(returns.columns)

        # 2. Rolling statistics
        rolling = compute_rolling_stats(returns, window=10)
        assert len(rolling) == len(returns)

        # 3. Technical indicators
        tech = compute_technical_indicators(prices)
        assert len(tech) == len(prices)

        # 4. Momentum features
        momentum = create_momentum_features(prices)
        assert len(momentum) == len(prices)

        # All should have same index
        assert lagged.index.equals(returns.index)
        assert rolling.index.equals(returns.index)
        assert tech.index.equals(prices.index)
        assert momentum.index.equals(prices.index)

    def test_feature_combination(self, sample_data):
        """Test combining multiple feature sets"""
        prices, returns = sample_data

        # Create various features
        lagged = create_lagged_features(returns, lags=3)
        rolling = compute_rolling_stats(returns, window=10)
        tech = compute_technical_indicators(prices)

        # Combine them
        combined = pd.concat([lagged, rolling, tech], axis=1)

        # Should have all features
        assert len(combined.columns) > len(lagged.columns)
        assert len(combined) == len(returns)

    def test_no_data_leakage(self, sample_data):
        """Test that features don't leak future information"""
        prices, returns = sample_data

        # Lagged features should use past data only
        lagged = create_lagged_features(returns, lags=1)

        # lag1 at time t should equal value at time t-1
        for i in range(1, len(returns)):
            for col in returns.columns:
                assert lagged[f'{col}_lag1'].iloc[i] == returns[col].iloc[i-1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
