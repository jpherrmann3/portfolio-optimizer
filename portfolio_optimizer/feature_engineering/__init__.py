"""
Feature Engineering Module
===========================
Create ML-ready features from financial time series data.

Main functions:
- create_lagged_features: Generate lagged variables for time series
- compute_rolling_stats: Rolling statistics (mean, vol, skew, kurtosis)
- compute_technical_indicators: Technical indicators (RSI, SMA, momentum)
- detect_regimes: Regime detection using HMM or clustering
"""

from .features import (
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

__all__ = [
    "create_lagged_features",
    "compute_rolling_stats",
    "compute_rolling_volatility",
    "compute_technical_indicators",
    "detect_regimes",
    "create_momentum_features",
    "compute_rsi",
    "compute_sma",
    "compute_ema",
    "compute_bollinger_bands",
]
