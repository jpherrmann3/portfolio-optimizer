"""
Portfolio Optimizer Package
============================

A comprehensive toolkit for portfolio optimization with ML forecasting and risk modeling.

Modules:
- data_loader: Load and preprocess financial data
- feature_engineering: Create features for ML models
- models: ML-based forecasting models
- risk_model: Risk estimation and metrics
- optimizer: Portfolio optimization strategies
- backtesting: Performance evaluation and backtesting
"""

__version__ = "0.1.0"
__author__ = "James Herrmann"

# Import main components for easy access
from . import data_loader
from . import feature_engineering
from . import models
from . import risk_model
from . import optimizer
from . import backtesting

__all__ = [
    "data_loader",
    "feature_engineering",
    "models",
    "risk_model",
    "optimizer",
    "backtesting",
]
