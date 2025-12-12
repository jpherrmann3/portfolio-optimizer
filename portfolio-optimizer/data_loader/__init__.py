"""
Data Loader Module
==================
Handles data ingestion, cleaning, and preprocessing for portfolio optimization.

Main functions:
- load_prices: Download historical price data
- compute_returns: Calculate returns from prices
- load_factors: Fetch factor data (Fama-French)
- merge_factors: Merge factors with returns data
"""

from .loader import (
    load_prices,
    compute_returns,
    load_factors,
    merge_factors,
)

__all__ = [
    "load_prices",
    "compute_returns",
    "load_factors",
    "merge_factors",
]
