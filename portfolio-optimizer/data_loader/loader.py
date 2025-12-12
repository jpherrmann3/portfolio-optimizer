"""
Core data loading and preprocessing functions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Union, Optional
from datetime import datetime
import warnings


def load_prices(
    tickers: Union[List[str], str],
    start: Union[str, datetime] = "2020-01-01",
    end: Union[str, datetime] = None,
    source: str = "yahoo",
    column: str = "Adj Close"
) -> pd.DataFrame:
    """
    Download historical price data for specified tickers.

    Parameters
    ----------
    tickers : list of str or str
        Ticker symbols to download (e.g., ['AAPL', 'MSFT', 'GOOG'])
    start : str or datetime, default '2020-01-01'
        Start date for historical data
    end : str or datetime, optional
        End date for historical data (defaults to today)
    source : str, default 'yahoo'
        Data source ('yahoo' is currently supported)
    column : str, default 'Adj Close'
        Price column to extract ('Adj Close', 'Close', 'Open', etc.)

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns

    Examples
    --------
    >>> prices = load_prices(['AAPL', 'MSFT'], start='2020-01-01', end='2023-12-31')
    >>> prices.head()
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    if source.lower() == "yahoo":
        try:
            # Download data
            data = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                show_errors=False
            )

            # Handle single vs multiple tickers
            if len(tickers) == 1:
                if column in data.columns:
                    prices = data[[column]].copy()
                    prices.columns = tickers
                else:
                    raise ValueError(f"Column '{column}' not found in data")
            else:
                if column in data.columns.levels[0]:
                    prices = data[column].copy()
                else:
                    raise ValueError(f"Column '{column}' not found in data")

            # Remove any rows with all NaN
            prices = prices.dropna(how='all')

            # Forward fill missing values (holidays, etc.)
            prices = prices.fillna(method='ffill').fillna(method='bfill')

            return prices

        except Exception as e:
            raise RuntimeError(f"Failed to download data from Yahoo Finance: {str(e)}")
    else:
        raise ValueError(f"Source '{source}' not supported. Use 'yahoo'.")


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
    periods: int = 1,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Compute returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of prices with dates as index
    method : str, default 'simple'
        Return calculation method:
        - 'simple': (P_t - P_{t-1}) / P_{t-1}
        - 'log': log(P_t / P_{t-1})
    periods : int, default 1
        Number of periods for return calculation
        (1 = daily, 5 = weekly, 21 = monthly for daily data)
    dropna : bool, default True
        Whether to drop NaN values

    Returns
    -------
    pd.DataFrame
        DataFrame of returns with same structure as input

    Examples
    --------
    >>> prices = load_prices(['AAPL', 'MSFT'])
    >>> returns = compute_returns(prices, method='simple')
    >>> log_returns = compute_returns(prices, method='log')
    """
    if method.lower() == "simple":
        returns = prices.pct_change(periods=periods)
    elif method.lower() == "log":
        returns = np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'simple' or 'log'.")

    if dropna:
        returns = returns.dropna()

    return returns


def load_factors(
    start: Union[str, datetime] = "2020-01-01",
    end: Union[str, datetime] = None,
    frequency: str = "daily",
    factors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load Fama-French factor data from Kenneth French's data library.

    Parameters
    ----------
    start : str or datetime, default '2020-01-01'
        Start date for factor data
    end : str or datetime, optional
        End date (defaults to today)
    frequency : str, default 'daily'
        Data frequency ('daily', 'monthly')
    factors : list of str, optional
        Specific factors to load. If None, loads common factors.
        Available: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

    Returns
    -------
    pd.DataFrame
        Factor returns with dates as index

    Examples
    --------
    >>> factors = load_factors(start='2020-01-01', frequency='daily')
    >>> factors.head()

    Notes
    -----
    This function fetches data from Kenneth French's Data Library.
    Factor returns are typically in percentage points (e.g., 0.5 = 0.5%).
    """
    try:
        import pandas_datareader as pdr
    except ImportError:
        warnings.warn(
            "pandas_datareader not installed. Install it with: "
            "pip install pandas-datareader"
        )
        # Return a dummy DataFrame for development
        return _create_dummy_factors(start, end, frequency)

    if end is None:
        end = datetime.now()

    # Convert to datetime
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    try:
        # Fetch Fama-French factors
        if frequency.lower() == "daily":
            # FF 5 factors daily
            ff_data = pdr.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start=start_date,
                end=end_date
            )[0]
        else:
            # FF 5 factors monthly
            ff_data = pdr.DataReader(
                'F-F_Research_Data_5_Factors_2x3',
                'famafrench',
                start=start_date,
                end=end_date
            )[0]

        # Convert from percentage to decimal
        ff_data = ff_data / 100.0

        # Filter specific factors if requested
        if factors is not None:
            available_factors = [f for f in factors if f in ff_data.columns]
            if not available_factors:
                raise ValueError(f"None of the requested factors found: {factors}")
            ff_data = ff_data[available_factors]

        return ff_data

    except Exception as e:
        warnings.warn(f"Failed to load Fama-French data: {str(e)}. Using dummy data.")
        return _create_dummy_factors(start, end, frequency)


def _create_dummy_factors(
    start: Union[str, datetime],
    end: Union[str, datetime],
    frequency: str
) -> pd.DataFrame:
    """
    Create dummy factor data for testing when real data is unavailable.

    Internal function - not part of public API.
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end) if end else pd.to_datetime(datetime.now())

    if frequency.lower() == "daily":
        dates = pd.bdate_range(start=start_date, end=end_date)
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='M')

    # Create random factor returns
    np.random.seed(42)
    n = len(dates)

    dummy_data = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0003, 0.01, n),
        'SMB': np.random.normal(0.0001, 0.005, n),
        'HML': np.random.normal(0.0001, 0.005, n),
        'RMW': np.random.normal(0.0001, 0.004, n),
        'CMA': np.random.normal(0.0001, 0.004, n),
        'RF': np.random.normal(0.00001, 0.0001, n),
    }, index=dates)

    return dummy_data


def merge_factors(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    how: str = "inner"
) -> pd.DataFrame:
    """
    Merge asset returns with factor data.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with dates as index
    factors : pd.DataFrame
        Factor returns with dates as index
    how : str, default 'inner'
        Merge method ('inner', 'outer', 'left', 'right')

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with both returns and factors

    Examples
    --------
    >>> returns = compute_returns(load_prices(['AAPL', 'MSFT']))
    >>> factors = load_factors()
    >>> combined = merge_factors(returns, factors)
    >>> print(combined.columns)  # ['AAPL', 'MSFT', 'Mkt-RF', 'SMB', ...]
    """
    # Ensure both have datetime index
    returns.index = pd.to_datetime(returns.index)
    factors.index = pd.to_datetime(factors.index)

    # Merge on date index
    merged = returns.join(factors, how=how)

    # Handle any missing values based on merge type
    if how == "outer":
        # Forward fill then backward fill for outer join
        merged = merged.fillna(method='ffill').fillna(method='bfill')

    return merged


def align_data(
    *dataframes: pd.DataFrame,
    method: str = "inner"
) -> tuple:
    """
    Align multiple DataFrames to have matching date indices.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        Variable number of DataFrames to align
    method : str, default 'inner'
        Alignment method ('inner' or 'outer')

    Returns
    -------
    tuple of pd.DataFrame
        Aligned DataFrames in the same order as input

    Examples
    --------
    >>> prices1 = load_prices(['AAPL'])
    >>> prices2 = load_prices(['MSFT'])
    >>> aligned1, aligned2 = align_data(prices1, prices2, method='inner')
    """
    if not dataframes:
        return tuple()

    # Start with the first DataFrame
    result = [dataframes[0]]

    # Align each subsequent DataFrame
    for df in dataframes[1:]:
        if method == "inner":
            common_dates = result[0].index.intersection(df.index)
            result[0] = result[0].loc[common_dates]
            result.append(df.loc[common_dates])
        else:
            all_dates = result[0].index.union(df.index)
            result[0] = result[0].reindex(all_dates)
            result.append(df.reindex(all_dates))

    return tuple(result)


def handle_missing_data(
    data: pd.DataFrame,
    method: str = "ffill",
    limit: Optional[int] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Handle missing data in price or return series.

    Parameters
    ----------
    data : pd.DataFrame
        Data with potential missing values
    method : str, default 'ffill'
        Method to handle missing data:
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with any missing values
        - 'drop_cols': Drop columns with too many missing values
    limit : int, optional
        Maximum number of consecutive NaNs to fill
    threshold : float, default 0.5
        For 'drop_cols': drop columns with more than this fraction of NaNs

    Returns
    -------
    pd.DataFrame
        Data with missing values handled
    """
    if method == "ffill":
        return data.fillna(method='ffill', limit=limit)
    elif method == "bfill":
        return data.fillna(method='bfill', limit=limit)
    elif method == "interpolate":
        return data.interpolate(method='linear', limit=limit)
    elif method == "drop":
        return data.dropna()
    elif method == "drop_cols":
        # Calculate fraction of NaNs per column
        na_frac = data.isna().sum() / len(data)
        cols_to_keep = na_frac[na_frac <= threshold].index
        return data[cols_to_keep]
    else:
        raise ValueError(
            f"Method '{method}' not supported. "
            f"Use 'ffill', 'bfill', 'interpolate', 'drop', or 'drop_cols'."
        )
