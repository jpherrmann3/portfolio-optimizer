"""
Core feature engineering functions for financial time series.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings


def create_lagged_features(
    data: pd.DataFrame,
    lags: Union[int, List[int]] = 5,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create lagged features for time series data.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with time series
    lags : int or list of int, default 5
        Number of lags to create. If int, creates lags 1 to lags.
        If list, creates specific lag values.
    columns : list of str, optional
        Specific columns to create lags for. If None, uses all columns.

    Returns
    -------
    pd.DataFrame
        Original data with lagged features added

    Examples
    --------
    >>> returns = compute_returns(prices)
    >>> lagged = create_lagged_features(returns, lags=5)
    >>> lagged = create_lagged_features(returns, lags=[1, 5, 21])
    """
    result = data.copy()

    # Determine which columns to lag
    if columns is None:
        columns = data.columns.tolist()

    # Convert lags to list if int
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = lags

    # Create lagged features
    for col in columns:
        for lag in lag_list:
            result[f'{col}_lag{lag}'] = data[col].shift(lag)

    return result


def compute_rolling_stats(
    data: pd.DataFrame,
    window: int = 21,
    stats: Optional[List[str]] = None,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute rolling window statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data
    window : int, default 21
        Rolling window size (e.g., 21 days ≈ 1 month)
    stats : list of str, optional
        Statistics to compute: ['mean', 'std', 'skew', 'kurt', 'min', 'max']
        If None, computes all.
    min_periods : int, optional
        Minimum number of observations required. Defaults to window.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling statistics

    Examples
    --------
    >>> returns = compute_returns(prices)
    >>> rolling = compute_rolling_stats(returns, window=21)
    """
    if stats is None:
        stats = ['mean', 'std', 'skew', 'kurt']

    if min_periods is None:
        min_periods = window

    result = pd.DataFrame(index=data.index)

    for col in data.columns:
        if 'mean' in stats:
            result[f'{col}_roll_mean_{window}'] = data[col].rolling(
                window=window, min_periods=min_periods
            ).mean()

        if 'std' in stats:
            result[f'{col}_roll_std_{window}'] = data[col].rolling(
                window=window, min_periods=min_periods
            ).std()

        if 'skew' in stats:
            result[f'{col}_roll_skew_{window}'] = data[col].rolling(
                window=window, min_periods=min_periods
            ).skew()

        if 'kurt' in stats:
            result[f'{col}_roll_kurt_{window}'] = data[col].rolling(
                window=window, min_periods=min_periods
            ).kurt()

        if 'min' in stats:
            result[f'{col}_roll_min_{window}'] = data[col].rolling(
                window=window, min_periods=min_periods
            ).min()

        if 'max' in stats:
            result[f'{col}_roll_max_{window}'] = data[col].rolling(
                window=window, min_periods=min_periods
            ).max()

    return result


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute rolling volatility from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series
    window : int, default 21
        Rolling window size
    annualize : bool, default True
        Whether to annualize volatility (assumes 252 trading days)
    min_periods : int, optional
        Minimum observations required

    Returns
    -------
    pd.DataFrame
        Rolling volatility estimates

    Examples
    --------
    >>> returns = compute_returns(prices)
    >>> volatility = compute_rolling_volatility(returns, window=21)
    """
    if min_periods is None:
        min_periods = window

    vol = returns.rolling(window=window, min_periods=min_periods).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def compute_technical_indicators(
    prices: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute common technical indicators.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    indicators : list of str, optional
        Indicators to compute: ['sma', 'ema', 'rsi', 'bollinger', 'momentum']
        If None, computes all.

    Returns
    -------
    pd.DataFrame
        DataFrame with technical indicators

    Examples
    --------
    >>> tech_indicators = compute_technical_indicators(prices)
    """
    if indicators is None:
        indicators = ['sma', 'ema', 'rsi', 'momentum']

    result = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        if 'sma' in indicators:
            result[f'{col}_sma_20'] = compute_sma(prices[col], window=20)
            result[f'{col}_sma_50'] = compute_sma(prices[col], window=50)

        if 'ema' in indicators:
            result[f'{col}_ema_12'] = compute_ema(prices[col], span=12)
            result[f'{col}_ema_26'] = compute_ema(prices[col], span=26)

        if 'rsi' in indicators:
            result[f'{col}_rsi_14'] = compute_rsi(prices[col], window=14)

        if 'momentum' in indicators:
            result[f'{col}_momentum_10'] = prices[col].pct_change(periods=10)
            result[f'{col}_momentum_21'] = prices[col].pct_change(periods=21)

        if 'bollinger' in indicators:
            bb_mid, bb_upper, bb_lower = compute_bollinger_bands(prices[col], window=20)
            result[f'{col}_bb_mid'] = bb_mid
            result[f'{col}_bb_upper'] = bb_upper
            result[f'{col}_bb_lower'] = bb_lower
            result[f'{col}_bb_width'] = (bb_upper - bb_lower) / bb_mid

    return result


def compute_sma(
    series: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute Simple Moving Average.

    Parameters
    ----------
    series : pd.Series
        Price series
    window : int, default 20
        Window size

    Returns
    -------
    pd.Series
        SMA values
    """
    return series.rolling(window=window).mean()


def compute_ema(
    series: pd.Series,
    span: int = 20
) -> pd.Series:
    """
    Compute Exponential Moving Average.

    Parameters
    ----------
    series : pd.Series
        Price series
    span : int, default 20
        Span for EMA calculation

    Returns
    -------
    pd.Series
        EMA values
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(
    series: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Price series
    window : int, default 14
        Lookback window

    Returns
    -------
    pd.Series
        RSI values (0-100)

    Notes
    -----
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate rolling averages
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    series : pd.Series
        Price series
    window : int, default 20
        Window size for moving average
    num_std : float, default 2.0
        Number of standard deviations for bands

    Returns
    -------
    tuple of pd.Series
        (middle_band, upper_band, lower_band)
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return middle, upper, lower


def create_momentum_features(
    prices: pd.DataFrame,
    periods: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create momentum features at multiple time horizons.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    periods : list of int, optional
        Momentum periods. Defaults to [5, 10, 21, 63] (1w, 2w, 1m, 3m)

    Returns
    -------
    pd.DataFrame
        Momentum features

    Examples
    --------
    >>> momentum = create_momentum_features(prices, periods=[10, 21, 63])
    """
    if periods is None:
        periods = [5, 10, 21, 63]

    result = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        for period in periods:
            # Price momentum (percentage change)
            result[f'{col}_momentum_{period}'] = prices[col].pct_change(periods=period)

            # Price relative to moving average
            sma = prices[col].rolling(window=period).mean()
            result[f'{col}_price_to_sma_{period}'] = prices[col] / sma - 1

    return result


def detect_regimes(
    returns: pd.DataFrame,
    n_regimes: int = 3,
    method: str = "kmeans",
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Detect market regimes using clustering or HMM.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series
    n_regimes : int, default 3
        Number of regimes to detect (e.g., bull, bear, sideways)
    method : str, default 'kmeans'
        Detection method: 'kmeans' or 'hmm'
    features : list of str, optional
        Features for regime detection. Defaults to ['return', 'volatility']

    Returns
    -------
    pd.DataFrame
        DataFrame with regime labels and probabilities

    Examples
    --------
    >>> regimes = detect_regimes(returns, n_regimes=3, method='kmeans')
    """
    # Prepare features for regime detection
    feature_data = []
    feature_names = []

    if features is None:
        features = ['return', 'volatility']

    for col in returns.columns:
        if 'return' in features:
            feature_data.append(returns[col])
            feature_names.append(f'{col}_return')

        if 'volatility' in features:
            vol = returns[col].rolling(window=21).std()
            feature_data.append(vol)
            feature_names.append(f'{col}_volatility')

    # Combine features
    X = pd.concat(feature_data, axis=1)
    X.columns = feature_names

    # Drop NaN rows
    X_clean = X.dropna()

    if len(X_clean) == 0:
        warnings.warn("Not enough data for regime detection")
        return pd.DataFrame(index=returns.index)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    if method.lower() == "kmeans":
        # K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Create result DataFrame
        result = pd.DataFrame(index=returns.index)
        result['regime'] = np.nan
        result.loc[X_clean.index, 'regime'] = labels

        # Forward fill regime labels
        result['regime'] = result['regime'].fillna(method='ffill')

        # Add one-hot encoded regimes
        for i in range(n_regimes):
            result[f'regime_{i}'] = (result['regime'] == i).astype(int)

        return result

    elif method.lower() == "hmm":
        try:
            from hmmlearn import hmm
        except ImportError:
            warnings.warn(
                "hmmlearn not installed. Install with: pip install hmmlearn\n"
                "Falling back to kmeans method."
            )
            return detect_regimes(returns, n_regimes, method="kmeans", features=features)

        # Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

        model.fit(X_scaled)
        hidden_states = model.predict(X_scaled)

        # Create result DataFrame
        result = pd.DataFrame(index=returns.index)
        result['regime'] = np.nan
        result.loc[X_clean.index, 'regime'] = hidden_states

        # Forward fill
        result['regime'] = result['regime'].fillna(method='ffill')

        # Add regime probabilities
        state_probs = model.predict_proba(X_scaled)
        for i in range(n_regimes):
            result[f'regime_{i}_prob'] = np.nan
            result.loc[X_clean.index, f'regime_{i}_prob'] = state_probs[:, i]
            result[f'regime_{i}_prob'] = result[f'regime_{i}_prob'].fillna(method='ffill')

        return result

    else:
        raise ValueError(f"Method '{method}' not supported. Use 'kmeans' or 'hmm'.")


def compute_cross_sectional_features(
    returns: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Compute cross-sectional features across assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series for multiple assets
    window : int, default 21
        Rolling window for statistics

    Returns
    -------
    pd.DataFrame
        Cross-sectional features for each asset

    Examples
    --------
    >>> cross_features = compute_cross_sectional_features(returns)
    """
    result = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        # Rank within cross-section (percentile)
        result[f'{col}_rank'] = returns[col].rolling(window=window).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else np.nan
        )

        # Z-score relative to cross-section
        mean = returns.rolling(window=window).mean()
        std = returns.rolling(window=window).std()
        result[f'{col}_zscore'] = (returns[col] - mean[col]) / std[col]

        # Relative performance vs market (assuming first column is market)
        if len(returns.columns) > 1:
            market = returns.iloc[:, 0]
            result[f'{col}_vs_market'] = returns[col] - market

    return result


def create_interaction_features(
    data: pd.DataFrame,
    interactions: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Create interaction features between variables.

    Parameters
    ----------
    data : pd.DataFrame
        Input features
    interactions : list of tuple, optional
        Pairs of columns to interact. If None, creates all pairwise interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with interaction features

    Examples
    --------
    >>> interactions = create_interaction_features(
    ...     features,
    ...     interactions=[('AAPL_return', 'MSFT_return')]
    ... )
    """
    result = pd.DataFrame(index=data.index)

    if interactions is None:
        # Create all pairwise interactions (can be expensive)
        cols = data.columns.tolist()
        interactions = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i+1, len(cols))]

    for col1, col2 in interactions:
        if col1 in data.columns and col2 in data.columns:
            result[f'{col1}_x_{col2}'] = data[col1] * data[col2]

    return result


def create_time_features(
    index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Create time-based features from datetime index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Datetime index

    Returns
    -------
    pd.DataFrame
        Time-based features

    Examples
    --------
    >>> time_features = create_time_features(returns.index)
    """
    result = pd.DataFrame(index=index)

    # Calendar features
    result['day_of_week'] = index.dayofweek
    result['day_of_month'] = index.day
    result['week_of_year'] = index.isocalendar().week
    result['month'] = index.month
    result['quarter'] = index.quarter
    result['year'] = index.year

    # Cyclical encoding for periodic features
    result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 5)
    result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 5)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

    # Special days
    result['is_month_start'] = index.is_month_start.astype(int)
    result['is_month_end'] = index.is_month_end.astype(int)
    result['is_quarter_start'] = index.is_quarter_start.astype(int)
    result['is_quarter_end'] = index.is_quarter_end.astype(int)
    result['is_year_start'] = index.is_year_start.astype(int)
    result['is_year_end'] = index.is_year_end.astype(int)

    return result
