from datetime import datetime, timedelta
import re
from typing import Optional,Literal

from openbb import obb
import openbb_core.app.model.obbject
import pandas as pd
import yfinance as yf

import config

def generate_date_list(start_str: str, end_str: str) -> list[str]:
    """
    Generate a list of dates between start_str and end_str (inclusive) in YYYY-MM-DD format

    Parameters
    ----------
    start_str : str
        Start date in YYYY-MM-DD format
    end_str : str
        End date in YYYY-MM-DD format

    Returns
    -------
    list[str]
        List of dates in YYYY-MM-DD format

    Raises
    ------
    ValueError
        If start_str or end_str is not in YYYY-MM-DD format
    """
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    date_list: list[str] = []
    current: datetime = start_date
    while current <= end_date:
        date_list.append(current.strftime("%Y/%m/%d"))
        current += timedelta(days=1)

    return date_list

def maturity_to_years(label: str) -> float:
    """
    Converts a maturity label into a float number of years.

    Parameters
    ----------
    label : str
        Maturity label (e.g. 'month_3', 'year_1')

    Returns
    -------
    float
        Number of years corresponding to the maturity label

    Raises
    ------
    ValueError
        If label is not in the format 'month_<num>' or 'year_<num>'
    """
    num = int(re.search(r"\d+", label).group())
    if label.startswith("month_"):
        return num / 12
    elif label.startswith("year_"):
        return num
    else:
        raise ValueError(f"Unknown maturity label: {label}")

def get_yield_curve(start_str: str, end_str: str, provider: Optional[Literal['ecb', 'econdb', 'federal_reserve', 'fmp', 'fred']] = 'federal_reserve') -> pd.DataFrame:
    """
    Retrieves a yield curve for a given date range from the specified provider.

    Parameters
    ----------
    start_str : str
        Start date of the range in the format '%Y/%m/%d'
    end_str : str
        End date of the range in the format '%Y/%m/%d'
    provider : Optional[Literal['ecb', 'econdb', 'federal_reserve', 'fmp', 'fred']], default='federal_reserve'
        Provider of the yield curve data

    Returns
    -------
    pd.DataFrame
        Dataframe containing the yield curve data with the date as the index and the maturity as the column.

    Raises
    ------
    ValueError
        If start_str or end_str is not in the correct format
    """
    dates: list[str] = generate_date_list(start_str, end_str)
    yield_curve: openbb_core.app.model.obbject.OBBject = obb.fixedincome.government.yield_curve(provider=provider, date=dates)
    curve_dicts: list[dict] = [
        {
            "date": row.date,
            "maturity": row.maturity,
            "rate": row.rate
        }
        for row in yield_curve.results
    ]
    yield_curve_df: pd.DataFrame = pd.DataFrame(curve_dicts)
    yield_curve_df = yield_curve_df.pivot(index='date', columns='maturity', values='rate')
    sorted_cols = sorted(yield_curve_df.columns, key=maturity_to_years)
    yield_curve_df = yield_curve_df[sorted_cols]
    return yield_curve_df
        
def get_vix_index(start_str: str, end_str: str) -> pd.DataFrame:
    """
    Retrieves the VIX index historical data for a given date range.

    Parameters
    ----------
    start_str : str
        Start date of the range in the format '%Y/%m/%d'
    end_str : str
        End date of the range in the format '%Y/%m/%d'

    Returns
    -------
    pd.DataFrame
        Dataframe containing the VIX index historical data with the date as the index and the close value as the column.

    Raises
    ------
    ValueError
        If start_str or end_str is not in the correct format
    """
    vix_data: openbb_core.app.model.obbject.OBBject = obb.equity.price.historical(
        symbol="^VIX",
        provider="yfinance",
        start_date=start_str,
        end_date=end_str,
    )
    vix_data_dicts: list[dict] = [
        {
            "date": row.date,
            "close": row.close,
        }
        for row in vix_data.results
    ]
    vix_data_df: pd.DataFrame = pd.DataFrame(vix_data_dicts)
    vix_data_df.rename(columns={'close': 'vix_index'}, inplace=True)
    return vix_data_df

def get_volatility_skew_index(start_str: str, end_str: str) -> pd.DataFrame:
    """
    Retrieves the volatility skew index historical data for a given date range.

    Parameters
    ----------
    start_str : str
        Start date of the range in the format '%Y/%m/%d'
    end_str : str
        End date of the range in the format '%Y/%m/%d'

    Returns
    -------
    pd.DataFrame
        Dataframe containing the volatility skew index historical data with the date as the index and the close value as the column.

    Raises
    ------
    ValueError
        If start_str or end_str is not in the correct format
    """
    skew_index_df: pd.DataFrame | None = yf.Ticker('^SKEW').history(start=start_str, end=end_str)
    if skew_index_df is None or skew_index_df.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data is fetched
    skew_index_df.reset_index(inplace=True)
    skew_index_df = skew_index_df[['Date', 'Close']]
    skew_index_df.rename(columns={'Date': 'date', 'Close': 'skew_index'}, inplace=True)
    return skew_index_df

def get_underyling_features(symbol: str, start_str: str, end_str: str, volatility_timeframes: list[int] = [5, 10, 20, 50],
                            momentum_timeframes: list[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Retrieves the underlying features for a given symbol and date range.

    Parameters
    ----------
    symbol : str
        Symbol of the underlying asset
    start_str : str
        Start date of the range in the format '%Y/%m/%d'
    end_str : str
        End date of the range in the format '%Y/%m/%d'
    volatility_timeframes : list[int], optional
        List of timeframes (in days) to calculate the volatility features, by default [5, 10, 20, 50]
    momentum_timeframes : list[int], optional
        List of timeframes (in days) to calculate the momentum features, by default [5, 10, 20, 50]

    Returns
    -------
    pd.DataFrame
        Dataframe containing the underlying features with the date as the index and the feature names as the columns.

    Raises
    ------
    ValueError
        If start_str or end_str is not in the correct format
    """
    max_lookback = max(volatility_timeframes + momentum_timeframes)
    start_date = pd.to_datetime(start_str) - pd.Timedelta(days=max_lookback)
    start_str_adjusted = start_date.strftime("%Y-%m-%d")
    vol_data: openbb_core.app.model.obbject.OBBject = obb.equity.price.historical(
        symbol=symbol,
        provider="yfinance",
        start_date=start_str_adjusted,
        end_date=end_str,
    )
    vol_data_dicts: list[dict] = [
        {
            "date": row.date,
            "close": row.close,
        }
        for row in vol_data.results
    ]
    vol_data_df: pd.DataFrame = pd.DataFrame(vol_data_dicts)
    for timeframe in volatility_timeframes:
        vol_data_df[f'vol_{timeframe}d'] = vol_data_df['close'].pct_change().rolling(window=timeframe).std() * (config.TRADING_DAYS_PER_YEAR ** 0.5)
    for timeframe in momentum_timeframes:
        vol_data_df[f'sma_{timeframe}d'] = vol_data_df['close'].rolling(window=timeframe).mean()
        vol_data_df[f'momentum_{timeframe}d'] = vol_data_df['close'] / vol_data_df[f'sma_{timeframe}d'] - 1
        vol_data_df.drop(columns=[f'sma_{timeframe}d'], inplace=True)
    vol_data_df.dropna(inplace=True)
    vol_data_df.rename(columns={'close': 'underlying_price'}, inplace=True)
    return vol_data_df

def create_full_dataset(underyling_df: pd.DataFrame,) -> pd.DataFrame:
    max_lookback = max(volatility_timeframes + momentum_timeframes)
    start_date = pd.to_datetime(start_str) - pd.Timedelta(days=max_lookback)
    start_str_adjusted = start_date.strftime("%Y-%m-%d")
    dfs = [
        get_yield_curve(start_str_adjusted, end_str),
        get_vix_index(start_str_adjusted, end_str),
        get_volatility_skew_index(start_str_adjusted, end_str),
        get_underyling_features(symbol, start_str_adjusted, end_str,
                                volatility_timeframes, momentum_timeframes)
    ]
    for i, df in enumerate(dfs):
        if 'date' not in df.columns:
            df = df.reset_index()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # remove timezone info
        dfs[i] = df
    dataset_df = dfs[0]
    for df in dfs[1:]:
        dataset_df = dataset_df.merge(df, on='date', how='inner')
    dataset_df.set_index('date', inplace=True)
    dataset_df = dataset_df.loc[dataset_df.index >= pd.to_datetime(start_str)]
    return dataset_df

def get_daily_dividend_yield(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetches historical dividends and creates a daily forward-filled annualized yield series.
    This corrected version handles zero-dividend stocks gracefully.
    """
    ticker = yf.Ticker(symbol)
    prices = ticker.history(start=start_date, end=end_date)['Close']

    if prices.empty:
        logger.warning("Could not fetch price data for %s. Assuming 0%% yield.", symbol)
        return pd.DataFrame({'date': [], 'dividend_yield': []})

    dividends = ticker.dividends

    # Handle stocks that have never paid a dividend
    if dividends.empty:
        logger.info("No dividend data found for symbol %s. Assuming 0%% yield.", symbol)
        daily_yield_df = pd.DataFrame(index=prices.index)
        daily_yield_df['dividend_yield'] = 0.0
        daily_yield_df = daily_yield_df.reset_index()
        daily_yield_df.rename(columns={'Date': 'date'}, inplace=True)
        daily_yield_df['date'] = pd.to_datetime(daily_yield_df['date']).dt.tz_localize(None)
        return daily_yield_df

    dividends = dividends.loc[start_date:end_date]
    if dividends.empty:
        logger.info("No dividends paid by %s in the specified date range. Assuming 0%% yield.", symbol)
        daily_yield_df = pd.DataFrame(index=prices.index)
        daily_yield_df['dividend_yield'] = 0.0
        daily_yield_df = daily_yield_df.reset_index()
        daily_yield_df.rename(columns={'Date': 'date'}, inplace=True)
        daily_yield_df['date'] = pd.to_datetime(daily_yield_df['date']).dt.tz_localize(None)
        return daily_yield_df

    df = pd.DataFrame(prices).join(pd.DataFrame(dividends)).dropna(subset=['Dividends'])
    df['annualized_dividend'] = df['Dividends'] * 4 # Assuming quarterly dividends
    df['dividend_yield'] = df['annualized_dividend'] / df['Close']
    
    daily_yield_series = df['dividend_yield'].reindex(prices.index, method='ffill').fillna(0)
    
    daily_yield_df = pd.DataFrame(daily_yield_series).reset_index()
    daily_yield_df.columns = ['date', 'dividend_yield']
    daily_yield_df['date'] = pd.to_datetime(daily_yield_df['date']).dt.tz_localize(None)

    return daily_yield_df
