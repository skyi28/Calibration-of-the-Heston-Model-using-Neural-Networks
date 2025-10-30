"""
This script implements a sophisticated framework for calibrating the Heston stochastic
volatility model to daily options market data. It leverages a deep neural network
to predict the Heston parameters (v0, kappa, theta, sigma, rho) based on a rich
set of market and macroeconomic features.

The core methodology involves:
1.  **Feature Engineering**: Downloads and processes a wide array of features,
    including historical underlying volatility, momentum, VIX, SKEW, and the
    yield curve, creating a comprehensive feature set for each trading day.
2.  **Data Handling**: Utilizes a DuckDB database for efficient storage and retrieval
    of both raw options data and engineered features.
3.  **Heston Model Integration**: Employs the QuantLib library for the analytical
    pricing of European options under the Heston model.
4.  **Deep Learning for Calibration**: A TensorFlow/Keras neural network learns the
    mapping from market features to the optimal Heston parameters. A key innovation
    is the use of a residual architecture, where the network predicts a correction
    to a baseline set of parameters.
5.  **Custom Training Loop**: Since the QuantLib pricing engine is not natively
    differentiable in TensorFlow, a custom gradient is implemented using
    `tf.custom_gradient`. The gradient of the loss function with respect to the
    Heston parameters is computed numerically (finite differences), allowing for
    end-to-end training.
6.  **Hyperparameter Tuning**: Keras Tuner is used to systematically search for the
    optimal neural network architecture (e.g., number of layers, neurons, dropout).
7.  **Adaptive Window Evaluation**: The script simulates a real-world scenario using a
    forward-chaining or "adaptive window" methodology. The model is initially trained
    on a historical data segment. It then makes predictions on subsequent days until
    its performance degrades, which is detected using statistical change-point
    analysis. Upon degradation, the model is recalibrated (retrained) using the
    newly accumulated data, mimicking a live production environment.
8.  **Comprehensive Evaluation and Plotting**: After the simulation, the script
    generates a detailed analysis, including daily performance metrics (MAE, RMSE),
    parameter drift over time, and visualizations of the implied volatility surface.

The overall goal is to create a robust, adaptive calibration model that can
dynamically adjust to changing market conditions, outperforming static calibration
methods.
"""
import datetime
import os
import sys
import traceback
import json
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import random

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import QuantLib as ql
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import ruptures as rpt
import keras_tuner as kt
import yfinance as yf
from tqdm import tqdm

# Set the global floating-point policy for TensorFlow to float64.
# This is crucial for numerical stability in financial calculations, especially
# when dealing with small gradients in the custom training loop.
tf.keras.mixed_precision.set_global_policy('float64')

import config
from database import Database
from logger import create_logger
import utils
import plotting

# --- Setup Logger ---
# Initialize the logger to capture script execution details, warnings, and errors.
# This is essential for debugging and monitoring long-running processes.
try:
    logger = create_logger("calibration.log")
    logger.info("Logger initialized successfully.")
except Exception:
    print(f"FATAL: Failed to create logger!\n{traceback.format_exc()}")
    sys.exit()
    
def set_global_seed(seed: int):
    """
    Set global random seeds for Python, NumPy, and TensorFlow to ensure
    reproducibility of results across different runs.

    This function is critical for experiments, as it ensures that any stochastic
    processes (like weight initialization in neural networks or random data splits)
    produce the same outcome every time the script is executed with the same seed.

    Parameters
    ----------
    seed : int
        The integer value to be used as the seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    msg = f"Global seed set to {seed} for reproducibility."
    logger.info(msg)
    print(msg)

# --- Setup Database ---
# Establish a connection to the DuckDB database. If the main options data table
# does not exist, it's created from a source CSV file.
try:
    logger.info("--- Database Initialization ---")
    db: Database = Database(config.DB_NAME)
    if not db.check_if_table_exists(config.OPTIONS_DATA_TABLE):
        logger.info(f"Table '{config.OPTIONS_DATA_TABLE}' not found. Creating from CSV...")
        db.create_table_from_csv(config.OPTIONS_DATA_TABLE, config.PATH_TO_OPTIONS_CSV)
        logger.info(f"Table '{config.OPTIONS_DATA_TABLE}' created successfully.")
    else:
        logger.info(f"Table '{config.OPTIONS_DATA_TABLE}' already exists.")
except Exception:
    logger.critical("Failed to initialize the database.", exc_info=True)
    sys.exit()

# --- Check if option data exists ---
# A sanity check to ensure the options data is available before proceeding.
try:
    if not db.check_if_table_exists(config.OPTIONS_DATA_TABLE):
        logger.critical(f"Option data table '{config.OPTIONS_DATA_TABLE}' does not exist. Download from: https://www.dolthub.com/repositories/post-no-preference/options/data/master")
    else:
        logger.info("Option data check passed.")
except Exception:
    logger.warning("Failed to check if option data exists.", exc_info=True)

# --- Check if feature data exists ---
# A sanity check to ensure the feature data is available before proceeding.
try:
    if not db.check_if_table_exists(config.FEATURES_DATA_TABLE):
            logger.info(f"Table '{config.FEATURES_DATA_TABLE}' not found. Creating from CSV...")
            db.create_table_from_csv(config.FEATURES_DATA_TABLE, config.PATH_TO_FEATURES_CSV)
            logger.info(f"Table '{config.FEATURES_DATA_TABLE}' created successfully.")
    else:
        logger.info(f"Table '{config.FEATURES_DATA_TABLE}' already exists.")
except Exception:
    logger.warning("Failed to check if feature data exists.", exc_info=True)

# --- Add features to option data ---
# This block controls the feature engineering pipeline. If DOWNLOAD_FEATURES is
# True in the config, it generates and saves a comprehensive feature set.
# This is a one-time process (unless the config is changed) to avoid expensive
# re-computation on every run.
if config.DOWNLOAD_FEATURES:
    logger.info("\n--- Feature Engineering ---")
    try:
        # Determine the full date range of the available options data for the target symbol.
        query_min: str = f'SELECT MIN(date) FROM {config.OPTIONS_DATA_TABLE} WHERE act_symbol = ?'
        min_date: datetime.datetime = db.execute_read_query(query_min, args=(config.SYMBOL,), return_as_dataframe=False)[0][0]
        query_max: str = f'SELECT MAX(date) FROM {config.OPTIONS_DATA_TABLE} WHERE act_symbol = ?'
        max_date: datetime.datetime = db.execute_read_query(query_max, args=(config.SYMBOL,), return_as_dataframe=False)[0][0]
        logger.info(f"Option data date range for '{config.SYMBOL}': {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.")
    except Exception:
        logger.critical("Failed to get min and max date from the database.", exc_info=True)
        sys.exit()

    # Get all distinct symbols present in the database to process features for all of them.
    query_symbols: str = f'SELECT DISTINCT act_symbol FROM {config.OPTIONS_DATA_TABLE} ORDER BY act_symbol'
    distinct_symbols_in_database: pd.DataFrame | list | None = db.execute_read_query(query_symbols)
    if not isinstance(distinct_symbols_in_database, pd.DataFrame):
        logger.critical("Failed to get distinct symbols from the database.")
        sys.exit()
    logger.info("Fetching non-underlying features (Yield Curve, VIX, SKEW)...")
    non_underlying_features_dfs: list[pd.DataFrame] = []
    # Attempt to download broad market features that are not specific to the underlying asset.
    try:
        non_underlying_features_dfs.append(utils.get_yield_curve(min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")))
    except Exception:
        logger.warning("Failed to create yield curve dataset.", exc_info=True)
    try: 
        non_underlying_features_dfs.append(utils.get_vix_index(min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")))
    except Exception:
        logger.warning("Failed to create VIX index dataset.", exc_info=True)
    try: non_underlying_features_dfs.append(utils.get_volatility_skew_index(min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")))
    except Exception:
        logger.warning("Failed to create volatility skew index dataset.", exc_info=True)

    # Standardize the 'date' column format across all downloaded feature dataframes.
    for i, df in enumerate(non_underlying_features_dfs):
        df_processed = df.reset_index() if 'date' not in df.columns else df
        df_processed['date'] = pd.to_datetime(df_processed['date']).dt.tz_localize(None)
        non_underlying_features_dfs[i] = df_processed

    if non_underlying_features_dfs:
        # Merge all non-underlying feature dataframes into a single dataframe based on the date.
        all_non_underlying_features_df = non_underlying_features_dfs[0]
        for df in non_underlying_features_dfs[1:]:
            all_non_underlying_features_df = pd.merge(all_non_underlying_features_df, df, on='date', how='inner')
        all_non_underlying_features_df.set_index('date', inplace=True)
        all_non_underlying_features_df = all_non_underlying_features_df[all_non_underlying_features_df.index >= pd.to_datetime(min_date)]
                
        msg = "Creating and merging features for each symbol..."
        logger.info(msg)
        print(msg)
        features_for_all_symbols = []
        # Iterate through each symbol to generate its specific underlying features (e.g., historical volatility).
        for idx, symbol in enumerate(distinct_symbols_in_database["act_symbol"]):
            try:
                dataset = utils.get_underyling_features(symbol, min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"), config.VOLATILITY_TIMEFRAMES, config.MOMENTUM_TIMEFRAMES)
                dataset["act_symbol"] = symbol
                dataset = dataset.reset_index() if 'date' not in dataset.columns else dataset
                dataset['date'] = pd.to_datetime(dataset['date']).dt.tz_localize(None)
                # Merge the underlying-specific features with the general market features.
                dataset = pd.merge(dataset, all_non_underlying_features_df, on='date', how='inner')
                features_for_all_symbols.append(dataset)
                progress = (idx + 1) / len(distinct_symbols_in_database) * 100
                logger.info(f"({progress:.2f}%) Created dataset for symbol '{symbol}'.")
            except Exception: 
                msg = f"Failed to create dataset for symbol '{symbol}'."
                logger.warning(msg, exc_info=True)
                print(msg)
                sys.exit()
        
        # Concatenate all feature dataframes into one and save to the database and optionally to a CSV.
        if features_for_all_symbols:
            features_df_all = pd.concat(features_for_all_symbols)
            logger.info(f"Saving {len(features_df_all)} rows of features to database table '{config.FEATURES_DATA_TABLE}'...")
            if config.SAVE_FEATURES_TO_CSV: 
                features_df_all.to_csv('data/features.csv', index=False)
                logger.info("Features also saved to 'data/features.csv'.")
            db.create_table_from_df(config.FEATURES_DATA_TABLE, features_df_all, if_exists='replace')
            msg = f"Features saved to database table '{config.FEATURES_DATA_TABLE}' successfully."
            logger.info(msg)
            print(msg)
else:
    # If feature downloading is disabled, log this and check if the feature table already exists.
    msg = "\n--- Feature Engineering ---"
    logger.info(msg)
    print(msg)
    if not db.check_if_table_exists(config.FEATURES_DATA_TABLE):
        msg = f"Feature table '{config.FEATURES_DATA_TABLE}' does not exist. Skipping feature download."
        logger.warning(msg)
        print(msg)
    else:
        msg = "Feature download skipped due to configuration settings (DOWNLOAD_FEATURES=False)."
        logger.info(msg)
        print(msg)


def load_and_split_data(db_connection: duckdb.DuckDBPyConnection, symbol: str, options_table: str, features_table: str) -> Tuple[List, List]:
    """
    Loads and partitions the data for a specific symbol into an initial training set
    and a subsequent monitoring set for adaptive validation.

    This function first identifies all dates for which both options and feature data
    are available to ensure data integrity. It then splits these dates chronologically
    into a training set (used for initial model training and tuning) and a monitoring
    set (used to simulate live performance and trigger recalibrations).

    Parameters
    ----------
    db_connection : duckdb.DuckDBPyConnection
        An active connection to the DuckDB database.
    symbol : str
        The ticker symbol for which to load data (e.g., 'SPY').
    options_table : str
        The name of the table containing options data.
    features_table : str
        The name of the table containing engineered features.

    Returns
    -------
    Tuple[List, List]
        A tuple containing two lists:
        - The first list is the initial training data.
        - The second list is the monitoring data.
        Each element in these lists is a tuple: (date_string, options_df, features_df).
    """
    msg = f"\n--- Loading and Splitting Data for Symbol: {symbol} ---"
    logger.info(msg)
    print(msg)
    
    # Query for dates that have data in both the features and options tables to avoid mismatches.
    query = f"""
    SELECT DISTINCT f.date::VARCHAR AS date FROM {features_table} f
    JOIN {options_table} o ON f.date = o.date AND f.act_symbol = o.act_symbol
    WHERE f.act_symbol = '{symbol}' ORDER BY f.date;
    """
    try:
        all_dates = db_connection.execute(query).df()['date'].tolist()
        if not all_dates:
            raise ValueError(f"No matching dates found for symbol '{symbol}'.")
        msg = f"Found {len(all_dates)} days with complete data from {all_dates[0]} to {all_dates[-1]}."
        logger.info(msg)
        print(msg)
    except Exception:
        msg = f"Failed to query dates for data loading for symbol '{symbol}'."
        logger.error(msg, exc_info=True)
        print(msg)

    # Fetch daily dividend yield data, as this is a crucial input for option pricing models.
    try:
        dividend_df = utils.get_daily_dividend_yield_cached(symbol, all_dates[0], all_dates[-1])
        dividend_df['date'] = pd.to_datetime(dividend_df['date']).dt.tz_localize(None)
        msg = f"Successfully fetched {len(dividend_df)} dividend yield rows for {symbol}."
        logger.info(msg)
        print(msg)
    except Exception as e:
        msg = f"Could not fetch dividend yield data for {symbol}. Reason: {e}"
        logger.warning(msg)
        print(msg)
        dividend_df = pd.DataFrame({'date': [], 'dividend_yield': []})

    # Fetch historical ex-dividend dates to calculate 'days until next dividend', a potentially useful feature.
    try:
        ticker = yf.Ticker(symbol)
        historical_dividends = ticker.dividends
        if not historical_dividends.empty:
            dividend_dates = pd.to_datetime(historical_dividends.index).tz_localize(None).sort_values()
            msg = f"Successfully fetched {len(dividend_dates)} dividend dates for {symbol}."
            logger.info(msg)
            print(msg)
        else:
            dividend_dates = pd.Series(dtype='datetime64[ns]')
            msg = f"No historical dividend data found for {symbol}."
            logger.info(msg)
            print(msg)
    except Exception as e:
        msg = f"Could not fetch dividend dates for {symbol}. Reason: {e}"
        logger.warning(msg)
        print(msg)
        dividend_dates = pd.Series(dtype='datetime64[ns]')

    # Split dates into an initial training set and a subsequent monitoring set for adaptive validation.
    # This chronological split mimics a real-world scenario where a model is trained on past data
    # and evaluated on future, unseen data.
    train_idx = int(len(all_dates) * config.TRAINING_SET_PERCENTAGE / 100)
    initial_train_dates, monitoring_dates = all_dates[:train_idx], all_dates[train_idx:]

    if not initial_train_dates or not monitoring_dates:
        raise ValueError("A data split is empty after date partitioning.")
    msg = f"Data split: {len(initial_train_dates)} days for initial training, {len(monitoring_dates)} days for monitoring."
    logger.info(msg)
    print(msg)

    def _load(dates: List[str]) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
        """
        Loads option and feature data for the given dates.

        Parameters
        ----------
        dates : List[str]
            A list of dates for which to load the data.

        Returns
        -------
        List[Tuple[str, pd.DataFrame, pd.DataFrame]]
            A list of tuples, where each tuple contains the date, the corresponding options data for that date,
            and the feature data for that date.
        """
        if not dates: return []
        # Construct a single query for all dates to minimize database round-trips.
        dates_str = "','".join(dates)
        options_query = f"SELECT * FROM {options_table} WHERE act_symbol = '{symbol}' AND date IN ('{dates_str}')"
        features_query = f"SELECT * FROM {features_table} WHERE act_symbol = '{symbol}' AND date IN ('{dates_str}')"
        options_df = db_connection.execute(options_query).df()
        features_df = db_connection.execute(features_query).df()
        
        options_df['date'] = pd.to_datetime(options_df['date']).dt.tz_localize(None)
        features_df['date'] = pd.to_datetime(features_df['date']).dt.tz_localize(None)
        
        # Merge dividend yield data, forward-filling to handle non-trading days and filling any remaining NaNs with 0.
        if not dividend_df.empty:
            features_df = pd.merge(features_df, dividend_df, on='date', how='left')
            features_df['dividend_yield'] = features_df['dividend_yield'].ffill().fillna(0)
        else: features_df['dividend_yield'] = 0.0
        
        # Create the Volatility Risk Premium (VRP) feature, which is the spread between implied (VIX) and realized volatility.
        features_df['volatility_risk_premium'] = features_df['vix_index'] - features_df[f'vol_{max(config.VOLATILITY_TIMEFRAMES)}d']
        
        # Calculate and add the 'days_until_next_dividend' feature.
        if not dividend_dates.empty:
            next_dividend_dates = [dividend_dates[dividend_dates > d].min() for d in features_df['date']]
            features_df['next_dividend_date'] = pd.to_datetime(next_dividend_dates)
            features_df['days_until_next_dividend'] = (features_df['next_dividend_date'] - features_df['date']).dt.days.fillna(0)
            features_df.drop(columns=['next_dividend_date'], inplace=True)
        else:
            features_df['days_until_next_dividend'] = 0.0

        # Group data by date and return a list of tuples, each containing the data for one day.
        options_by_date = dict(tuple(options_df.groupby('date')))
        features_by_date = dict(tuple(features_df.groupby('date')))
        return [(d, options_by_date[pd.to_datetime(d)], f) for d, f in features_by_date.items() if pd.to_datetime(d) in options_by_date and not f.empty]

    return _load(initial_train_dates), _load(monitoring_dates)


def prepare_feature_scaler(training_data: List) -> Tuple[StandardScaler, List[str]]:
    """
    Fits a StandardScaler on the initial training data.

    Standardizing features (scaling to zero mean and unit variance) is a crucial
    preprocessing step for neural networks. It helps the optimization algorithm
    converge faster and more reliably.

    Importantly, the scaler is fitted *only* on the training data to prevent data
    leakage from the validation or monitoring sets. The same fitted scaler is then
    used to transform all subsequent data.

    Parameters
    ----------
    training_data : List
        The initial training dataset, a list of (date, options_df, features_df) tuples.

    Returns
    -------
    Tuple[StandardScaler, List[str]]
        A tuple containing the fitted StandardScaler instance and the list of feature
        column names that were scaled.
    """
    msg = "\n--- Preparing Feature Scaler ---"
    logger.info(msg)
    print(msg)
    if not training_data: 
        raise ValueError("Cannot prepare scaler with no training data.")
    # Concatenate feature dataframes from all training days.
    features = pd.concat([f for _, _, f in training_data], ignore_index=True)
    # Identify feature columns to be scaled (i.e., all columns except identifiers).
    cols = [c for c in features.columns if c not in ['date', 'act_symbol']]
    if not cols: 
        raise ValueError("No feature columns found to fit the scaler.")
    scaler = StandardScaler().fit(features[cols].values)
    msg = f"Fitted StandardScaler on {len(features)} rows and {len(cols)} feature columns."
    logger.info(msg)
    print(msg)
    return scaler, cols


def prepare_option_helpers(eval_date_str, options_df, features_df, dividend):
    """
    Sets up the QuantLib environment and constructs option 'helpers' for a given day.

    This function acts as a bridge between the pandas/numpy data world and the
    QuantLib financial modeling world. It configures the global evaluation date,
    builds the yield and dividend curves, and wraps each option contract into a
    QuantLib `EuropeanOption` object along with its market price. These are
    collectively known as 'helpers' in QuantLib's calibration terminology.

    Parameters
    ----------
    eval_date_str : str
        The evaluation date in 'YYYY-MM-DD' format.
    options_df : pd.DataFrame
        DataFrame of options contracts for the evaluation date.
    features_df : pd.DataFrame
        DataFrame with features (including yield curve rates) for the evaluation date.
    dividend : float
        The continuous dividend yield for the evaluation date.

    Returns
    -------
    Tuple
        A tuple containing:
        - A list of option helpers (QL option object, market price, weight, strike).
        - A handle to the risk-free yield curve.
        - A handle to the dividend yield curve.
        - A handle to the underlying spot price.
    """
    py_date = pd.to_datetime(eval_date_str).date()
    eval_date = ql.Date.from_date(py_date)
    # Set the global evaluation date for all subsequent QuantLib calculations.
    ql.Settings.instance().evaluationDate = eval_date
    calendar, day_count = ql.UnitedStates(ql.UnitedStates.NYSE), ql.Actual365Fixed()
    
    # Map feature column names to QuantLib Period objects to build the yield curve.
    yield_curve_map = {name: ql.Period(val, unit) for name, val, unit in
                       [('month_1', 1, ql.Months), ('month_3', 3, ql.Months), ('month_6', 6, ql.Months),
                        ('year_1', 1, ql.Years), ('year_2', 2, ql.Years), ('year_3', 3, ql.Years),
                        ('year_5', 5, ql.Years), ('year_7', 7, ql.Years), ('year_10', 10, ql.Years),
                        ('year_20', 20, ql.Years), ('year_30', 30, ql.Years)]}
    row = features_df.iloc[0]
    tenors, rates = [eval_date], [row['month_1']] # Start with the shortest rate.
    for name, period in yield_curve_map.items():
        if name in row: rates.append(row[name]); tenors.append(calendar.advance(eval_date, period))
    
    # Create RelinkableTermStructureHandles. These are powerful QuantLib constructs
    # that allow the underlying curves or quotes to be changed without having to
    # rebuild the entire pricing engine, which is useful in dynamic models.
    rf_h = ql.RelinkableYieldTermStructureHandle(ql.ZeroCurve(tenors, rates, day_count, calendar, ql.Linear(), ql.Continuous))
    div_h = ql.RelinkableYieldTermStructureHandle(ql.FlatForward(eval_date, dividend, day_count, ql.Continuous))
    spot_h = ql.QuoteHandle(ql.SimpleQuote(row['underlying_price']))
    
    options_df['moneyness'] = np.where(
        options_df['call_put'] == 'Call',
        (row['underlying_price']) / options_df['strike'],
        options_df['strike'] / (row['underlying_price'])
    )
    # Filter options based on moneyness and valid bid-ask spreads.
    options_df = options_df[
    (options_df['moneyness'] >= config.MIN_MONEYNESS) & (options_df['moneyness'] <= config.MAX_MONEYNESS) &
    (options_df['bid'] > 0) &
    (options_df['ask'] > options_df['bid'])
    ].copy()
    
    helpers = []
    # Iterate through each option contract to create a QuantLib helper.
    for _, r in options_df.iterrows():
        # Filter out illiquid options (low vega) or options with invalid prices.
        # Vega weighting is used to give more importance to at-the-money options,
        # which are more sensitive to volatility changes.
        if r.get('vega', 0) > 1e-4 and r.get('bid', 0) > 0 and r.get('ask', 0) > r['bid']:
            exp = ql.Date.from_date(pd.to_datetime(r['expiration']).date())
            if exp > eval_date: # Ensure the option has not expired.
                strike = r['strike']
                payoff = ql.PlainVanillaPayoff(ql.Option.Call if r['call_put'].strip().upper() == 'CALL' else ql.Option.Put, strike)
                helpers.append((ql.EuropeanOption(payoff, ql.EuropeanExercise(exp)),
                                (r['bid'] + r['ask']) / 2, # Mid-market price
                                r['vega'] / (r['ask'] - r['bid']), # Vega-based weight
                                strike))
    return helpers, rf_h, div_h, spot_h


def preprocess_ql_helpers(daily_data: List) -> List:
    """
    Pre-calculates QuantLib helpers for all days in a dataset to avoid
    re-computation in every training epoch.

    Parameters
    ----------
    daily_data : List
        A list of tuples, where each is (date_string, options_df, features_df).

    Returns
    -------
    List
        A new list of tuples, where each is now in the format:
        (date_string, options_df, features_df, precalculated_ql_objects).
        The precalculated_ql_objects is the tuple (helpers, rf_h, div_h, spot_h).
    """
    msg = "\n--- Pre-processing QuantLib Helpers for All Days ---"
    logger.info(msg)
    print(msg)
    processed_list = []
    for date_str, options_df, features_df in tqdm(daily_data, desc="Preparing QuantLib Helpers"):
        dividend = features_df['dividend_yield'].iloc[0] if 'dividend_yield' in features_df.columns else 0.0
        ql_objects = prepare_option_helpers(date_str, options_df, features_df, dividend)
        processed_list.append((date_str, options_df, features_df, ql_objects))
        
    return processed_list


class ResidualParameterModel(tf.keras.Model):
    """
    A Keras model that predicts Heston parameters using a residual architecture.

    Instead of predicting the parameters directly, the model predicts a *delta* or
    *correction* to a pre-defined initial guess. This can stabilize training, as
    the network only needs to learn the adjustment from a reasonable baseline,
    rather than learning the full parameter space from scratch.

    The output layer uses scaled sigmoid and tanh activations to constrain the
    predicted parameters within their valid financial ranges (e.g., volatilities
    must be positive, correlation must be between -1 and 1).
    """
    def __init__(self, total_params, upper_bounds_sigmoid, neuron_counts: list, use_dropout, dropout_rate, **kwargs):
        """
        Initializes the model architecture.

        Parameters
        ----------
        total_params : int
            Total number of parameters to predict (5 for Heston).
        upper_bounds_sigmoid : tf.Tensor
            The upper bounds for the parameters constrained by sigmoid (v0, kappa, theta, sigma).
        neuron_counts : list[int]
            A list specifying the number of neurons in each hidden layer.
        use_dropout : bool
            Whether to include Dropout layers for regularization.
        dropout_rate : float
            The dropout rate to apply if use_dropout is True.
        """
        super().__init__(**kwargs)
        self.upper_bounds_sigmoid = tf.constant(upper_bounds_sigmoid, dtype=self.dtype)
        self.hidden_layers = [tf.keras.layers.Dense(n, activation='relu') for n in neuron_counts]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout_rate) for _ in range(len(neuron_counts))] if use_dropout else []
        # The output layer is initialized with zeros, so the initial prediction is just the
        # initial guess, reinforcing the residual learning concept.
        self.output_layer = tf.keras.layers.Dense(total_params, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs, training=False):
        """
        Forward pass through the model.

        Parameters
        ----------
        inputs : tuple
            A tuple containing (features, initial_logits).
        training : bool
            Flag indicating whether the model is in training mode, which activates dropout.

        Returns
        -------
        tf.Tensor
            The final predicted Heston parameters.
        """
        features, initial_logits = inputs
        x = features
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.dropout_layers and training: x = self.dropout_layers[i](x)
        
        # The core of the residual connection: add the network's output to the initial guess logits.
        final_logits = initial_logits + self.output_layer(x)
        
        # Apply activation functions to transform logits into valid parameter ranges.
        # Sigmoid for positive-bounded parameters (v0, kappa, theta, sigma).
        params_sig = self.upper_bounds_sigmoid * tf.sigmoid(final_logits[:, :4])
        # Tanh for the correlation parameter (rho), which is bounded between -1 and 1.
        param_tanh = tf.tanh(final_logits[:, 4:])
        return tf.concat([params_sig, param_tanh], axis=1)

def _calculate_loss_for_day_heston(params, helpers, rf_h, div_h, spot_h, eval_date):
    """
    Calculates the mean squared error between the model prices and the market prices for a given evaluation date.

    Parameters
    ----------
    params : list
        The Heston parameters [v0, kappa, theta, sigma, rho].
    helpers : list
        List of QuantLib option helpers for the day.
    rf_h, div_h, spot_h : QuantLib Handles
        Handles for risk-free rate, dividend yield, and spot price.
    eval_date : ql.Date
        The evaluation date.

    Returns
    -------
    float
        The calculated weighted MSE. Returns a large penalty value (1e6) if pricing fails.
    """
    try:
        ql.Settings.instance().evaluationDate = eval_date
        # Set up the Heston model and pricing engine with the given parameters.
        engine = ql.AnalyticHestonEngine(ql.HestonModel(ql.HestonProcess(rf_h, div_h, spot_h, *params)))
        # Calculate the squared error for each option, weighted by its vega and the bid-ask spread.
        errors = [(o.NPV() - mkt_p) ** 2 * w for o, mkt_p, w, _ in helpers if o.setPricingEngine(engine) or True]
        return np.mean(errors) if errors else 1e6
    except Exception:
        # Catch any QuantLib errors during pricing (e.g., from invalid parameters)
        # and return a large loss to penalize this parameter set.
        return 1e6

def _py_loss_and_grad_wrapper(params_tensor, helpers, rf_h, div_h, spot_h, settings, eval_date_str):
    """
    Wrapper function for the loss and gradient calculation.

    Parameters
    ----------
    params_tensor : tf.Tensor
        The Heston model parameters as a tensor.
    helpers : List[tuple]
        A list of tuples containing the option helpers, the midpoint of the bid and ask prices, the vega of the option, and the strike price.
    rf_h : float
        The risk-free rate.
    div_h : float
        The dividend yield.
    spot_h : float
        The spot volatility.
    settings : Dict[str, Any]
        A dictionary containing the model settings.
    eval_date_str : str
        The evaluation date as a string.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the loss and the gradient, both as NumPy arrays.
    """
    params = params_tensor.numpy()
    py_date = pd.to_datetime(eval_date_str).date()
    eval_date = ql.Date.from_date(py_date)    
    gradient_method = settings.get("gradient_method", "central").lower()
    h = settings['h_relative']
    loss = _calculate_loss_for_day_heston(params, helpers, rf_h, div_h, spot_h, eval_date)

    def grad_fn(i):        
        """
        Calculates the gradient of the loss with respect to the i-th parameter using the selected gradient method ("forward" or "central").

        Parameters
        ----------
        i : int
            The index of the parameter for which the gradient is calculated.

        Returns
        -------
        float
            The gradient of the loss with respect to the i-th parameter.
        """
        if gradient_method == "forward":
            # Forward difference formula: (f(x+h) - f(x)) / h. Less accurate but faster.
            p_plus = params.copy(); p_plus[i] += h
            loss_plus = _calculate_loss_for_day_heston(p_plus, helpers, rf_h, div_h, spot_h, eval_date)
            return (loss_plus - loss) / h
        else:
            # Central difference formula: (f(x+h) - f(x-h)) / (2*h). More accurate but requires
            # two additional loss calculations per parameter.
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h; p_minus[i] -= h
            loss_plus = _calculate_loss_for_day_heston(p_plus, helpers, rf_h, div_h, spot_h, eval_date)
            loss_minus = _calculate_loss_for_day_heston(p_minus, helpers, rf_h, div_h, spot_h, eval_date)
            return (loss_plus - loss_minus) / (2 * h)

    # Parallelize the gradient calculation across parameters to speed up training.
    with ThreadPoolExecutor(max_workers=settings.get("num_threads", 1)) as executor:
        grad = np.array(list(executor.map(grad_fn, range(len(params)))))
        
    return np.array(loss), grad

def calculate_implied_volatility(option, target_price, process):
    """
    Calculates the implied volatility of a given option using the Black-Scholes model.
    
    Parameters
    ----------
    option : ql.Option
        The QuantLib option object.
    target_price : float
        The target price for which to calculate the implied volatility.
    process : ql.BlackScholesProcess
        The process (risk-free rate, dividend yield, spot volatility) to use for the calculation.
    
    Returns
    -------
    float or None
        The calculated implied volatility, or None if the calculation fails.
    """
    try: 
        # QuantLib's built-in solver for implied volatility.
        return option.impliedVolatility(target_price, process, 1.0e-4, 100, 1e-7, 4.0)
    except Exception: 
        # The calculation can fail if the price is outside the no-arbitrage bounds.
        msg = f"Could not calculate implied volatility for option with target_price {target_price}."
        logger.debug(msg, exc_info=True)
        # print(msg)
        return None

def perform_recalibration_tests(daily_metrics_df: pd.DataFrame, settings: Dict) -> bool:
    """
    Checks if the model needs to be recalibrated by performing a change point detection on the mean absolute error (MAE) and root mean squared error (RMSE) of the model.

    Parameters
    ----------
    daily_metrics_df : pd.DataFrame
        Dataframe containing the daily performance metrics of the model.
    settings : Dict[str, Any]
        Dictionary containing the settings for the model.

    Returns
    -------
    bool
        True if the model needs to be recalibrated, False otherwise.

    Notes
    -----
    The check is performed by fitting a Piecewise Regression (Pelt) model to the errors and checking
    if the last segment has a mean error that is greater than the previous segment by a certain factor
    (default is 1.20). If this condition is met, the function returns True, indicating that the model
    needs to be recalibrated. If not enough data is available, the function returns False.
    """
    recalibration_needed = False
    
    # Do not perform the check if there isn't enough data to be statistically meaningful.
    min_days_for_check = settings.get("MIN_FOLD_DAYS", 20)
    if daily_metrics_df.empty or len(daily_metrics_df) < min_days_for_check:
        return False

    msg = f"Recalibration check on {len(daily_metrics_df)} days of performance data."
    logger.info(msg)
    print(msg)
    
    for metric in ['mae', 'rmse']:
        errors = daily_metrics_df[metric].values
        try:
            # Fit the change-point detection algorithm.
            algo = rpt.Pelt(model="rbf").fit(errors)
            penalty = 2 * np.log(len(errors)) # Common penalty value (BIC)
            change_points = algo.predict(pen=penalty)
            
            # If at least one change point is detected...
            if len(change_points) > 1:
                last_segment_start = change_points[-2]
                last_segment_mean = np.mean(errors[last_segment_start:])
                prev_segment_mean = np.mean(errors[:last_segment_start])
                
                # Check if the error in the most recent segment has increased beyond a threshold.
                threshold_factor = settings.get('RECALIBRATION_THRESHOLD_FACTOR', 1.20)
                if last_segment_mean > prev_segment_mean * threshold_factor:
                    msg = f"TRIGGER: Change point in {metric.upper()}. New error mean ({last_segment_mean:.4f}) is > {int((threshold_factor-1)*100)}% of previous mean ({prev_segment_mean:.4f})."
                    logger.warning(msg)
                    print(msg)
                    recalibration_needed = True
                    break # Trigger recalibration and stop checking other metrics.
        except Exception as e:
            logger.error(f"Could not perform change point detection for {metric.upper()}: {e}")
    return recalibration_needed

def evaluate_single_day(model, scaler, cols, logits, day_data) -> Tuple[Dict, List[Dict]]:
    """
    Performs a comprehensive evaluation of the model's performance on a single day's data.

    It predicts Heston parameters, prices all options, calculates aggregate error
    metrics (MAE, RMSE), and collects detailed option-level results (like implied
    volatilities) for deeper analysis and plotting.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Heston parameter prediction model.
    scaler : StandardScaler
        The fitted feature scaler.
    cols : List[str]
        The list of feature column names.
    logits : tf.Tensor
        The baseline logits for the residual model.
    day_data : tuple
        A tuple of (date_string, options_df, features_df).

    Returns
    -------
    Tuple[Dict, List[Dict]]
        - A dictionary with daily summary metrics ('date', 'mae', 'rmse').
        - A list of dictionaries, each containing detailed results for a single option.
    """
    date_str, options, features, ql_objects = day_data
    # Scale features and predict Heston parameters for the day.
    ftrs = scaler.transform(features[cols].values)
    params = model((ftrs, logits), training=False).numpy()[0]
    dividend = features['dividend_yield'].iloc[0] if 'dividend_yield' in features.columns else 0.0
    helpers, rf_h, div_h, spot_h = ql_objects

    if not helpers:
        return {'date': date_str, 'mae': np.nan, 'rmse': np.nan}, []

    py_date = pd.to_datetime(date_str).date()
    eval_date = ql.Date.from_date(py_date)
    ql.Settings.instance().evaluationDate = eval_date
    day_count = ql.Actual365Fixed()

    # Set up the Heston pricing engine with the predicted parameters.
    engine = ql.AnalyticHestonEngine(ql.HestonModel(ql.HestonProcess(rf_h, div_h, spot_h, *params)))
    # Set up a generic Black-Scholes process needed for implied volatility calculations.
    bsm_p = ql.BlackScholesMertonProcess(spot_h, div_h, rf_h, ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, ql.UnitedStates(ql.UnitedStates.NYSE), 0.2, day_count)))

    option_level_results = []
    market_prices, model_prices = [], []

    for option, mkt_price, _, strike in helpers:
        option.setPricingEngine(engine)
        model_price = option.NPV()

        market_prices.append(mkt_price)
        model_prices.append(model_price)

        # Calculate metrics for detailed analysis.
        maturity_date = option.exercise().lastDate()
        time_to_maturity = day_count.yearFraction(eval_date, maturity_date)

        market_iv = calculate_implied_volatility(option, mkt_price, bsm_p)
        model_iv = calculate_implied_volatility(option, model_price, bsm_p)

        if market_iv is not None and model_iv is not None:
            option_level_results.append({
                'date': date_str,
                'market_price': mkt_price,
                'model_price': model_price,
                'moneyness': strike / spot_h.value(),
                'time_to_maturity': time_to_maturity,
                'market_iv': market_iv,
                'model_iv': model_iv
            })

    if not model_prices:
        return {'date': date_str, 'mae': np.nan, 'rmse': np.nan}, []

    # Calculate aggregate daily error metrics.
    mae = np.mean(np.abs(np.array(market_prices) - np.array(model_prices)))
    rmse = np.sqrt(np.mean((np.array(market_prices) - np.array(model_prices)) ** 2))

    daily_metrics = {'date': date_str, 'mae': mae, 'rmse': rmse}
    return daily_metrics, option_level_results

def _evaluate_model_on_data(model, data, scaler, cols, logits):
    """
    Evaluates the model on the given data.

    Parameters
    ----------
    model : keras.Model
        The model to evaluate.
    data : List[Tuple[str, pd.DataFrame, pd.DataFrame]]
        A list of tuples containing the date string, the options data, and the features data for each day.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to scale the features.
    cols : List[str]
        The columns of the features to use.
    logits : List[int]
        The number of days to consider for each option.

    Returns
    -------
    float
        The mean loss of the model on the given data. If no data is available, returns float('inf').
    """
    losses = []
    for date, options, features, ql_objects in data:
        helpers, rf_h, div_h, spot_h = ql_objects
        if helpers:
            params = model((scaler.transform(features[cols].values), logits), training=False).numpy()[0]
            py_date = pd.to_datetime(date).date()
            eval_date = ql.Date.from_date(py_date)
            losses.append(_calculate_loss_for_day_heston(params, helpers, rf_h, div_h, spot_h, eval_date))
    return np.mean(losses) if losses else float('inf')

def train_model_for_fold(model, best_hps, training_data, validation_data, scaler, cols, logits, settings, fold_num):
    """
    Trains the neural network for a single fold using a custom training loop,
    augmented with Keras callbacks for adaptive learning rate scheduling and
    robust early stopping.

    This function implements:
    - A custom gradient calculation via tf.py_function.
    - Stochastic training by subsampling options for each day to accelerate epochs.
    - `ReduceLROnPlateau`: Automatically reduces the learning rate when validation loss stagnates.
    - `EarlyStopping`: Stops training when no improvement is seen and automatically
      restores the weights from the best performing epoch.

    Parameters
    ----------
    model : tf.keras.Model
        The model instance to be trained.
    best_hps : kt.HyperParameters
        The set of best hyperparameters found by the tuner.
    training_data, validation_data : List
        The pre-processed datasets for training and validation.
    scaler, cols, logits, settings :
        Standard tools and settings for training.
    fold_num : int
        The identifier for the current training fold (0 for initial, >0 for recalibrations).

    Returns
    -------
    tf.keras.Model
        The trained model with the best weights automatically restored.
    """
    msg = f"--- Training model for Fold {fold_num} | Training Days: {len(training_data)} | Validation Days: {len(validation_data)} ---"
    logger.info(msg)
    print(msg)
        
    # --- 1. Model and Optimizer Setup ---
    # Build the model from hyperparameters for the very first training run
    if fold_num == 0:
        model = hypermodel.build(best_hps)
    
    # Set up the optimizer with the optimal *initial* learning rate from the tuner
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_hps.get('lr'))
    
    # We must compile the model to connect the optimizer, which allows callbacks
    # to access and modify the learning rate. This does not interfere with our
    # custom gradient logic.
    model.compile(optimizer=optimizer)

    # --- 2. Callback Definition ---
    # This callback reduces the learning rate when validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=settings.get("learning_rate_scheduler_factor", 0.2),   # new_lr = lr * factor
        patience=settings.get("learning_rate_scheduler_patience", 3), # Epochs to wait for improvement before reducing LR
        verbose=settings.get("learning_rate_verbose", 1),             # Print a message when the LR is updated
        min_lr=settings.get("learning_rate_lower_bound", 1e-7)        # Lower bound on the learning rate
    )

    # This callback stops training early and automatically restores the best weights
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=settings.get("early_stopping_patience", 10),
        verbose=settings.get("early_stopping_verbose", 1),
        restore_best_weights=True # Automatically saves/restores the best model state
    )
    
    lr_scheduler.set_model(model)
    early_stopper.set_model(model)
    
    lr_scheduler.on_train_begin()
    early_stopper.on_train_begin()

    # --- 3. Main Training Loop ---
    subsample_frac = settings.get("OPTION_SUBSAMPLE_PERCENTAGE", 100) / 100.0
    max_epochs = settings['num_epochs_final_training']

    for epoch in range(max_epochs):
        print(f"\n--- Fold {fold_num}, Epoch {epoch + 1}/{max_epochs} ---")
        
        # Log the current learning rate at the start of each epoch
        current_lr = float(model.optimizer.learning_rate.numpy())
        print(f"Current Learning Rate: {current_lr:.8f}")

        # Loop through each day in the training data with a progress bar
        for day_idx, (date, options, features, ql_objects) in enumerate(tqdm(training_data, desc=f'Progress within epoch {epoch + 1} of fold {fold_num}')):
            helpers, rf_h, div_h, spot_h = ql_objects
            if not helpers: continue

            # --- Option Subsampling Logic ---
            sampled_helpers = helpers
            if subsample_frac < 1.0:
                # Create a deterministic seed for reproducibility
                day_epoch_seed = config.GLOBAL_SEED + epoch * 100000 + day_idx
                rng = random.Random(day_epoch_seed)
                num_to_sample = int(len(helpers) * subsample_frac)
                if num_to_sample > 0:
                    sampled_helpers = rng.sample(helpers, num_to_sample)
            
            if not sampled_helpers: continue

            # --- Custom Training Step ---
            with tf.GradientTape() as tape:
                params = model((scaler.transform(features[cols].values), logits), training=True)
                
                @tf.custom_gradient
                def loss_op(p):
                    loss, grad = tf.py_function(lambda t: _py_loss_and_grad_wrapper(t, sampled_helpers, rf_h, div_h, spot_h, settings, date), [p], [tf.float64]*2)
                    def grad_fn(dy): 
                        """
                        Gradient function for the loss calculation.

                        Parameters
                        ----------
                        dy : tf.float64
                            The upstream gradient.

                        Returns
                        -------
                        tf.float64
                            The downstream gradient.
                        """
                        # The chain rule: upstream_gradient * local_gradient
                        return dy * tf.reshape(grad, tf.shape(p))
                    return loss, grad_fn
                
                loss = loss_op(params[0])
            
            # Apply the computed gradients to the model's trainable variables
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

        # --- 4. End-of-Epoch Evaluation and Callback Handling ---
        val_loss = _evaluate_model_on_data(model, validation_data, scaler, cols, logits)
        msg = f"Validation loss after Epoch {epoch + 1} of Fold {fold_num}: {val_loss:.6f}"
        logger.info(msg)
        print(msg)
        
        # Manually trigger the callbacks, passing them the latest validation loss
        logs = {'val_loss': val_loss}
        lr_scheduler.on_epoch_end(epoch, logs)
        early_stopper.on_epoch_end(epoch, logs)

        # Check the flag set by the EarlyStopping callback
        if model.stop_training:
            msg = f"  -> Early stopping triggered for Fold {fold_num} after {epoch + 1} epochs."
            logger.info(msg)
            print(msg)
            break
            
    early_stopper.on_train_end()
    lr_scheduler.on_train_end()
            
    print(f"\nFinished training for Fold {fold_num}. The model has been restored to its best state.")   
    return model


def run_adaptive_window_evaluation(model, scaler, cols, logits, settings, best_hps, initial_training_data: List,
                                   monitoring_data: List,
                                   run_output_dir: str):
    """
    Runs an adaptive window evaluation on the given model.

    The evaluation is an iterative process, where the model is re-trained on the most recent data points and evaluated
    on the next set of days. The process is repeated until the maximum number of days per fold is reached or the model's
    performance degrades.

    The function returns the model with the best parameters found during the evaluation process.

    Parameters
    ----------
    model : keras.Model
        The model to evaluate.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to scale the features.
    cols : List[str]
        The columns of the features to use.
    logits : List[int]
        The number of days to consider for each option.
    settings : Dict[str, Any]
        Dictionary containing the settings for the model.
    best_hps : Dict[str, Any]
        Dictionary containing the best hyperparameters found so far.
    initial_training_data : List
        The initial data points to use for training.
    monitoring_data : List
        The data points to use for evaluation.
    run_output_dir : str
        The directory where to save the evaluation results.

    Returns
    -------
    keras.Model
        The model with the best parameters found during the evaluation process.
    """
    msg = "\n--- Starting Adaptive Window Evaluation ---"
    logger.info(msg)
    print(msg)

    current_training_data = list(initial_training_data)
    all_fold_summaries, all_daily_metrics, all_option_results, all_model_parameters = [], [], [], []
    surface_plot_data = None
    monitoring_idx = 0

    # The main loop that steps through the monitoring data.
    while monitoring_idx < len(monitoring_data):
        fold_num = len(all_fold_summaries) + 1

        # Recalibrate the model at the beginning of each new fold (except the first).
        if fold_num > 1:
            val_split_idx = int(len(current_training_data) * 0.8)
            train_split, val_split = current_training_data[:val_split_idx], current_training_data[val_split_idx:]
            msg = f"Splitting data for fold {fold_num} retraining: {len(train_split)} days for training, {len(val_split)} for validation."
            logger.info(msg)
            print(msg)
            # Fine-tune the existing model on the updated training data.
            model = train_model_for_fold(model, best_hps, train_split, val_split, scaler, cols, logits, settings, fold_num)
            model_save_path = os.path.join(run_output_dir, f"fold_{fold_num - 1}_model.weights.h5")
            model.save_weights(model_save_path)
            msg = f"Retrained model for Fold {fold_num - 1} saved to {model_save_path}."
            logger.info(msg)
            print(msg)

        # Record the model parameters used at the start of the fold.
        last_day_features = scaler.transform(current_training_data[-1][2][cols].values)
        params = model((last_day_features, logits), training=False).numpy()[0]
        param_names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
        param_dict = {name: val for name, val in zip(param_names, params)}
        param_dict['fold_num'] = fold_num
        all_model_parameters.append(param_dict)

        msg = f"Monitoring model performance for fold {fold_num}..."
        logger.info(msg)
        print(msg)
        current_fold_metrics = []
        fold_start_date = monitoring_data[monitoring_idx][0]
        reason = "End of data"

        # Inner loop to evaluate day-by-day until a recalibration is needed.
        while monitoring_idx < len(monitoring_data):
            day_data = monitoring_data[monitoring_idx]
            daily_metrics, option_results = evaluate_single_day(model, scaler, cols, logits, day_data)

            if not np.isnan(daily_metrics['mae']):
                current_fold_metrics.append(daily_metrics)
                all_option_results.extend(option_results)

            # Capture data for a 3D surface plot on the first day of evaluation.
            if fold_num == 1 and not surface_plot_data:
                surface_plot_data = (pd.DataFrame(option_results), day_data[0])

            monitoring_idx += 1
            metrics_df = pd.DataFrame(current_fold_metrics)
            recalibration_needed = perform_recalibration_tests(metrics_df, settings)
            max_days_reached = len(metrics_df) >= settings.get('MAX_FOLD_DAYS', 100)

            if recalibration_needed or max_days_reached:
                reason = "Performance Degradation" if recalibration_needed else "Max Fold Length Reached"
                msg = f"Fold {fold_num} ended. Reason: {reason}."
                logger.info(msg)
                print(msg)
                break

        fold_end_date = monitoring_data[monitoring_idx - 1][0]
        final_metrics_df = pd.DataFrame(current_fold_metrics)
        all_daily_metrics.extend(current_fold_metrics)

        # Summarize the performance of the completed fold.
        if not final_metrics_df.empty:
            summary = {'fold_num': fold_num, 'start_date': fold_start_date, 'end_date': fold_end_date,
                       'duration_days': len(final_metrics_df), 'avg_mae': final_metrics_df['mae'].mean(),
                       'avg_rmse': final_metrics_df['rmse'].mean(), 'end_reason': reason}
            all_fold_summaries.append(summary)
            msg = f"Fold {fold_num} Summary: Duration={summary['duration_days']} days, Avg MAE={summary['avg_mae']:.4f}, Avg RMSE={summary['avg_rmse']:.4f}"
            logger.info(msg)
            print(msg)

        # Expand the training data with the data from the fold that just completed.
        current_training_data.extend(monitoring_data[monitoring_idx - len(final_metrics_df): monitoring_idx])

    # --- Save all collected data and generate final plots ---
    msg = "\n--- Finalizing Run: Saving Data and Generating Plots ---"
    logger.info(msg)
    print(msg)
    summary_df = pd.DataFrame(all_fold_summaries)
    daily_metrics_df = pd.DataFrame(all_daily_metrics)
    option_results_df = pd.DataFrame(all_option_results)
    parameter_df = pd.DataFrame(all_model_parameters)

    summary_df.to_csv(os.path.join(run_output_dir, "evaluation_summary.csv"), index=False)
    daily_metrics_df.to_csv(os.path.join(run_output_dir, "daily_metrics.csv"), index=False)
    option_results_df.to_csv(os.path.join(run_output_dir, "option_level_results.csv"), index=False)
    parameter_df.to_csv(os.path.join(run_output_dir, "model_parameters.csv"), index=False)
    msg = f"Saved all result dataframes to {run_output_dir}"
    logger.info(msg)
    print(msg)

    plotting.plot_daily_performance(summary_df, daily_metrics_df, run_output_dir)
    plotting.plot_fold_summary(summary_df, run_output_dir)
    plotting.plot_parameter_drift(parameter_df, run_output_dir)
    
    if not option_results_df.empty:
        logger.info("Pre-processing option-level results for plotting...")
        # Bin options by moneyness and maturity for more granular performance analysis.
        moneyness_bins = [0, 0.9, 1.0, 1.1, np.inf]
        moneyness_labels = ['Deep OTM (<0.9)', 'OTM (0.9-1.0)', 'ATM (1.0-1.1)', 'ITM (>1.1)']
        maturity_bins = [0, 0.25, 0.5, 1, np.inf]
        maturity_labels = ['<3 Months', '3-6 Months', '6-12 Months', '>12 Months']
        option_results_df['moneyness_bin'] = pd.cut(option_results_df['moneyness'], bins=moneyness_bins, labels=moneyness_labels, right=False)
        option_results_df['maturity_bin'] = pd.cut(option_results_df['time_to_maturity'], bins=maturity_bins, labels=maturity_labels, right=False)
        msg = "Binning for 'moneyness' and 'time_to_maturity' complete."
        logger.info(msg)
        print(msg)

        plotting.plot_binned_rmse_heatmap(option_results_df, run_output_dir)
        plotting.plot_iv_rmse_heatmap(option_results_df, run_output_dir)
        plotting.plot_mape_heatmap(option_results_df, run_output_dir)
        
        if surface_plot_data and not surface_plot_data[0].empty:
            date_for_filename = pd.to_datetime(surface_plot_data[1]).strftime('%Y-%m-%d')
            plotting.plot_3d_vol_surface(surface_plot_data[0], run_output_dir, date_for_filename)

class HestonHyperModel(kt.HyperModel):
    """
    Defines the search space for the neural network architecture for Keras Tuner.
    
    This class tells the tuner which hyperparameters to optimize (e.g., number of layers,
    neurons per layer, dropout usage) and the range of values to try for each.
    """
    def __init__(self, total_params, upper_bounds):
        """
        Initializes the HestonHyperModel instance.

        Parameters
        ----------
        total_params : int
            Total number of parameters to predict (5 for Heston).
        upper_bounds : tf.Tensor
            The upper bounds for the parameters constrained by sigmoid (v0, kappa, theta, sigma).
        """
        self.total_params = total_params
        self.upper_bounds = upper_bounds

    def build(self, hp):
        """
        Builds and returns the Keras model with a specific set of hyperparameters.
        This method is called by the Keras Tuner for each trial.

        Parameters
        ----------
        hp : kt.HyperParameters
            The hyperparameters for the current trial, provided by the tuner.

        Returns
        -------
        ResidualParameterModel
            An instance of the model ready to be trained.
        """
        # Define a tunable number of layers.
        num_layers = hp.Int('num_layers', min_value=1, max_value=4)
        
        # Define a list of neuron counts, one for each layer.
        # This allows the tuner to find optimal widths for each layer individually.
        neuron_counts = [
            hp.Int(f'neurons_layer_{i}', min_value=16, max_value=128, step=16)
            for i in range(num_layers)
        ]

        return ResidualParameterModel(
            total_params=self.total_params,
            upper_bounds_sigmoid=self.upper_bounds,
            neuron_counts=neuron_counts,
            use_dropout=hp.Boolean('use_dropout'),
            dropout_rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        )
        
class HestonTuner(kt.Hyperband):
    """
    Custom Keras Tuner class that inherits from Hyperband to manage the
    hyperparameter search process. It overrides the run_trial method to
    implement the custom training loop required by the Heston model, which
    cannot use a standard model.fit() call.
    """
    def save_model(self, trial_id, model, step=0):
        """
        Saves the model weights for a given trial.

        Parameters
        ----------
        trial_id : str
            The ID of the trial to save the model for.
        model : keras.Model
            The model to save the weights for.
        step : int, optional
            The step number to include in the saved model filename. Defaults to 0.
        """
        model.save_weights(os.path.join(self.get_trial_dir(trial_id), f"w_{step}.weights.h5"))

    def load_model(self, trial):
        """
        Loads a model from a given trial.

        Parameters
        ----------
        trial : kt.Trial
            The trial to load the model from.

        Returns
        -------
        keras.Model
            The loaded model. If no weights are found, returns an untrained model.
        """
        model = self.hypermodel.build(trial.hyperparameters)
        weights_path = os.path.join(self.get_trial_dir(trial.trial_id), f"w_{trial.best_step}.weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        return model

    def run_trial(self, trial, train_data, val_data, scaler, cols, logits, settings):
        """
        Executes a single trial of the hyperparameter search.

        Parameters
        ----------
        trial : kt.Trial
            The trial object containing the hyperparameters to test in this iteration.
        train_data : tuple
            A tuple containing (dates, options, features) for the training data.
        val_data : tuple
            A tuple containing (dates, options, features) for the validation data.
        scaler : sklearn.preprocessing.StandardScaler
            The scaler object used to normalize the features.
        cols : list
            A list of column names to select from the features DataFrame.
        logits : tf.Tensor
            The tensor containing the initial guess of the Heston parameters.
        settings : dict
            A dictionary containing the configuration settings for the Heston model.

        Returns
        -------
        None
        """
        hp = trial.hyperparameters
        start_epoch = trial.get_state().get("epochs", 0)
        
        # Build a new model or load the existing one for this trial.
        model = self.load_model(trial) if start_epoch > 0 else self.hypermodel.build(hp)
        optimizer = tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log'))
        
        subsample_frac = settings.get("OPTION_SUBSAMPLE_PERCENTAGE", 100) / 100.0
        
        for epoch in range(start_epoch, hp.get("tuner/epochs")):
            # --- Training Step ---
            for day_idx, (date, options, features, ql_objects) in enumerate(tqdm(train_data, desc=f"Epoch {epoch+1} Trial {trial.trial_id}", leave=False)):
                helpers, rf_h, div_h, spot_h = prepare_option_helpers(date, options, features, features['dividend_yield'].iloc[0])
                if not helpers: continue
                
                sampled_helpers = helpers
                if subsample_frac < 1.0:
                    day_epoch_seed = config.GLOBAL_SEED + epoch * 100000 + day_idx
                    rng = random.Random(day_epoch_seed)
                    num_to_sample = int(len(helpers) * subsample_frac)
                    sampled_helpers = rng.sample(helpers, num_to_sample)
                
                with tf.GradientTape() as tape:
                    params = model((scaler.transform(features[cols].values), logits), training=True)
                    @tf.custom_gradient
                    def loss_op(p):
                        loss, grad = tf.py_function(lambda t: _py_loss_and_grad_wrapper(t, sampled_helpers, rf_h, div_h, spot_h, settings, date), [p], [tf.float64]*2)
                        def grad_fn(dy): return dy * tf.reshape(grad, tf.shape(p))
                        return loss, grad_fn
                    loss = loss_op(params[0])
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # --- Validation Step ---
            val_loss = self.evaluate_trial(model, val_data, scaler, cols, logits, settings)
            self.oracle.update_trial(trial.trial_id, {'val_loss': val_loss}, step=epoch + 1)
            trial.get_state()["epochs"] = epoch + 1
            
        self.save_model(trial.trial_id, model, step=hp.get("tuner/epochs"))

    def evaluate_trial(self, model, val_data, scaler, cols, logits, settings):
        """
        Evaluates the performance of the model on the validation set.

        Parameters
        ----------
        model : tf.keras.Model
            The model instance to evaluate.
        val_data : tuple
            A tuple containing (dates, options, features) for the validation data.
        scaler : sklearn.preprocessing.StandardScaler
            The scaler object used to normalize the features.
        cols : list
            A list of column names to select from the features DataFrame.
        logits : tf.Tensor
            The tensor containing the initial guess of the Heston parameters.
        settings : dict
            A dictionary containing the configuration settings for the Heston model.

        Returns
        -------
        float
            The mean loss of the model on the validation set. If no data is available, returns float('inf').
        """
        losses = []
        for date, options, features in val_data:
            helpers, rf_h, div_h, spot_h = prepare_option_helpers(date, options, features, features['dividend_yield'].iloc[0])
            if helpers:
                params = model((scaler.transform(features[cols].values), logits), training=False).numpy()[0]
                py_date = pd.to_datetime(date).date()
                eval_date = ql.Date.from_date(py_date)
                losses.append(_calculate_loss_for_day_heston(params, helpers, rf_h, div_h, spot_h, eval_date))
        return np.mean(losses) if losses else float('inf')


if __name__ == '__main__':
    try:
        SETTINGS = config.SETTINGS
        set_global_seed(config.GLOBAL_SEED)
        
        # Create a unique directory for this specific run to store all outputs.
        RUN_ID = f"{config.SYMBOL}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        RUN_OUTPUT_DIR = os.path.join(config.OUTPUT_BASE_DIR, RUN_ID)
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
        msg = f"Created output directory for this run: {RUN_OUTPUT_DIR}"
        logger.info(msg)
        print(msg)

        initial_train_data, monitoring_data = load_and_split_data(db.connection, config.SYMBOL, config.OPTIONS_DATA_TABLE, config.FEATURES_DATA_TABLE)
        if not initial_train_data: 
            msg = "No initial training data loaded. Aborting execution."
            logger.critical(msg)
            print(msg)
            sys.exit()
            
        initial_train_data_processed = preprocess_ql_helpers(initial_train_data)
        monitoring_data_processed = preprocess_ql_helpers(monitoring_data)

        scaler, cols = prepare_feature_scaler(initial_train_data)
        
        # Convert the initial guess for Heston parameters into the "logit" space.
        # This is the inverse of the activation functions (sigmoid, tanh).
        # The neural network's output is added to these logits, making it a residual model.
        guess = np.array(SETTINGS['initial_guess'])
        bounds = np.array(SETTINGS['upper_bounds_sigmoid'])
        logits = tf.constant([np.concatenate([np.log(np.clip(guess[:4]/bounds,1e-9,1-1e-9)/(1-np.clip(guess[:4]/bounds,1e-9,1-1e-9))), [np.arctanh(np.clip(guess[4],-0.99,0.99))]])])

        hypermodel = HestonHyperModel(5, SETTINGS['upper_bounds_sigmoid'])

        if config.USE_MODEL:
            # If a pre-trained model path is specified, load it and skip tuning/initial training.
            msg = f"\n--- Resuming evaluation with pre-trained model: {config.USE_MODEL} ---"
            logger.info(msg)
            print(msg)
            weights_path = config.USE_MODEL
            hp_path = os.path.join(os.path.dirname(weights_path), "best_hyperparameters.json")
            
            if not os.path.exists(weights_path) or not os.path.exists(hp_path):
                msg = f"Model or hyperparameter file not found. Check paths:\n- {weights_path}\n- {hp_path}"
                logger.critical(msg)
                print(msg)
                sys.exit()

            with open(hp_path, 'r') as f: hp_values = json.load(f)
            best_hps = kt.HyperParameters()
            best_hps.values = hp_values
            msg = f"Loaded hyperparameters from {hp_path}: \n{best_hps.values}"
            logger.info(msg)
            print(msg)            
            trained_initial_model = hypermodel.build(best_hps)
            trained_initial_model.load_weights(weights_path)
            msg = f"Loaded model weights from {weights_path}."
            logger.info(msg)
            print(msg)
        
        else:
            # If no pre-trained model is provided, start a fresh run.
            msg = f"\n--- No pre-trained model specified. Starting fresh run. ---"
            logger.info(msg)
            print(msg)
            TUNER_DIR = config.TUNER_DIR
            HP_FILE_PATH = os.path.join(TUNER_DIR, 'best_hyperparameters.json')

            if config.RUN_HYPERPARAMETER_TUNING:
                # Perform hyperparameter search if configured to do so.
                logger.info("\n--- Starting Hyperparameter Search on Initial Data ---")
                tuner_split_idx = int(len(initial_train_data) * 0.8)
                tuner = HestonTuner(hypermodel=hypermodel, objective="val_loss",
                                    max_epochs=SETTINGS['max_epochs_tuning'],
                                    factor=SETTINGS['hyperband_factor'], 
                                    executions_per_trial=SETTINGS['executions_per_trial'],
                                    directory=TUNER_DIR, project_name='heston_calibration', 
                                    overwrite=SETTINGS['overwrite_tuner'])
                tuner.search(train_data=initial_train_data_processed[:tuner_split_idx],
                             val_data=initial_train_data_processed[tuner_split_idx:], scaler=scaler,
                             cols=cols, logits=logits, settings=SETTINGS)
                tuner.results_summary()
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                with open(HP_FILE_PATH, 'w') as f: json.dump(best_hps.values, f, indent=4)
            else:
                # If tuning is disabled, load the best hyperparameters from a previous run.
                try:
                    with open(HP_FILE_PATH, 'r') as f: hp_values = json.load(f)
                    best_hps = kt.HyperParameters()
                    best_hps.values = hp_values
                except FileNotFoundError:
                    logger.critical(f"Hyperparameter file not found at '{HP_FILE_PATH}'. Run with RUN_HYPERPARAMETER_TUNING=True first.")
                    sys.exit()
            
            with open(os.path.join(RUN_OUTPUT_DIR, 'best_hyperparameters.json'), 'w') as f:
                json.dump(best_hps.values, f, indent=4)
            
            msg = f"\n--- Best Hyperparameters Selected ---\n{best_hps.values}"
            logger.info(msg)
            print(msg)
            
            # Perform the initial model training using the best hyperparameters.
            val_split_idx = int(len(initial_train_data) * (1 - config.VALIDATION_SET_PERCENTAGE / 100))
            initial_train_split, initial_val_split = initial_train_data_processed[:val_split_idx], initial_train_data_processed[val_split_idx:]
            msg = f"Initial training data split into {len(initial_train_split)} training days and {len(initial_val_split)} validation days."
            logger.info(msg)
            print(msg)
            
            initial_model_to_train = hypermodel.build(best_hps)
            trained_initial_model = train_model_for_fold(
                initial_model_to_train, best_hps, initial_train_split, initial_val_split, 
                scaler, cols, logits, SETTINGS, fold_num=0 # Fold 0 indicates initial training.
            )

            initial_model_path = os.path.join(RUN_OUTPUT_DIR, "fold_0_initial_model.weights.h5")
            trained_initial_model.save_weights(initial_model_path)
            msg = f"Trained initial model saved to {initial_model_path}."
            logger.info(msg)
            print(msg)
        
        # Start the main adaptive window evaluation process.
        if monitoring_data:
            run_adaptive_window_evaluation(trained_initial_model, scaler, cols, logits, SETTINGS, best_hps,
                                           initial_train_data_processed, monitoring_data_processed, RUN_OUTPUT_DIR)
        else:
            msg = "No monitoring data available to proceed with adaptive window evaluation."
            logger.warning(msg)
            print(msg)
            
    except Exception:
        logger.critical("An unexpected error occurred in the main execution block.", exc_info=True)
        traceback.print_exc()