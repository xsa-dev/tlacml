import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta

from src.timeseries.features import add_features, normalize_data, create_sequences
from src.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_DAYS, DEFAULT_SEQ_LENGTH

def fetch_ohlcv_data(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, days=DEFAULT_DAYS):
    """
    Fetch OHLCV timeseries with second resolution.
    
    TODO Future improvements:
    1. Split into smaller functions:
       - initialize_exchange(): Handle exchange setup
       - calculate_time_range(): Calculate start/end times
       - fetch_data_in_chunks(): Handle chunked timeseries fetching
       - convert_to_dataframe(): Format timeseries as DataFrame
    
    2. Add progress tracking with tqdm for better UX during long fetches
    
    3. Add configuration options for different exchanges
    """
    # TODO Future improvement: Create retry mechanism for API calls
    # 
    # 1. Implement exponential backoff for failed requests:
    #    - Start with small delay (e.g., 1s)
    #    - Double delay after each failure
    #    - Cap at reasonable maximum (e.g., 60s)
    #
    # 2. Handle specific exchange errors differently:
    #    - Rate limiting errors: Wait and retry
    #    - Authentication errors: Fail fast
    #    - Network errors: Retry with backoff
    #
    # 3. Add circuit breaker pattern to avoid hammering failing APIs
    #    - Track consecutive failures
    #    - Temporarily disable requests after threshold reached
    # TODO Future improvement: Add timeseries validation
    #
    # 1. Validate raw OHLCV timeseries for common issues:
    #    - Check for empty datasets
    #    - Detect and handle missing timestamps
    #    - Identify outliers (extreme price values)
    #    - Validate timestamp sequence integrity
    #
    # 2. Add timeseries quality metrics:
    #    - Percentage of missing values
    #    - Number of gaps in time series
    #    - Statistics on price jumps/volatility
    #
    # 3. Implement recovery strategies:
    #    - Interpolation for small gaps
    #    - Retry fetching for segments with problems
    #    - Warning/error thresholds for quality issues
    exchange = ccxt.binance()
    
    # Calculate start time (days ago from now)
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    print(f"Fetching {timeframe} timeseries for {symbol} since {exchange.iso8601(since)}")
    
    # Fetch timeseries in chunks to avoid rate limits
    all_ohlcv = []
    current_since = since
    
    try:
        while True:
            print(f"Fetching chunk from {exchange.iso8601(current_since)}")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit=1000)
            
            if len(ohlcv) == 0:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Update since for next iteration
            current_since = ohlcv[-1][0] + 1  # +1 to avoid duplicates
            
            # Add delay to avoid rate limits
            time.sleep(1)
            
            # Check if we've reached current time
            if current_since >= exchange.milliseconds():
                break
    except Exception as e:
        print(f"Error fetching timeseries: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetched {len(df)} timeseries points")
    return df

# TODO Future improvement: Consolidate feature calculations
#
# 1. Extract shared feature engineering code between dataset.py and live_trade.py
#    - Create reusable functions for each feature group (MA, RSI, BB, etc.)
#    - Ensure consistency in feature calculation across modules
#
# 2. Add feature versioning/fingerprinting to track changes
#    - Generate feature set hash to detect when preprocessing changes
#    - Allow backwards compatibility with saved models
#
# 3. Consider using a feature store pattern:
#    - Cache calculated features to avoid redundant computation
#    - Track feature dependencies for incremental updates
# TODO Future improvement: Refine overall timeseries architecture
#
# 1. Implement proper pipeline design:
#    - Separate concerns between fetching, cleaning, and feature engineering
#    - Make pipeline resumable after failures
#    - Add logging at each stage
#
# 2. Consider adding timeseries versioning:
#    - Save raw and processed datasets with version tracking
#    - Enable reproducibility of training results
#
# 3. Optimize for memory efficiency:
#    - Process large datasets in chunks
#    - Use appropriate timeseries types to reduce memory usage
def prepare_data(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, days=DEFAULT_DAYS, 
                seq_length=DEFAULT_SEQ_LENGTH, save_path=None, load_path=None):
    """
    Prepare timeseries for training and testing.
    """
    if load_path and os.path.exists(load_path):
        print(f"Loading timeseries from {load_path}")
        df = pd.read_csv(load_path, index_col=0, parse_dates=True)
    else:
        df = fetch_ohlcv_data(symbol, timeframe, days)
        if save_path:
            df.to_csv(save_path)
            print(f"Data saved to {save_path}")
    
    # Add features
    df_with_features = add_features(df)
    print(f"DataFrame shape after adding features: {df_with_features.shape}")
    
    # Normalize timeseries
    df_normalized = normalize_data(df_with_features)
    
    # Create sequences
    X, y = create_sequences(df_normalized, seq_length)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split timeseries into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, df_normalized