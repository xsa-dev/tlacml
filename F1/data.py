import os
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def fetch_ohlcv_data(symbol='BTC/USDT', timeframe='1s', days=1):
    """
    Fetch OHLCV data with second resolution.
    Note: Most exchanges limit historical data for second resolution,
    so we'll fetch data for the last 'days' days.
    """
    exchange = ccxt.binance()
    
    # Calculate start time (days ago from now)
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    print(f"Fetching {timeframe} data for {symbol} since {exchange.iso8601(since)}")
    
    # Fetch data in chunks to avoid rate limits
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
        print(f"Error fetching data: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetched {len(df)} data points")
    return df

def add_features(df):
    """
    Add technical indicators and features to the dataframe.
    """
    # Copy the dataframe to avoid modifying the original
    df = df.copy()
    
    # Price changes
    df['returns'] = df['close'].pct_change()
    df['direction'] = np.where(df['returns'] > 0, 1, 0)  # 1 if price went up, 0 if down
    
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ma_vol_{window}'] = df['volume'].rolling(window=window).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    for window in [14, 28]:
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for window in [20]:
        df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
        df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def normalize_data(df):
    """
    Normalize data using min-max scaling.
    """
    # Copy the dataframe to avoid modifying the original
    df = df.copy()
    
    # Save the direction column for later
    direction = df['direction'].copy()
    
    # Columns to exclude from normalization
    exclude_cols = ['direction']
    
    # Normalize all columns except excluded ones
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    
    # Restore the direction column
    df['direction'] = direction
    
    return df

def create_sequences(df, seq_length=60):
    """
    Create sequences for time series models.
    """
    # Get feature columns (all except 'direction')
    feature_cols = [col for col in df.columns if col != 'direction']
    
    # Convert to numpy arrays
    features = df[feature_cols].values
    target = df['direction'].values
    
    X, y = [], []
    
    for i in range(len(df) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    
    return np.array(X), np.array(y)

def prepare_data(symbol='BTC/USDT', timeframe='1s', days=1, seq_length=60, save_path=None, load_path=None):
    """
    Prepare data for training and testing.
    """
    if load_path and os.path.exists(load_path):
        print(f"Loading data from {load_path}")
        df = pd.read_csv(load_path, index_col=0, parse_dates=True)
    else:
        df = fetch_ohlcv_data(symbol, timeframe, days)
        if save_path:
            df.to_csv(save_path)
            print(f"Data saved to {save_path}")
    
    # Add features
    df_with_features = add_features(df)
    print(f"DataFrame shape after adding features: {df_with_features.shape}")
    
    # Normalize data
    df_normalized = normalize_data(df_with_features)
    
    # Create sequences
    X, y = create_sequences(df_normalized, seq_length)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, df_normalized
