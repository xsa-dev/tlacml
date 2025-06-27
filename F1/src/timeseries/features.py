import numpy as np


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
    Normalize timeseries using min-max scaling.
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