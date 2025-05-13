import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from typing import Dict, Any, Optional, Union, List, Tuple
import os
from datetime import datetime

class SignalStrategy(Strategy):
    """
    A strategy that uses pre-calculated signals from a DataFrame.
    
    Signals:
    - 1: Buy
    - -1: Sell
    - 0: Hold
    """
    
    def init(self):
        """
        Initialize the strategy.
        """
        # Get the Signal column from the data
        self.signal = self.data.Signal
    
    def next(self):
        """
        For each bar, check the signal and execute trades accordingly.
        """
        # Check if we have a buy signal and no open position
        if self.signal[-1] == 1 and not self.position:
            # Enter a long position with all available cash
            self.buy()
        
        # Check if we have a sell signal and an open position
        elif self.signal[-1] == -1 and self.position:
            # Close the position
            self.position.close()


def prepare_backtest_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for backtesting.
    
    Args:
        df: DataFrame with OHLCV data and Signal column
        
    Returns:
        DataFrame formatted for backtesting.py
    """
    # Make a copy to avoid modifying the original
    bt_data = df.copy()
    
    # Ensure column names are correct for backtesting.py
    # It expects: Open, High, Low, Close, Volume
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # Rename columns if they exist in lowercase
    for old_col, new_col in column_mapping.items():
        if old_col in bt_data.columns:
            bt_data.rename(columns={old_col: new_col}, inplace=True)
    
    # Check for missing required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Signal']
    missing_columns = [col for col in required_columns if col not in bt_data.columns]
    
    # Print column names for debugging
    print(f"\nAvailable columns in dataframe: {bt_data.columns.tolist()}")
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        
        # Try to fix common issues
        if 'Open' in missing_columns and 'open' in bt_data.columns:
            bt_data['Open'] = bt_data['open']
        if 'High' in missing_columns and 'high' in bt_data.columns:
            bt_data['High'] = bt_data['high']
        if 'Low' in missing_columns and 'low' in bt_data.columns:
            bt_data['Low'] = bt_data['low']
        if 'Close' in missing_columns and 'close' in bt_data.columns:
            bt_data['Close'] = bt_data['close']
        if 'Volume' in missing_columns and 'volume' in bt_data.columns:
            bt_data['Volume'] = bt_data['volume']
            
        # Check again after fixes
        missing_columns = [col for col in required_columns if col not in bt_data.columns]
        if missing_columns:
            raise ValueError(f"Still missing required columns after fixes: {missing_columns}")
    
    # Convert index to datetime if it's not already
    if not isinstance(bt_data.index, pd.DatetimeIndex):
        if 'timestamp' in bt_data.columns:
            bt_data.set_index('timestamp', inplace=True)
        else:
            # Create a datetime index if none exists
            bt_data.index = pd.date_range(
                start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                periods=len(bt_data),
                freq='1s'
            )
    
    return bt_data


def generate_signals_from_agent(env, agent, df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Generate trading signals from a trained PPO agent.
    
    Args:
        env: Trading environment
        agent: Trained PPO agent
        df: DataFrame with OHLCV data
        window_size: Size of the observation window
        
    Returns:
        DataFrame with added Signal column
    """
    # Make a copy to avoid modifying the original
    df_signals = df.copy()
    
    # Reset environment
    observation, _ = env.reset()
    
    # Initialize signals array
    signals = np.zeros(len(df_signals))
    
    # Start from window_size to match environment
    current_step = window_size
    
    # Run through the data
    while current_step < len(df_signals) - 1:
        # Get action from agent
        action, _, _ = agent.choose_action(observation)
        
        # Convert action to signal
        # 0: Hold, 1: Buy, 2: Sell
        if action == 1:  # Buy
            signals[current_step] = 1
        elif action == 2:  # Sell
            signals[current_step] = -1
        
        # Step environment
        observation, _, done, _, _ = env.step(action)
        
        # Move to next step
        current_step += 1
        
        if done:
            break
    
    # Add signals to DataFrame
    df_signals['Signal'] = signals
    
    return df_signals


def generate_signals_from_ensemble(ensemble_model, df: pd.DataFrame, window_size: int, threshold: float = 0.55) -> pd.DataFrame:
    """
    Generate trading signals from an ensemble models.
    
    Args:
        ensemble_model: Trained ensemble models
        df: DataFrame with OHLCV data and features
        window_size: Size of the observation window
        threshold: Probability threshold for buy/sell decisions
        
    Returns:
        DataFrame with added Signal column
    """
    import torch
    
    # Make a copy to avoid modifying the original
    df_signals = df.copy()
    
    # Initialize signals array
    signals = np.zeros(len(df_signals))
    
    # Get feature columns (all except 'timestamp' and 'direction')
    feature_cols = [col for col in df_signals.columns if col not in ['timestamp', 'direction']]
    
    # Track our position to avoid repeated buy/sell signals
    in_position = False
    
    # Track prediction probabilities for diagnostics
    all_probs = []
    
    # For more aggressive strategy
    buy_threshold = threshold - 0.02  # Lower threshold for buying
    sell_threshold = 1 - threshold + 0.02  # Higher threshold for selling
    
    # Trend tracking variables
    up_trend_count = 0
    down_trend_count = 0
    trend_memory = 5  # How many predictions to consider for trend
    
    print(f"Using buy threshold: {buy_threshold:.4f}, sell threshold: {sell_threshold:.4f}")
    
    # Start from window_size
    for i in range(window_size, len(df_signals) - 1):
        # Get data for the current window
        window_data = df_signals.iloc[i - window_size:i][feature_cols].values
        window_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Get ensemble prediction
        with torch.no_grad():
            probs = ensemble_model.predict(window_tensor)[0]
            up_prob = probs[1].item()  # Probability of price going up
        
        all_probs.append(up_prob)
        
        # Update trend counters
        if up_prob > 0.5:
            up_trend_count += 1
            down_trend_count = max(0, down_trend_count - 1)
        else:
            down_trend_count += 1
            up_trend_count = max(0, up_trend_count - 1)
            
        # Keep trend counters within range
        up_trend_count = min(trend_memory, up_trend_count)
        down_trend_count = min(trend_memory, down_trend_count)
        
        # More aggressive trading strategy with dynamic thresholds
        if (up_prob > buy_threshold or up_trend_count >= trend_memory - 1) and not in_position:
            signals[i] = 1  # Buy
            in_position = True
            print(f"BUY signal at step {i}, up_prob: {up_prob:.4f}, up_trend: {up_trend_count}")
        elif (up_prob < sell_threshold or down_trend_count >= trend_memory - 1) and in_position:
            signals[i] = -1  # Sell
            in_position = False
            print(f"SELL signal at step {i}, up_prob: {up_prob:.4f}, down_trend: {down_trend_count}")
    
    # Add signals to DataFrame
    df_signals['Signal'] = signals
    
    # Print diagnostics
    all_probs = np.array(all_probs)
    print(f"\nPrediction probabilities statistics:")
    print(f"Min: {all_probs.min():.4f}, Max: {all_probs.max():.4f}, Mean: {all_probs.mean():.4f}")
    print(f"Above buy threshold ({buy_threshold}): {(all_probs > buy_threshold).sum()} ({(all_probs > buy_threshold).sum() / len(all_probs) * 100:.2f}%)")
    print(f"Below sell threshold ({sell_threshold}): {(all_probs < sell_threshold).sum()} ({(all_probs < sell_threshold).sum() / len(all_probs) * 100:.2f}%)")
    print(f"In neutral zone: {((all_probs <= buy_threshold) & (all_probs >= sell_threshold)).sum()} ({((all_probs <= buy_threshold) & (all_probs >= sell_threshold)).sum() / len(all_probs) * 100:.2f}%)")
    print(f"Total predictions: {len(all_probs)}")
    
    # Calculate potential signals based on trend analysis
    print(f"Potential additional buy signals from trend analysis: {((all_probs <= buy_threshold) & (all_probs > 0.5)).sum()}")
    print(f"Potential additional sell signals from trend analysis: {((all_probs >= sell_threshold) & (all_probs < 0.5)).sum()}")
    
    return df_signals


def run_backtest(df: pd.DataFrame, cash: float = 10000, commission: float = 0.002, 
                plot: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a backtest on the given DataFrame with trading signals.
    
    Args:
        df: DataFrame with OHLCV data and Signal column (1=Buy, -1=Sell, 0=Hold)
        cash: Initial cash amount
        commission: Commission rate for trades
        plot: Whether to plot the backtest results
        save_path: Path to save the plot and results
        
    Returns:
        Dictionary with backtest metrics
    """
    # Prepare data for backtesting
    bt_data = prepare_backtest_data(df)
    
    # Create and run backtest
    bt = Backtest(bt_data, SignalStrategy, cash=cash, commission=commission)
    result = bt.run()
    
    # Extract key metrics
    metrics = {
        'total_return': result['Return [%]'] / 100,
        'sharpe_ratio': result['Sharpe Ratio'],
        'max_drawdown': result['Max. Drawdown [%]'] / 100,
        'win_rate': result['Win Rate [%]'] / 100,
        'trades': result['# Trades'],
        'final_equity': result['Equity Final [$]'],
        'buy_hold_return': result['Buy & Hold Return [%]'] / 100,
        'sqn': result['SQN']
    }
    
    # Plot results if requested
    if plot:
        # Plot backtest results using the default method
        bt.plot()
        
        # Create separate figure for equity curve and drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot equity curve
        result['_equity_curve']['Equity'].plot(ax=ax1)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity [$]')
        ax1.grid(True)
        
        # Plot drawdown
        result['_equity_curve']['DrawdownPct'].plot(ax=ax2)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown [%]')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_csv_path = os.path.splitext(save_path)[0] + '_metrics.csv'
            metrics_df.to_csv(metrics_csv_path, index=False)
            
            # Save trade log to CSV
            trades_csv_path = os.path.splitext(save_path)[0] + '_trades.csv'
            result['_trades'].to_csv(trades_csv_path, index=False)
        
        plt.show()
    
    return metrics


def compare_strategies(df: pd.DataFrame, strategies: Dict[str, np.ndarray], 
                      cash: float = 10000, commission: float = 0.002,
                      save_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple trading strategies on the same data.
    
    Args:
        df: DataFrame with OHLCV data
        strategies: Dictionary mapping strategy names to signal arrays
        cash: Initial cash amount
        commission: Commission rate for trades
        save_path: Path to save the plot and results
        
    Returns:
        Dictionary mapping strategy names to their metrics
    """
    results = {}
    
    # Create figure for comparison
    plt.figure(figsize=(15, 10))
    
    # Plot price
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax1.plot(df.index, df['Close'], label='Price', color='black', alpha=0.5)
    ax1.set_title('Price and Signals')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot equity curves
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    
    # Run backtest for each strategy
    for name, signals in strategies.items():
        # Create a copy of the DataFrame with the current strategy's signals
        df_strategy = df.copy()
        df_strategy['Signal'] = signals
        
        # Print signal counts for this strategy
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        print(f"\nStrategy '{name}': {buy_count} buy signals, {sell_count} sell signals")
        
        # Run backtest
        bt_data = prepare_backtest_data(df_strategy)
        bt = Backtest(bt_data, SignalStrategy, cash=cash, commission=commission)
        result = bt.run()
        
        # Extract metrics
        metrics = {
            'total_return': result['Return [%]'] / 100,
            'sharpe_ratio': result['Sharpe Ratio'],
            'max_drawdown': result['Max. Drawdown [%]'] / 100,
            'win_rate': result['Win Rate [%]'] / 100,
            'trades': result['# Trades'],
            'final_equity': result['Equity Final [$]'],
            'buy_hold_return': result['Buy & Hold Return [%]'] / 100,
            'sqn': result['SQN']
        }
        
        results[name] = metrics
        
        # Plot equity curve
        result['_equity_curve']['Equity'].plot(ax=ax2, label=name)
        
        # Plot buy/sell signals on price chart
        buy_signals = df_strategy[df_strategy['Signal'] == 1]
        sell_signals = df_strategy[df_strategy['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color=f'C{list(strategies.keys()).index(name)}', 
                   s=100, label=f'{name} Buy')
        ax1.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color=f'C{list(strategies.keys()).index(name)}', 
                   s=100, label=f'{name} Sell')
    
    ax1.legend(loc='upper left')
    ax2.set_title('Equity Curves')
    ax2.set_ylabel('Equity [$]')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(results).T
        metrics_csv_path = os.path.splitext(save_path)[0] + '_comparison_metrics.csv'
        metrics_df.to_csv(metrics_csv_path)
    
    plt.show()
    
    return results


def create_buy_hold_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Create signals for a buy and hold strategy.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Array of signals (1 for buy at start, 0 elsewhere)
    """
    signals = np.zeros(len(df))
    signals[min(20, len(df)-1)] = 1  # Buy after a few ticks to match other strategies start time
    print(f"Buy & Hold: 1 buy signal at position {min(20, len(df)-1)}")
    return signals


def create_sma_crossover_signals(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> np.ndarray:
    """
    Create signals based on SMA crossover strategy.
    
    Args:
        df: DataFrame with OHLCV data
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        Array of signals
    """
    # Calculate moving averages
    df_temp = df.copy()
    
    # Use 'close' or 'Close' column depending on which exists
    price_col = 'Close' if 'Close' in df_temp.columns else 'close'
    
    df_temp['short_ma'] = df_temp[price_col].rolling(window=short_window).mean()
    df_temp['long_ma'] = df_temp[price_col].rolling(window=long_window).mean()
    
    # Initialize signals
    signals = np.zeros(len(df_temp))
    
    # Track position to create more realistic signals
    in_position = False
    
    # Generate signals based on crossovers
    for i in range(long_window, len(df_temp)):
        # Golden cross (short MA crosses above long MA)
        if (df_temp['short_ma'].iloc[i-1] <= df_temp['long_ma'].iloc[i-1] and 
            df_temp['short_ma'].iloc[i] > df_temp['long_ma'].iloc[i] and
            not in_position):
            signals[i] = 1  # Buy
            in_position = True
        
        # Death cross (short MA crosses below long MA)
        elif (df_temp['short_ma'].iloc[i-1] >= df_temp['long_ma'].iloc[i-1] and 
              df_temp['short_ma'].iloc[i] < df_temp['long_ma'].iloc[i] and
              in_position):
            signals[i] = -1  # Sell
            in_position = False
    
    # Print diagnostic information
    buy_count = (signals == 1).sum()
    sell_count = (signals == -1).sum()
    print(f"SMA Crossover ({short_window}/{long_window}): {buy_count} buy signals, {sell_count} sell signals")
    
    return signals
