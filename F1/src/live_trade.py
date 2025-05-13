import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import ccxt
import logging
from datetime import datetime
from data import prepare_data
from models import LSTMPredictor, GRUPredictor, CNNPredictor, MLPPredictor
from ensemble import EnsembleModel
from backtest import generate_signals_from_ensemble

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../deploy/trading_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_trader")

class LiveTrader:
    """
    Live trader that executes trades based on ensemble models predictions.
    """
    def __init__(
        self,
        exchange_id='binance',
        symbol='BTC/USDT',
        timeframe='1m',
        api_key=None,
        api_secret=None,
        threshold=0.52,
        cash_limit=None,
        position_size=0.1,
        window_size=60,
        commission=0.001,
        model_paths=None,
        ensemble_method='weighted_average',
        dry_run=True
    ):
        """
        Initialize the live trader.
        
        Args:
            exchange_id: ID of the exchange to use (default: 'binance')
            symbol: Trading symbol (default: 'BTC/USDT')
            timeframe: Data timeframe (default: '1m')
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            threshold: Probability threshold for buy/sell decisions (default: 0.52)
            cash_limit: Maximum amount of cash to use (default: None, use all available)
            position_size: Size of position relative to available cash (default: 0.1 = 10%)
            window_size: Size of the observation window (default: 60)
            commission: Commission rate for trades (default: 0.001)
            model_paths: Paths to pre-trained models
            ensemble_method: Method for combining predictions (default: 'weighted_average')
            dry_run: If True, only simulate trades without executing them (default: True)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.threshold = threshold
        self.cash_limit = cash_limit
        self.position_size = position_size
        self.window_size = window_size
        self.commission = commission
        self.dry_run = dry_run
        
        # Set up exchange
        self.exchange = self._setup_exchange(exchange_id, api_key, api_secret)
        
        # Load models
        if model_paths is None:
            model_paths = {
                'lstm': 'lstm_model.pt',
                'gru': 'gru_model.pt',
                'cnn': 'cnn_model.pt',
                'mlp': 'mlp_model.pt'
            }
        
        self.models = self._load_models(model_paths)
        
        # Create ensemble models
        if self.models:
            self.ensemble_model = EnsembleModel(
                models=list(self.models.values()),
                ensemble_method=ensemble_method
            )
        else:
            logger.error("No models were loaded, cannot create ensemble.")
            raise ValueError("No models were loaded, cannot create ensemble.")
        
        # Trading state
        self.in_position = False
        self.entry_price = 0
        self.position_amount = 0
        self.trades = []
        self.last_signal_time = None
        
        logger.info(f"LiveTrader initialized with: Symbol={symbol}, Threshold={threshold}, Dry Run={dry_run}")
    
    def _setup_exchange(self, exchange_id, api_key, api_secret):
        """Set up the exchange connection."""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'adjustForTimeDifference': True}
            })
            exchange.load_markets()
            logger.info(f"Connected to {exchange_id} exchange")
            return exchange
        except Exception as e:
            if self.dry_run:
                logger.warning(f"Failed to connect to exchange: {e}. Running in dry run mode.")
                return None
            else:
                logger.error(f"Failed to connect to exchange: {e}")
                raise
    
    def _load_models(self, model_paths):
        """Load pre-trained models."""
        models = {}
        input_size = 28  # Default assumption, will be updated when fetching data
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                logger.info(f"Loading {name.upper()} models from {path}...")
                
                # Create models
                if name == 'lstm':
                    model = LSTMPredictor(input_size=input_size)
                elif name == 'gru':
                    model = GRUPredictor(input_size=input_size)
                elif name == 'cnn':
                    model = CNNPredictor(input_size=input_size, seq_length=self.window_size)
                elif name == 'mlp':
                    model = MLPPredictor(input_size=input_size, seq_length=self.window_size)
                
                # Load state dict
                model.load_state_dict(torch.load(path))
                model.eval()
                
                models[name] = model
                logger.info(f"Successfully loaded {name.upper()} models")
            else:
                logger.warning(f"{name.upper()} models not found at {path}")
        
        if not models:
            logger.error("No models were loaded. Check models paths.")
        
        return models
    
    def fetch_latest_data(self, limit=500):
        """
        Fetch and prepare the latest market data.
        
        Args:
            limit: Number of candles to fetch (default: 500)
            
        Returns:
            DataFrame with normalized data
        """
        try:
            logger.info(f"Fetching {limit} candles of {self.timeframe} data for {self.symbol}")
            
            if self.exchange is None and self.dry_run:
                # Use local data for dry run if exchange is not connected
                logger.info("Using local data for dry run")
                _, _, _, _, _, _, df_normalized = prepare_data(
                    symbol=self.symbol.replace('/', '_'),
                    timeframe=self.timeframe,
                    days=1,
                    seq_length=self.window_size,
                    load_path='data/btc_usdt_1s.csv'
                )
                return df_normalized
            
            # Fetch OHLCV data from exchange
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate features
            df['returns'] = df['Close'].pct_change()
            df['direction'] = np.where(df['returns'] > 0, 1, 0)
            
            # Moving averages
            for period in [5, 10, 20, 50, 100]:
                df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'ma_vol_{period}'] = df['Volume'].rolling(window=period).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain_14 = gain.rolling(window=14).mean()
            avg_loss_14 = loss.rolling(window=14).mean()
            rs_14 = avg_gain_14 / avg_loss_14
            df['rsi_14'] = 100 - (100 / (1 + rs_14))
            
            avg_gain_28 = gain.rolling(window=28).mean()
            avg_loss_28 = loss.rolling(window=28).mean()
            rs_28 = avg_gain_28 / avg_loss_28
            df['rsi_28'] = 100 - (100 / (1 + rs_28))
            
            # Bollinger Bands
            df['bb_middle_20'] = df['Close'].rolling(window=20).mean()
            df['bb_std_20'] = df['Close'].rolling(window=20).std()
            df['bb_upper_20'] = df['bb_middle_20'] + 2 * df['bb_std_20']
            df['bb_lower_20'] = df['bb_middle_20'] - 2 * df['bb_std_20']
            df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_middle_20']
            
            # MACD
            df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Normalize data
            # Here we're doing a simple min-max normalization, but you might want to use a more sophisticated method
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_normalized = df.copy()
            
            for col in numeric_cols:
                if col != 'direction':  # Don't normalize the target variable
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            
            logger.info(f"Successfully fetched and prepared data with shape {df_normalized.shape}")
            
            # Update models input size if needed
            if list(self.models.values()):
                input_size = df_normalized.shape[1] - 1  # -1 for direction column
                for model in self.models.values():
                    if hasattr(model, 'lstm') and model.lstm.input_size != input_size:
                        model.lstm.input_size = input_size
                    elif hasattr(model, 'gru') and model.gru.input_size != input_size:
                        model.gru.input_size = input_size
                    # Add similar checks for other models types if needed
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def generate_signal(self, df):
        """
        Generate trading signal from the ensemble models.
        
        Args:
            df: DataFrame with normalized data
            
        Returns:
            Signal (-1 for sell, 0 for hold, 1 for buy)
        """
        if len(df) < self.window_size:
            logger.warning(f"Not enough data points. Need at least {self.window_size}, got {len(df)}")
            return 0
        
        try:
            # Generate signals using the ensemble models
            df_signals = generate_signals_from_ensemble(
                self.ensemble_model, 
                df, 
                self.window_size, 
                self.threshold
            )
            
            # Get the latest signal
            latest_signal = df_signals['Signal'].iloc[-1]
            
            logger.info(f"Generated signal: {latest_signal}")
            return latest_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0
    
    def get_account_balance(self):
        """Get account balance."""
        if self.exchange is None or self.dry_run:
            return {'USDT': 10000.0, 'BTC': 0.0}  # Mock balance for dry run
        
        try:
            balance = self.exchange.fetch_balance()
            base_currency = self.symbol.split('/')[0]
            quote_currency = self.symbol.split('/')[1]
            
            base_balance = float(balance[base_currency]['free'])
            quote_balance = float(balance[quote_currency]['free'])
            
            return {base_currency: base_balance, quote_currency: quote_balance}
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_current_price(self):
        """Get current market price."""
        if self.exchange is None and self.dry_run:
            # For dry run, use random walk price
            if not hasattr(self, 'mock_price'):
                self.mock_price = 30000.0  # Starting price
            
            # Random walk with 0.1% standard deviation
            random_change = np.random.normal(0, 0.001)
            self.mock_price *= (1 + random_change)
            return self.mock_price
        
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            raise
    
    def execute_trade(self, signal):
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal (-1 for sell, 0 for hold, 1 for buy)
            
        Returns:
            True if trade was executed, False otherwise
        """
        if signal == 0:
            return False
        
        current_price = self.get_current_price()
        balance = self.get_account_balance()
        
        base_currency = self.symbol.split('/')[0]
        quote_currency = self.symbol.split('/')[1]
        
        logger.info(f"Current price: {current_price}, Balance: {balance}")
        
        try:
            if signal == 1 and not self.in_position:  # Buy signal
                # Calculate amount to buy
                quote_amount = balance[quote_currency]
                if self.cash_limit is not None:
                    quote_amount = min(quote_amount, self.cash_limit)
                
                trade_amount = (quote_amount * self.position_size) / current_price
                cost = trade_amount * current_price
                
                if self.exchange is not None and not self.dry_run:
                    # Execute buy order
                    order = self.exchange.create_market_buy_order(
                        self.symbol,
                        trade_amount
                    )
                    logger.info(f"Buy order executed: {order}")
                else:
                    logger.info(f"[DRY RUN] Would buy {trade_amount} {base_currency} at {current_price} {quote_currency}")
                
                # Update state
                self.in_position = True
                self.entry_price = current_price
                self.position_amount = trade_amount
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'type': 'buy',
                    'price': current_price,
                    'amount': trade_amount,
                    'cost': cost,
                    'fee': cost * self.commission
                }
                self.trades.append(trade)
                logger.info(f"Buy executed: {trade}")
                
                return True
                
            elif signal == -1 and self.in_position:  # Sell signal
                if self.exchange is not None and not self.dry_run:
                    # Execute sell order
                    order = self.exchange.create_market_sell_order(
                        self.symbol,
                        self.position_amount
                    )
                    logger.info(f"Sell order executed: {order}")
                else:
                    logger.info(f"[DRY RUN] Would sell {self.position_amount} {base_currency} at {current_price} {quote_currency}")
                
                # Calculate profit/loss
                cost = self.position_amount * current_price
                profit = self.position_amount * (current_price - self.entry_price) - (cost * self.commission)
                profit_pct = (current_price / self.entry_price - 1) * 100 - (self.commission * 100)
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'type': 'sell',
                    'price': current_price,
                    'amount': self.position_amount,
                    'cost': cost,
                    'fee': cost * self.commission,
                    'profit': profit,
                    'profit_pct': profit_pct
                }
                self.trades.append(trade)
                logger.info(f"Sell executed: {trade}")
                
                # Update state
                self.in_position = False
                self.position_amount = 0
                self.entry_price = 0
                
                return True
                
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise
    
    def run_trading_cycle(self):
        """Run a single trading cycle."""
        try:
            # Fetch latest data
            df = self.fetch_latest_data()
            
            # Generate signal
            signal = self.generate_signal(df)
            
            # Execute trade if there's a signal
            if signal != 0:
                current_time = datetime.now()
                # Only execute if we haven't already traded in this minute or signal is different
                if (self.last_signal_time is None or
                    (current_time - self.last_signal_time).seconds >= 60):
                    success = self.execute_trade(signal)
                    if success:
                        self.last_signal_time = current_time
                        logger.info(f"Trade executed at {current_time}")
                else:
                    logger.info("Skipping trade execution - already traded in this minute")
            
            # Log current state
            if self.in_position:
                current_price = self.get_current_price()
                unrealized_profit = self.position_amount * (current_price - self.entry_price)
                unrealized_profit_pct = (current_price / self.entry_price - 1) * 100
                logger.info(f"In position: {self.position_amount} at {self.entry_price}, "
                           f"Current price: {current_price}, "
                           f"Unrealized P/L: {unrealized_profit:.2f} ({unrealized_profit_pct:.2f}%)")
            else:
                logger.info("Not in position")
            
            # Calculate and log performance metrics
            if self.trades:
                profits = [trade.get('profit', 0) for trade in self.trades if 'profit' in trade]
                if profits:
                    total_profit = sum(profits)
                    win_trades = sum(1 for p in profits if p > 0)
                    win_rate = win_trades / len(profits) if profits else 0
                    
                    logger.info(f"Performance: Total trades: {len(profits)}, "
                               f"Win rate: {win_rate:.2%}, "
                               f"Total profit: {total_profit:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return 0
    
    def run_continuous(self, interval=60):
        """
        Run the trading system continuously.
        
        Args:
            interval: Interval between cycles in seconds (default: 60 seconds = 1 minute)
        """
        logger.info(f"Starting continuous trading with {interval}s interval")
        
        try:
            while True:
                cycle_start = time.time()
                
                # Run a trading cycle
                self.run_trading_cycle()
                
                # Calculate time to sleep
                elapsed = time.time() - cycle_start
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"Sleeping for {sleep_time:.2f}s until next cycle")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Trading cycle took longer than interval: {elapsed:.2f}s")
        
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous trading: {e}")
            raise
        finally:
            self.save_trade_history()
    
    def save_trade_history(self, filename="trade_history.csv"):
        """Save trade history to CSV file."""
        if self.trades:
            try:
                df_trades = pd.DataFrame(self.trades)
                df_trades.to_csv(filename, index=False)
                logger.info(f"Trade history saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving trade history: {e}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Live Crypto Trading with Ensemble Model')
    
    # Exchange arguments
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to use (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1m',
                        help='Data timeframe (default: 1m)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for the exchange')
    parser.add_argument('--api-secret', type=str, default=None,
                        help='API secret for the exchange')
    
    # Trading arguments
    parser.add_argument('--threshold', type=float, default=0.52,
                        help='Probability threshold for signals (default: 0.52)')
    parser.add_argument('--cash-limit', type=float, default=None,
                        help='Maximum amount of cash to use (default: None, use all available)')
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as fraction of available cash (default: 0.1 = 10%%)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between trading cycles in seconds (default: 60)')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Size of the observation window (default: 60)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate for trades (default: 0.001 = 0.1%%)')
    
    # Model arguments
    parser.add_argument('--lstm-models-path', type=str, default='lstm_model.pt',
                        help='Path to pre-trained LSTM models (default: lstm_model.pt)')
    parser.add_argument('--gru-models-path', type=str, default='gru_model.pt',
                        help='Path to pre-trained GRU models (default: gru_model.pt)')
    parser.add_argument('--cnn-models-path', type=str, default='cnn_model.pt',
                        help='Path to pre-trained CNN models (default: cnn_model.pt)')
    parser.add_argument('--mlp-models-path', type=str, default='mlp_model.pt',
                        help='Path to pre-trained MLP models (default: mlp_model.pt)')
    parser.add_argument('--ensemble-method', type=str, default='weighted_average',
                        choices=['average', 'weighted_average', 'voting'],
                        help='Method for combining predictions (default: weighted_average)')
    
    # Run mode
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode (no real trades) (default: True)')
    parser.add_argument('--single-cycle', action='store_true',
                        help='Run a single trading cycle and exit (default: False)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create models paths dictionary
    model_paths = {
        'lstm': args.lstm_model_path,
        'gru': args.gru_model_path,
        'cnn': args.cnn_model_path,
        'mlp': args.mlp_model_path
    }
    
    # Create trader
    trader = LiveTrader(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        api_key=args.api_key,
        api_secret=args.api_secret,
        threshold=args.threshold,
        cash_limit=args.cash_limit,
        position_size=args.position_size,
        window_size=args.window_size,
        commission=args.commission,
        model_paths=model_paths,
        ensemble_method=args.ensemble_method,
        dry_run=args.dry_run
    )
    
    # Run trader
    if args.single_cycle:
        logger.info("Running a single trading cycle")
        trader.run_trading_cycle()
    else:
        logger.info("Running continuous trading")
        trader.run_continuous(interval=args.interval)

if __name__ == "__main__":
    main()