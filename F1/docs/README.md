# Crypto Trading with Ensemble Models and Optional DRL (PPO)

This project implements a cryptocurrency trading system for BTC/USDT using an ensemble of predictive models on second-resolution data, with optional Deep Reinforcement Learning (PPO) for enhanced trading strategies.

## Features

- Second-resolution OHLCV data collection from Binance
- Multiple price direction prediction models:
  - LSTM
  - GRU
  - 1D CNN
  - MLP
- Ensemble model combining predictions from all models for direct trading signals
- Optional PPO reinforcement learning agent for advanced trading
- PyTorch Lightning and ClearML integration
- Performance comparison with Buy & Hold strategy
- **Advanced backtesting** with comprehensive metrics and visualizations

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- ClearML
- Gymnasium
- CCXT
- Backtesting.py
- Pandas, NumPy, Matplotlib
- MacBook with M1 chip (MPS or CPU)

## Project Structure

- `data.py`: Data collection and preprocessing
- `models.py`: Predictive models (LSTM, GRU, 1D CNN, MLP)
- `ensemble.py`: Ensemble model
- `env.py`: Trading environment
- `ppo.py`: PPO agent
- `backtest.py`: Backtesting functionality
- `train.py`: Training pipeline
- `main.py`: Main script

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-drl.git
cd crypto-trading-drl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training and Evaluation

Run the main script to train models and evaluate performance using ensemble predictions directly:
```bash
python main.py
```

Train with the PPO agent for more advanced trading strategies:
```bash
python main.py --use-ppo
```

### Backtesting Only

To run backtesting with ensemble-based trading (default):
```bash
python main.py --backtest-only
```

To run backtesting with the PPO agent:
```bash
python main.py --backtest-only --use-ppo
```

### Using Makefile

The project includes a Makefile with common commands:
```bash
make train            # Train with ensemble models (default)
make train-ppo        # Train with PPO agent
make backtest         # Backtest with ensemble models
make backtest-ppo     # Backtest with PPO agent
make train-custom     # Train with custom parameters
make backtest-custom  # Backtest with custom parameters
make clean            # Remove generated models files and plots
```

### Command Line Arguments

#### Data Arguments
- `--symbol`: Trading symbol (default: BTC/USDT)
- `--timeframe`: Data timeframe (default: 1s)
- `--days`: Number of days of data to fetch (default: 1)
- `--seq-length`: Sequence length for time series models (default: 60)
- `--data-path`: Path to save/load data (default: btc_usdt_1s.csv)
- `--load-data`: Load data from file if available
- `--save-data`: Save data to file

#### Training Arguments
- `--batch-size`: Batch size for training (default: 64)
- `--max-epochs`: Maximum number of epochs for training predictive models (default: 50)
- `--n-episodes`: Number of episodes for training PPO agent (default: 100)
- `--max-steps`: Maximum number of steps per episode (default: 1000)
- `--ensemble-method`: Method for combining predictions in the ensemble (default: weighted_average)
- `--use-ppo`: Use PPO reinforcement learning agent for trading (default: False, use ensemble directly)

#### ClearML Arguments
- `--use-clearml`: Use ClearML for experiment tracking
- `--project-name`: ClearML project name (default: CryptoTrading)
- `--task-name`: ClearML task name (default: PPO_BTC_USDT)

#### Backtesting Arguments
- `--backtest-only`: Run only backtesting on pre-trained models (default: False)
- `--cash`: Initial cash amount for backtesting (default: 10000)
- `--commission`: Commission rate for trades (default: 0.002)
- `--threshold`: Probability threshold for ensemble model signals (default: 0.6)
- `--agent-model-path`: Path to pre-trained PPO agent model (default: ppo_agent.pt)
- `--lstm-model-path`: Path to pre-trained LSTM model (default: lstm_model.pt)
- `--gru-model-path`: Path to pre-trained GRU model (default: gru_model.pt)
- `--cnn-model-path`: Path to pre-trained CNN model (default: cnn_model.pt)
- `--mlp-model-path`: Path to pre-trained MLP model (default: mlp_model.pt)
- `--save-results`: Save backtest results to CSV (default: False)
- `--results-path`: Path to save backtest results (default: backtest_results)

## Implementation Details

### Data Collection and Preprocessing

The system fetches second-resolution BTC/USDT data from Binance using CCXT. The data is preprocessed by adding technical indicators, normalizing, and creating sequences for the time series models.

### Predictive Models

Four different models are implemented for price direction prediction:
1. LSTM: Long Short-Term Memory network
2. GRU: Gated Recurrent Unit network
3. 1D CNN: One-dimensional Convolutional Neural Network
4. MLP: Multi-Layer Perceptron

### Ensemble Model

The ensemble model combines predictions from all four models using one of the following methods:
- Average: Simple average of probabilities
- Weighted Average: Weighted average of probabilities
- Voting: Majority vote

By default, the system uses the ensemble model's predictions directly to generate trading signals, without needing the PPO agent.

### Trading Environment

The trading environment is implemented using Gymnasium. It simulates a cryptocurrency trading environment with the following actions:
- Hold: Do nothing
- Buy: Buy cryptocurrency with all available balance
- Sell: Sell all cryptocurrency

### Trading Strategies

The system supports two main trading approaches:

1. **Ensemble-based Trading (Default)**: Uses the ensemble model's predictions directly:
   - Buy when predicted upward probability > threshold
   - Sell when predicted upward probability < (1-threshold)
   - Simple, rule-based approach that requires no reinforcement learning

2. **PPO-based Trading (Optional)**: Uses the PPO agent for decision making:
   - The PPO (Proximal Policy Optimization) agent is implemented using PyTorch Lightning
   - Learns optimal trading strategies through trial and error
   - Takes both market data and ensemble predictions into account
   - More complex but potentially more adaptive to market conditions
   - Enabled with the `--use-ppo` flag

### Backtesting

The backtesting functionality is implemented using the backtesting.py library. It provides:
- Comprehensive performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- Visualization of trading signals and equity curves
- Comparison of multiple strategies (PPO Agent, Buy & Hold, SMA Crossover)
- Trade logs and detailed performance reports

### Performance Evaluation

The system evaluates the performance of the trading strategy (either ensemble-based or PPO-based) by comparing it with a Buy & Hold strategy and other baseline strategies. It plots the portfolio value over time, visualizes the trading actions, and calculates key performance metrics.

## Results

The results are saved in the following files:
- `trading_results.png`: Plot of portfolio value and trading actions
- `agent_backtest_results.png`: Detailed backtest results for the PPO agent
- `strategy_comparison.png`: Comparison of different trading strategies
- `*_metrics.csv`: CSV files with performance metrics
- `*_trades.csv`: CSV files with trade logs
- `lstm_model.pt`, `gru_model.pt`, `cnn_model.pt`, `mlp_model.pt`: Trained predictive models
- `ppo_agent.pt`: Trained PPO agent

## License

This project is licensed under the MIT License - see the LICENSE file for details.
