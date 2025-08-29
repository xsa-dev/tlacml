# Automated Profitable Trading Setup Guide

## Overview

This guide details how to set up the automated trading system to execute profitable trades every minute. This system uses an ensemble of machine learning models (LSTM, GRU, CNN, and MLP) to predict price movements and make trading decisions.

## Prerequisites

- Python 3.9+ installed
- Required Python packages (listed in `requirements.txt`)
- API credentials for your exchange (Binance is the default)
- Basic understanding of crypto trading

## Files

- `live_trade.py`: The main trading script that handles data fetching, signal generation, and trade execution
- `start_trading.sh`: Shell script to automate the startup of the trading process
- Pre-trained models: `lstm_model.pt`, `gru_model.pt`, `cnn_model.pt`, `mlp_model.pt`

## Setup Instructions

### 1. Install Dependencies

Ensure your virtual environment is set up and activated, then install the required dependencies:

```bash
# Activate virtual environment (if using one)
source .venv/bin/activate

# Install dependencies
pip install ccxt pandas numpy torch
```

### 2. Configure API Keys

Edit the `start_trading.sh` script to include your exchange API keys:

```bash
# Open the file
nano start_trading.sh

# Update these lines with your API credentials
API_KEY="your_api_key_here"
API_SECRET="your_api_secret_here"
```

### 3. Adjust Trading Parameters

The default trading parameters are set for cautious trading. You can adjust them based on your risk tolerance:

- `--threshold`: Probability threshold for trading signals (default: 0.52)
- `--position-size`: Fraction of available funds to use per trade (default: 0.1 = 10%)
- `--interval`: Time between trading cycles in seconds (default: 60 = 1 minute)

Lower threshold values will generate more trading signals but may include less confident predictions.

### 4. Test the System

Before running in production, test the system in dry-run mode:

```bash
python live_trade.py --dry-run --single-cycle
```

This runs a single trading cycle without executing real trades to verify the system works correctly.

### 5. Configure Minute-by-Minute Execution

To execute trades every minute, set up a cron job:

```bash
# Open crontab editor
crontab -e

# Add this line to run the script every minute
* * * * * cd /path/to/F1 && ./start_trading.sh >/dev/null 2>&1
```

Alternatively, you can start the trading script manually:

```bash
./start_trading.sh
```

This will run the script in the background with continuous 1-minute intervals.

### 6. Monitor and Analyze

The system generates detailed logs in the `logs/` directory. Monitor these logs to track performance:

```bash
# View the latest log file
tail -f logs/trading_*.log
```

Trading results are also saved to `trade_history.csv` for later analysis.

## Advanced Configuration

### Adjusting the Threshold

Based on the backtest results, a threshold of 0.52 provides a good balance between signal frequency and quality. You can adjust this value:

- Lower threshold (e.g., 0.51): More trades, potentially lower profit per trade
- Higher threshold (e.g., 0.55): Fewer trades, potentially higher profit per trade

### Position Sizing

By default, the system uses 10% of available funds per trade. Adjust the `--position-size` parameter for more aggressive or conservative trading.

### Executing Without the Shell Script

You can run the trading system directly:

```bash
python live_trade.py --threshold 0.52 --interval 60 --position-size 0.1
```

Remove the `--dry-run` flag when you're ready for real trading.

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Verify your API keys and ensure they have trading privileges
2. **Model Loading Errors**: Ensure all model files are in the correct location
3. **No Trading Signals**: Try lowering the threshold value (e.g., 0.51 or 0.50)

### Retraining Models

If market conditions change significantly, you may need to retrain the models:

```bash
python main.py
```

This will train new models using the most recent data.

## Disclaimer

This trading system involves financial risk. Always start with small amounts and carefully monitor performance. Past performance does not guarantee future results.