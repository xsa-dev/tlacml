# ===================
# Main Commands
# ===================

# Start downloading and train using ensemble model (default)
train:
	uv run main.py

# Train with PPO agent
train-ppo:
	uv run main.py --use-ppo

# ===================
# Backtest Commands
# ===================

# Backtest using ensemble model predictions only
backtest:
	uv run main.py --backtest-only

# Backtest with more aggressive trading (lower threshold)
backtest-aggressive:
	uv run main.py --backtest-only --threshold 0.52

# Backtest using PPO agent
backtest-ppo:
	uv run main.py --backtest-only --use-ppo

# ===================
# Advanced Commands
# ===================

# Train with custom parameters
train-custom:
	uv run main.py --symbol "BTC/USDT" --timeframe "1m" --days 5 --seq-length 120 --batch-size 128 --max-epochs 100

# Train with PPO and custom parameters
train-ppo-custom:
	uv run main.py --use-ppo --symbol "BTC/USDT" --timeframe "1m" --days 5 --n-episodes 200 --threshold 0.65

# Backtest with custom parameters
backtest-custom:
	uv run main.py --backtest-only --threshold 0.65 --commission 0.001 --cash 20000

# Save results to CSV
backtest-save:
	uv run main.py --backtest-only --save-results --results-path "backtest_results"

# ===================
# Utility Commands
# ===================

# Clean generated files
clean:
	rm -f *.pt *.png
	
# Deep clean (includes data files)
clean-all:
	rm -f *.pt *.png *.csv
	