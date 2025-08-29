# ===================
# Main Commands
# ===================

# Start downloading and train using ensemble model (default)
train:
	PYTHONPATH=$(PWD) uv run src/main.py
	PYTHONPATH=$(PWD) uv run src/predict.py

# Train with PPO agent
train-ppo:
	PYTHONPATH=$(PWD) uv run src/main.py --use-ppo
	PYTHONPATH=$(PWD) uv run src/predict.py

# Quick train for testing (1 epoch)
train-quick:
	mkdir -p timeseries/raw timeseries/processed
	PYTHONPATH=$(PWD) uv run src/main.py --max-epochs 1 --n-episodes 1 --max-steps 1 --batch-size 32 --days 1 --timeframe 1s --seq-length 30 --timeseries-path timeseries/raw/btc_usdt_1s.csv --load-timeseries --save-timeseries --lstm-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/lstm_model.pt --gru-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/gru_model.pt --cnn-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/cnn_model.pt --mlp-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/mlp_model.pt
	PYTHONPATH=$(PWD) uv run src/predict.py --timeseries-path timeseries/raw/btc_usdt_1s.csv --load-timeseries --lstm-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/lstm_model.pt --gru-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/gru_model.pt --cnn-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/cnn_model.pt --mlp-models-path /Users/alxy/Desktop/y/1PROJ/rbi-moon/tlacml/F1/models/mlp_model.pt

# ===================
# Backtest Commands
# ===================

# Backtest using ensemble model predictions only
backtest:
	PYTHONPATH=$(PWD) uv run src/main.py --backtest-only

# Backtest with more aggressive trading (lower threshold)
backtest-aggressive:
	PYTHONPATH=$(PWD) uv run src/main.py --backtest-only --threshold 0.52

# Backtest using PPO agent
backtest-ppo:
	PYTHONPATH=$(PWD) uv run src/main.py --backtest-only --use-ppo

# ===================
# Advanced Commands
# ===================

# Train with custom parameters
train-custom:
	PYTHONPATH=$(PWD) uv run src/main.py --symbol "BTC/USDT" --timeframe "1m" --days 5 --seq-length 120 --batch-size 128 --max-epochs 100
	PYTHONPATH=$(PWD) uv run src/predict.py

# Train with PPO and custom parameters
train-ppo-custom:
	PYTHONPATH=$(PWD) uv run src/main.py --use-ppo --symbol "BTC/USDT" --timeframe "1m" --days 5 --n-episodes 200 --threshold 0.65
	PYTHONPATH=$(PWD) uv run src/predict.py

# Backtest with custom parameters
backtest-custom:
	PYTHONPATH=$(PWD) uv run src/main.py --backtest-only --threshold 0.65 --commission 0.001 --cash 20000

# Save results to CSV
backtest-save:
	PYTHONPATH=$(PWD) uv run src/main.py --backtest-only --save-results --results-path "backtest_results"

# ===================
# Prediction Commands
# ===================

# Run prediction only
predict:
	PYTHONPATH=$(PWD) uv run src/predict.py --timeseries-path timeseries/raw/btc_usdt_1s.csv --load-timeseries --lstm-models-path models/lstm_model.pt --gru-models-path models/gru_model.pt --cnn-models-path models/cnn_model.pt --mlp-models-path models/mlp_model.pt

# ===================
# Utility Commands
# ===================

# Clean generated files
clean:
	rm -f *.pt *.png
	
# Deep clean (includes timeseries files)
clean-all:
	rm -f *.pt *.png *.csv

# Create necessary directories
create-dirs:
	mkdir -p timeseries/raw timeseries/processed