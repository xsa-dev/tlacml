#!/bin/bash

# start_trading.sh - Script to setup and run the live trading system
# This script can be used with cron to ensure trading runs continuously

# Set up environment variables (modify these values with your actual API keys)
API_KEY=""
API_SECRET=""

# Define log file
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/trading_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if there's already a trading process running
PID_FILE="trading.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "Trading process is already running with PID $PID"
        exit 1
    else
        echo "Stale PID file found, removing..."
        rm "$PID_FILE"
    fi
fi

# Activate virtual environment if one exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Start trading with optimized parameters (adjust as needed)
echo "Starting live trading at $(date)"
nohup python live_trade.py \
    --threshold 0.52 \
    --api-key "$API_KEY" \
    --api-secret "$API_SECRET" \
    --interval 60 \
    --position-size 0.1 \
    --dry-run > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$PID_FILE"
echo "Trading started with PID $!"
echo "Logs being saved to $LOG_FILE"

# To schedule with cron, add a line like this to your crontab:
# * * * * * cd /path/to/F1 && ./start_trading.sh >/dev/null 2>&1