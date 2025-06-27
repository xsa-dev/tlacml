#!/bin/bash

# check_trading.sh - Script to monitor the status of the trading system

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Define the PID file and log directory
PID_FILE="trading.pid"
LOG_DIR="logs"
LATEST_LOG=$(ls -t $LOG_DIR/trading_*.log 2>/dev/null | head -1)

echo -e "${YELLOW}===== Trading System Status Check =====${NC}"
echo "Time: $(date)"
echo ""

# Check if trading process is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo -e "${GREEN}✓ Trading process is RUNNING with PID $PID${NC}"
        
        # Get runtime
        if [ -d "/proc/$PID" ]; then
            start_time=$(stat -c %Y "/proc/$PID")
            current_time=$(date +%s)
            runtime=$((current_time - start_time))
            
            days=$((runtime / 86400))
            hours=$(( (runtime % 86400) / 3600 ))
            minutes=$(( (runtime % 3600) / 60 ))
            seconds=$((runtime % 60))
            
            echo "  Runtime: ${days}d ${hours}h ${minutes}m ${seconds}s"
        fi
    else
        echo -e "${RED}✗ Trading process is NOT RUNNING (stale PID file)${NC}"
        echo "  Last PID was: $PID"
    fi
else
    echo -e "${RED}✗ Trading process is NOT RUNNING (no PID file)${NC}"
fi

echo ""

# Check for recent logs
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo -e "${YELLOW}Recent log activity (last 10 lines):${NC}"
    echo "-----------------------------------------"
    tail -n 10 "$LATEST_LOG"
    echo "-----------------------------------------"
    
    # Get log file stats
    log_size=$(du -h "$LATEST_LOG" | cut -f1)
    log_time=$(stat -c %y "$LATEST_LOG")
    echo "Latest log: $LATEST_LOG ($log_size, last modified: $log_time)"
    
    # Check for recent activity
    last_modified=$(stat -c %Y "$LATEST_LOG")
    current_time=$(date +%s)
    time_diff=$((current_time - last_modified))
    
    if [ $time_diff -gt 300 ]; then  # 5 minutes
        echo -e "${RED}WARNING: Log hasn't been updated in $((time_diff / 60)) minutes${NC}"
    else
        echo -e "${GREEN}Log was updated recently ($((time_diff / 60)) minutes ago)${NC}"
    fi
else
    echo -e "${RED}No log files found in $LOG_DIR directory${NC}"
fi

echo ""

# Check for recent trades
TRADE_HISTORY="trade_history.csv"
if [ -f "$TRADE_HISTORY" ]; then
    num_trades=$(wc -l < "$TRADE_HISTORY")
    num_trades=$((num_trades - 1))  # Subtract header line
    
    echo -e "${YELLOW}Trading Statistics:${NC}"
    echo "Total trades recorded: $num_trades"
    
    if [ $num_trades -gt 0 ]; then
        # Get recent trade info
        echo "Most recent trades:"
        tail -n 3 "$TRADE_HISTORY" | column -t -s,
    fi
else
    echo -e "${YELLOW}No trade history found ($TRADE_HISTORY)${NC}"
fi

echo ""
echo -e "${YELLOW}===== System Information =====${NC}"
free -h | grep "Mem:"
uptime

echo ""
echo -e "${YELLOW}To view full logs:${NC} tail -f $LATEST_LOG"
echo -e "${YELLOW}To restart trading:${NC} ./start_trading.sh"