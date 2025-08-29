"""
Configuration parameters for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, 'timeseries')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
                MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Data parameters
DEFAULT_SYMBOL = 'BTC/USDT'
DEFAULT_TIMEFRAME = '1s'
DEFAULT_DAYS = 1
DEFAULT_SEQ_LENGTH = 60

# Model parameters
LSTM_HIDDEN_SIZE = 128
GRU_HIDDEN_SIZE = 128
CNN_FILTERS = [32, 64, 128]
MLP_HIDDEN_SIZES = [128, 64]
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Trading parameters
THRESHOLD = 0.52
COMMISSION_RATE = 0.001
POSITION_SIZE = 0.1

# File names
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.pt')
GRU_MODEL_PATH = os.path.join(MODELS_DIR, 'gru_model.pt')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.pt')
MLP_MODEL_PATH = os.path.join(MODELS_DIR, 'mlp_model.pt')
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.pt')