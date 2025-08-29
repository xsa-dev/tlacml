import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.modeling.models import LSTMPredictor, GRUPredictor, CNNPredictor, MLPPredictor
from src.modeling.ensemble import EnsembleModel
from src.eval.backtest import generate_signals_from_ensemble

def load_models(args, input_size, seq_length):
    """
    Load pre-trained models from disk.
    
    Args:
        args: Command line arguments
        input_size: Input size for the models
        seq_length: Sequence length for the models
        
    Returns:
        List of loaded models
    """
    models = []
    model_paths = {
        'lstm': (args.lstm_models_path, LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=2, output_size=2)),
        'gru': (args.gru_models_path, GRUPredictor(input_size=input_size, hidden_size=64, num_layers=2, output_size=2)),
        'cnn': (args.cnn_models_path, CNNPredictor(input_size=input_size, seq_length=seq_length, output_size=2)),
        'mlp': (args.mlp_models_path, MLPPredictor(input_size=input_size, seq_length=seq_length, hidden_size=128, output_size=2))
    }
    
    for name, (path, model) in model_paths.items():
        if os.path.exists(path):
            print(f"Loading {name.upper()} model from {path}...")
            model.load_state_dict(torch.load(path))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: {name.upper()} model file {path} not found, skipping...")
    
    if not models:
        raise ValueError("No models could be loaded. Please train models first or provide correct model paths.")
    
    return models

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)

def plot_predictions(df_signals, output_path):
    """
    Plot prediction signals and price.
    
    Args:
        df_signals: DataFrame with signals
        output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.plot(df_signals.index, df_signals['close'], label='Price', color='blue')
    
    # Find buy and sell signals
    buy_signals = df_signals[df_signals['Signal'] == 1]
    sell_signals = df_signals[df_signals['Signal'] == -1]
    
    # Plot signals
    if not buy_signals.empty:
        plt.scatter(buy_signals.index, buy_signals['close'], color='green', label='Buy Signal', marker='^', s=100)
    
    if not sell_signals.empty:
        plt.scatter(sell_signals.index, sell_signals['close'], color='red', label='Sell Signal', marker='v', s=100)
    
    plt.title('Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def generate_predictions(args, df_normalized):
    """
    Generate predictions for the given data.
    
    Args:
        args: Command line arguments
        df_normalized: Normalized DataFrame with features
        
    Returns:
        DataFrame with signals
    """
    # Get input size and sequence length
    input_size = df_normalized.shape[1] - 1  # -1 for direction column
    seq_length = args.seq_length
    
    # Load models
    try:
        models = load_models(args, input_size, seq_length)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    # Create ensemble
    print("Creating ensemble model...")
    ensemble_model = EnsembleModel(models, ensemble_method=args.ensemble_method)
    
    # Generate predictions and signals
    print(f"Generating predictions with threshold {args.threshold}...")
    df_signals = generate_signals_from_ensemble(
        ensemble_model=ensemble_model,
        df=df_normalized,
        window_size=seq_length,
        threshold=args.threshold
    )
    
    # Count signals
    buy_signals = (df_signals['Signal'] == 1).sum()
    sell_signals = (df_signals['Signal'] == -1).sum()
    print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    
    return df_signals

def save_predictions(df_signals, output_dir, output_file, generate_plot=True):
    """
    Save predictions to CSV and optionally generate a plot.
    
    Args:
        df_signals: DataFrame with signals
        output_dir: Directory to save the output
        output_file: Name of the output file
        generate_plot: Whether to generate a plot
        
    Returns:
        Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    # Save predictions to CSV
    output_path = os.path.join(output_dir, output_file)
    print(f"Saving predictions to {output_path}...")
    df_signals.to_csv(output_path)
    print(f"Predictions saved to {output_path}")
    
    # Generate plot if requested
    if generate_plot:
        plot_path = os.path.join(output_dir, f"{os.path.splitext(output_file)[0]}_plot.png")
        plot_predictions(df_signals, plot_path)
    
    return output_path
