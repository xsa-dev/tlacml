"""
Prediction functions for machine learning models.
"""

import os

import torch

from src.config import (LSTM_MODEL_PATH, GRU_MODEL_PATH,
                        CNN_MODEL_PATH, MLP_MODEL_PATH, DEFAULT_SEQ_LENGTH,
                        THRESHOLD)
from src.modeling.ensemble import EnsembleModel
from src.timeseries.features import create_sequences
from src.modeling.models import LSTMPredictor, GRUPredictor, CNNPredictor, MLPPredictor


def load_model(model_class, model_path, input_size=None, seq_length=None, **kwargs):
    """
    Load a PyTorch model.
    
    Args:
        model_class: Class of the model to load
        model_path: Path to the saved model
        input_size: Number of input features
        seq_length: Length of input sequences
        **kwargs: Additional arguments for the model
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None
        
        # Create model instance
        if model_class == LSTMPredictor or model_class == GRUPredictor:
            model = model_class(input_size=input_size, **kwargs)
        else:  # CNN or MLP
            model = model_class(input_size=input_size, seq_length=seq_length, **kwargs)
        
        # Load state dict
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_ensemble_model(input_size, seq_length, model_paths=None):
    """
    Load all models and create an ensemble.
    
    Args:
        input_size: Number of input features
        seq_length: Length of input sequences
        model_paths: Dictionary with model paths
        
    Returns:
        Ensemble model
    """
    if model_paths is None:
        model_paths = {
            'lstm': LSTM_MODEL_PATH,
            'gru': GRU_MODEL_PATH,
            'cnn': CNN_MODEL_PATH,
            'mlp': MLP_MODEL_PATH
        }
    
    models = {}
    
    # Load LSTM model
    if 'lstm' in model_paths:
        models['lstm'] = load_model(LSTMPredictor, model_paths['lstm'], input_size=input_size)
    
    # Load GRU model
    if 'gru' in model_paths:
        models['gru'] = load_model(GRUPredictor, model_paths['gru'], input_size=input_size)
    
    # Load CNN model
    if 'cnn' in model_paths:
        models['cnn'] = load_model(CNNPredictor, model_paths['cnn'], input_size=input_size, seq_length=seq_length)
    
    # Load MLP model
    if 'mlp' in model_paths:
        models['mlp'] = load_model(MLPPredictor, model_paths['mlp'], input_size=input_size, seq_length=seq_length)
    
    # Filter out None values
    models = {k: v for k, v in models.items() if v is not None}
    
    if not models:
        raise ValueError("No models were loaded successfully")
    
    # Create ensemble model
    ensemble = EnsembleModel(list(models.values()), weights=None)
    
    return ensemble

def predict_single_sequence(model, sequence, threshold=THRESHOLD):
    """
    Make a prediction for a single sequence.
    
    Args:
        model: Trained model
        sequence: Input sequence (numpy array)
        threshold: Threshold for binary classification
        
    Returns:
        Prediction (0 or 1) and probability
    """
    # Convert to tensor
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
    
    # Prediction
    with torch.no_grad():
        output = model(sequence_tensor)
        probability = output.item()
        prediction = 1 if probability > threshold else 0
    
    return prediction, probability

def generate_predictions(model, df, seq_length=DEFAULT_SEQ_LENGTH, threshold=THRESHOLD):
    """
    Generate predictions for a dataframe.
    
    Args:
        model: Trained model
        df: DataFrame with features
        seq_length: Length of input sequences
        threshold: Threshold for binary classification
        
    Returns:
        DataFrame with predictions
    """
    # Create sequences
    X, y = create_sequences(df, seq_length)
    
    # Make predictions
    predictions = []
    probabilities = []
    
    for i in range(len(X)):
        pred, prob = predict_single_sequence(model, X[i], threshold)
        predictions.append(pred)
        probabilities.append(prob)
    
    # Create results dataframe
    results_df = df.iloc[seq_length:].copy()
    results_df['prediction'] = predictions
    results_df['probability'] = probabilities
    results_df['true_direction'] = y
    results_df['correct'] = results_df['prediction'] == results_df['true_direction']
    
    return results_df

def generate_signals(model, df, seq_length=DEFAULT_SEQ_LENGTH, threshold=THRESHOLD):
    """
    Generate trading signals.
    
    Args:
        model: Trained model
        df: DataFrame with features
        seq_length: Length of input sequences
        threshold: Threshold for binary classification
        
    Returns:
        DataFrame with signals (-1 for sell, 0 for hold, 1 for buy)
    """
    # Generate predictions
    results_df = generate_predictions(model, df, seq_length, threshold)
    
    # Generate signals
    results_df['signal'] = 0  # Default: hold
    
    # Buy signal: prediction is 1 (up) and probability > threshold
    buy_condition = (results_df['prediction'] == 1) & (results_df['probability'] > threshold)
    results_df.loc[buy_condition, 'signal'] = 1
    
    # Sell signal: prediction is 0 (down) and probability > (1 - threshold)
    sell_condition = (results_df['prediction'] == 0) & (results_df['probability'] < (1 - threshold))
    results_df.loc[sell_condition, 'signal'] = -1
    
    return results_df