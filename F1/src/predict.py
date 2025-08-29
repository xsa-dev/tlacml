import os
from src.cli import parse_args
from src.timeseries.dataset import prepare_data
from src.modeling.prediction import generate_predictions, save_predictions

def main():
    """Main function for prediction script."""
    args = parse_args()
    
    # Define output directory and filename
    output_dir = "timeseries/processed"
    symbol = args.symbol.replace('/', '_')
    output_file = f"{symbol}_{args.timeframe}_predictions.csv"
    
    # Load data
    print("Loading data...")
    _, _, _, _, _, _, df_normalized = prepare_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        seq_length=args.seq_length,
        load_path=args.timeseries_path if args.load_timeseries and os.path.exists(args.timeseries_path) else None
    )
    
    if df_normalized is None or df_normalized.empty:
        print("Error: No data available for prediction.")
        return
    
    # Generate predictions
    df_signals = generate_predictions(args, df_normalized)
    if df_signals is None:
        return
    
    # Save predictions
    save_predictions(df_signals, output_dir, output_file, generate_plot=True)
    
    print("Prediction complete!")

if __name__ == "__main__":
    main()
