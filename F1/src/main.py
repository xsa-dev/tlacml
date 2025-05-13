import os
import argparse
from train import main as train_main
from train import run_agent_backtest, compare_agent_with_strategies, run_ensemble_backtest
import torch


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Crypto Trading with DRL (PPO) and Ensemble Models')

    # Data arguments
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1s',
                        help='Data timeframe (default: 1s)')
    parser.add_argument('--days', type=int, default=1,
                        help='Number of days of data to fetch (default: 1)')
    parser.add_argument('--seq-length', type=int, default=60,
                        help='Sequence length for time series models (default: 60)')
    parser.add_argument('--data-path', type=str, default='btc_usdt_1s.csv',
                        help='Path to save/load data (default: btc_usdt_1s.csv)')
    parser.add_argument('--load-data', action='store_true',
                        help='Load data from file if available (default: True)')
    parser.add_argument('--save-data', action='store_true',
                        help='Save data to file (default: True)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum number of epochs for training predictive models (default: 50)')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Number of episodes for training PPO agent (default: 100)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum number of steps per episode (default: 1000)')
    parser.add_argument('--ensemble-method', type=str, default='weighted_average',
                        choices=['average', 'weighted_average', 'voting'],
                        help='Method for combining predictions in the ensemble (default: weighted_average)')
    parser.add_argument('--use-ppo', action='store_true',
                        help='Use PPO reinforcement learning agent for trading (default: False, use ensemble models predictions directly for trading signals)')

    # ClearML arguments
    parser.add_argument('--use-clearml', action='store_true',
                        help='Use ClearML for experiment tracking (default: True)')
    parser.add_argument('--project-name', type=str, default='CryptoTrading',
                        help='ClearML project name (default: CryptoTrading)')
    parser.add_argument('--task-name', type=str, default='PPO_BTC_USDT',
                        help='ClearML task name (default: PPO_BTC_USDT)')

    # Backtesting arguments
    parser.add_argument('--backtest-only', action='store_true',
                        help='Run only backtesting on pre-trained models (default: False)')
    parser.add_argument('--cash', type=float, default=10000,
                        help='Initial cash amount for backtesting (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.002,
                        help='Commission rate for trades (default: 0.002)')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Probability threshold for ensemble models signals (default: 0.55)')
    parser.add_argument('--agent-models-path', type=str, default='ppo_agent.pt',
                        help='Path to pre-trained PPO agent models (default: ppo_agent.pt)')
    parser.add_argument('--lstm-models-path', type=str, default='lstm_model.pt',
                        help='Path to pre-trained LSTM models (default: lstm_model.pt)')
    parser.add_argument('--gru-models-path', type=str, default='gru_model.pt',
                        help='Path to pre-trained GRU models (default: gru_model.pt)')
    parser.add_argument('--cnn-models-path', type=str, default='cnn_model.pt',
                        help='Path to pre-trained CNN models (default: cnn_model.pt)')
    parser.add_argument('--mlp-models-path', type=str, default='mlp_model.pt',
                        help='Path to pre-trained MLP models (default: mlp_model.pt)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save backtest results to CSV (default: False)')
    parser.add_argument('--results-path', type=str, default='backtest_results',
                        help='Path to save backtest results (default: backtest_results)')

    return parser.parse_args()

def run_backtest_only(args):
    """
    Run backtesting on pre-trained models. By default, uses the ensemble models's predictions
    directly for trading signals, unless --use-ppo is specified.

    Args:
        args: Command-line arguments
    """
    from data import prepare_data
    from models import LSTMPredictor, GRUPredictor, CNNPredictor, MLPPredictor
    from src.ensemble import EnsembleModel
    from src.env import CryptoTradingEnv
    from ppo import PPOAgent

    # Load data
    print("Loading data...")
    _, _, _, _, _, _, df_normalized = prepare_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        seq_length=args.seq_length,
        load_path=args.data_path
    )

    # Load ensemble models
    model_paths = {
        'lstm': args.lstm_model_path,
        'gru': args.gru_model_path,
        'cnn': args.cnn_model_path,
        'mlp': args.mlp_model_path
    }

    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"Loading {name.upper()} models from {path}...")

            # Create models
            if name == 'lstm':
                model = LSTMPredictor(input_size=df_normalized.shape[1]-1)  # -1 for direction column
            elif name == 'gru':
                model = GRUPredictor(input_size=df_normalized.shape[1]-1)
            elif name == 'cnn':
                model = CNNPredictor(input_size=df_normalized.shape[1]-1, seq_length=args.seq_length)
            elif name == 'mlp':
                model = MLPPredictor(input_size=df_normalized.shape[1]-1, seq_length=args.seq_length)

            # Load state dict
            model.load_state_dict(torch.load(path))
            model.eval()

            models[name] = model
    
    # Create an ensemble if models are loaded
    ensemble_model = None
    if len(models) > 0:
        print("Creating ensemble models...")
        ensemble_model = EnsembleModel(
            models=list(models.values()),
            ensemble_method=args.ensemble_method
        )

        # Run ensemble backtest
        print("Running ensemble backtest...")
        run_ensemble_backtest(
            ensemble_model=ensemble_model,
            df=df_normalized,
            window_size=args.seq_length,
            threshold=args.threshold,
            cash=args.cash,
            commission=args.commission
        )
    else:
        print("No models were loaded, cannot create ensemble.")
        return

    # Create environment with ensemble models
    env = CryptoTradingEnv(
        df=df_normalized,
        window_size=args.seq_length,
        initial_balance=args.cash,
        transaction_fee=args.commission,
        use_ensemble=True,
        ensemble_model=ensemble_model
    )

    # Only load PPO agent if use_ppo is True
    agent = None
    if args.use_ppo and os.path.exists(args.agent_model_path):
        print(f"Loading PPO agent from {args.agent_model_path}...")

        # Get observation dimension
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Create agent
        agent = PPOAgent(
            input_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=256
        )

        # Load state dict
        agent.load_state_dict(torch.load(args.agent_model_path))
        agent.eval()

        # Run agent backtest
        print("Running PPO agent backtest...")
        run_agent_backtest(
            env=env,
            agent=agent,
            df=df_normalized,
            window_size=args.seq_length,
            cash=args.cash,
            commission=args.commission
        )
        
        # Compare strategies if agent is loaded
        print("Comparing strategies...")
        compare_agent_with_strategies(
            env=env,
            agent=agent,
            df=df_normalized,
            window_size=args.seq_length,
            cash=args.cash,
            commission=args.commission
        )
    elif args.use_ppo:
        print(f"PPO agent models not found at {args.agent_model_path}")

def main():
    """
    Main function. Provides two main functionalities:
    1. Training predictive models and optionally a PPO agent
    2. Backtesting with pre-trained models, using either ensemble predictions directly
       or a PPO agent (if --use-ppo is specified)
    """
    args = parse_args()

    if args.backtest_only:
        run_backtest_only(args)
    else:
        # Run a training pipeline
        train_main(args)

if __name__ == "__main__":
    main()
