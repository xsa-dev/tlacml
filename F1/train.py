import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from clearml import Task
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data import prepare_data, create_sequences

from models import (
    LSTMPredictor, GRUPredictor, CNNPredictor, MLPPredictor, create_dataloaders
)
from ensemble import EnsembleModel
from env import CryptoTradingEnv
from ppo import PPOAgent, train_ppo
from backtest import (
    run_backtest, generate_signals_from_agent, generate_signals_from_ensemble,
    compare_strategies, create_buy_hold_signals, create_sma_crossover_signals
)

def train_predictive_models(X_train, y_train, X_val, y_val, X_test, y_test, input_size, seq_length, batch_size=64, max_epochs=50):
    """
    Train the predictive models.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Test features
        y_test: Test targets
        input_size: Input size for the models
        seq_length: Sequence length for the models
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs for training

    Returns:
        Trained models
    """
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )

    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    # Set up logger
    logger = TensorBoardLogger('lightning_logs', name='predictive_models')

    # Initialize models
    lstm_model = LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=2, output_size=2)
    gru_model = GRUPredictor(input_size=input_size, hidden_size=64, num_layers=2, output_size=2)
    cnn_model = CNNPredictor(input_size=input_size, seq_length=seq_length, output_size=2)
    mlp_model = MLPPredictor(input_size=input_size, seq_length=seq_length, hidden_size=128, output_size=2)

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping],
        logger=logger,
        accelerator='auto',  # Use MPS on M1 Macs if available
        devices=1,
        log_every_n_steps=10,
    )

    # Train models
    print("Training LSTM model...")
    trainer.fit(lstm_model, train_loader, val_loader)

    print("Training GRU model...")
    trainer.fit(gru_model, train_loader, val_loader)

    print("Training CNN model...")
    trainer.fit(cnn_model, train_loader, val_loader)

    print("Training MLP model...")
    trainer.fit(mlp_model, train_loader, val_loader)

    # Evaluate models on test set
    lstm_results = trainer.test(lstm_model, test_loader)
    gru_results = trainer.test(gru_model, test_loader)
    cnn_results = trainer.test(cnn_model, test_loader)
    mlp_results = trainer.test(mlp_model, test_loader)

    print(f"LSTM Test Results: {lstm_results}")
    print(f"GRU Test Results: {gru_results}")
    print(f"CNN Test Results: {cnn_results}")
    print(f"MLP Test Results: {mlp_results}")

    # Move models to evaluation mode
    lstm_model.eval()
    gru_model.eval()
    cnn_model.eval()
    mlp_model.eval()

    return lstm_model, gru_model, cnn_model, mlp_model

def create_ensemble(models, val_loader, ensemble_method='average'):
    """
    Create an ensemble model from the trained models.

    Args:
        models: List of trained models
        val_loader: Validation dataloader
        ensemble_method: Method for combining predictions

    Returns:
        Ensemble model
    """
    # Create ensemble
    ensemble = EnsembleModel(models, ensemble_method=ensemble_method)

    # Evaluate ensemble on validation set
    ensemble_metrics = ensemble.evaluate(val_loader)
    print(f"Ensemble Validation Accuracy: {ensemble_metrics['accuracy']:.4f}")

    # Optimize ensemble weights
    print("Optimizing ensemble weights...")
    ensemble.optimize_weights(val_loader, num_iterations=100, learning_rate=0.01)

    # Re-evaluate ensemble with optimized weights
    ensemble_metrics = ensemble.evaluate(val_loader)
    print(f"Ensemble Validation Accuracy (after optimization): {ensemble_metrics['accuracy']:.4f}")

    return ensemble

def train_ppo_agent(df_normalized, ensemble_model, window_size=60, n_episodes=100, max_steps=1000):
    """
    Train a PPO agent for trading.

    Args:
        df_normalized: Normalized dataframe
        ensemble_model: Ensemble model for price direction prediction
        window_size: Size of the observation window
        n_episodes: Number of episodes for training
        max_steps: Maximum number of steps per episode

    Returns:
        Trained PPO agent and episode rewards
    """
    # Create environment
    env = CryptoTradingEnv(
        df=df_normalized,
        ensemble_model=ensemble_model,
        window_size=window_size,
        initial_balance=10000.0,
        transaction_fee=0.001,
        reward_scaling=1e-4,
        use_ensemble=True
    )

    # Get observation dimension
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create PPO agent
    agent = PPOAgent(
        input_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01
    )

    # Train agent
    print(f"Training PPO agent for {n_episodes} episodes...")
    episode_rewards = train_ppo(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        max_steps=max_steps,
        update_interval=20,
        log_interval=10,
        save_path='ppo_model.pt'
    )

    return agent, episode_rewards, env

def evaluate_agent(env, agent, max_steps=1000):
    """
    Evaluate the trained agent.

    Args:
        env: Trading environment
        agent: Trained PPO agent
        max_steps: Maximum number of steps

    Returns:
        Portfolio history
    """
    # Reset environment
    observation, _ = env.reset()
    done = False
    truncated = False
    step_count = 0

    # Run episode
    while not (done or truncated) and step_count < max_steps:
        action, _, _ = agent.choose_action(observation)
        observation, _, done, truncated, _ = env.step(action)
        step_count += 1

    # Get portfolio history
    portfolio_history = env.get_portfolio_history()

    # Calculate buy and hold value
    buy_and_hold_value = env.get_buy_and_hold_portfolio_value()

    # Calculate final portfolio value
    final_portfolio_value = portfolio_history['portfolio_value'].iloc[-1]

    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Buy and Hold Value: {buy_and_hold_value:.2f}")
    print(f"Relative Performance: {(final_portfolio_value / buy_and_hold_value - 1) * 100:.2f}%")

    return portfolio_history

def plot_results(portfolio_history, df_normalized, window_size):
    """
    Plot the results.

    Args:
        portfolio_history: Portfolio history from the agent
        df_normalized: Normalized dataframe
        window_size: Size of the observation window
    """
    # Create figure
    plt.figure(figsize=(14, 10))

    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_history['step'], portfolio_history['portfolio_value'], label='Portfolio Value')

    # Calculate buy and hold value
    initial_price = df_normalized.iloc[window_size]['close']
    initial_crypto = 10000 / initial_price
    buy_hold_values = []

    for step in portfolio_history['step']:
        price = df_normalized.iloc[step]['close']
        buy_hold_values.append(initial_crypto * price)

    plt.plot(portfolio_history['step'], buy_hold_values, label='Buy & Hold')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value vs Buy & Hold')
    plt.legend()
    plt.grid(True)

    # Plot actions
    plt.subplot(2, 1, 2)

    # Extract actions
    buy_steps = portfolio_history[portfolio_history['action'] == 1]['step']
    sell_steps = portfolio_history[portfolio_history['action'] == 2]['step']

    # Plot price
    plt.plot(portfolio_history['step'], portfolio_history['price'], label='Price', color='blue')

    # Plot buy and sell points
    for step in buy_steps:
        idx = portfolio_history['step'] == step
        plt.scatter(step, portfolio_history.loc[idx, 'price'].values[0], color='green', marker='^', s=100)

    for step in sell_steps:
        idx = portfolio_history['step'] == step
        plt.scatter(step, portfolio_history.loc[idx, 'price'].values[0], color='red', marker='v', s=100)

    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.title('Trading Actions (Green: Buy, Red: Sell)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('trading_results.png')
    plt.show()

def main(args):
    """
    Main function to run the training pipeline.
    """
    # Initialize ClearML task
    task = Task.init(project_name='CryptoTrading', task_name='PPO_BTC_USDT')

    # Prepare data
    print("Preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, df_normalized = prepare_data(
        symbol='BTC/USDT',
        timeframe='1s',
        days=1,
        seq_length=1000,
        save_path='btc_usdt_1s.csv',
        load_path='btc_usdt_1s.csv' if os.path.exists('btc_usdt_1s.csv') else None
    )

    # Get input size
    input_size = X_train.shape[2]
    seq_length = X_train.shape[1]

    # Train predictive models
    print("Training predictive models...")
    lstm_model, gru_model, cnn_model, mlp_model = train_predictive_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_size=input_size,
        seq_length=seq_length,
        batch_size=64,
        max_epochs=50
    )

    # Create dataloaders for ensemble
    _, val_loader, _ = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64
    )

    # Create ensemble
    print("Creating ensemble model...")
    ensemble_model = create_ensemble(
        models=[lstm_model, gru_model, cnn_model, mlp_model],
        val_loader=val_loader,
        ensemble_method='weighted_average'
    )

    # Train PPO agent
    print("Training PPO agent...")
    agent, episode_rewards, env = train_ppo_agent(
        df_normalized=df_normalized,
        ensemble_model=ensemble_model,
        window_size=seq_length,
        n_episodes=500,
        max_steps=1000
    )

    # Evaluate agent
    print("Evaluating agent...")
    portfolio_history = evaluate_agent(env, agent, max_steps=1000)

    # Plot results
    print("Plotting results...")
    plot_results(portfolio_history, df_normalized, seq_length)

    # Run backtest
    print("Running backtest...")
    run_agent_backtest(env, agent, df_normalized, seq_length)

    # Compare with other strategies
    print("Comparing strategies...")
    compare_agent_with_strategies(env, agent, df_normalized, seq_length)

    # Save models
    print("Saving models...")
    torch.save(lstm_model.state_dict(), 'lstm_model.pt')
    torch.save(gru_model.state_dict(), 'gru_model.pt')
    torch.save(cnn_model.state_dict(), 'cnn_model.pt')
    torch.save(mlp_model.state_dict(), 'mlp_model.pt')
    torch.save(agent.state_dict(), 'ppo_agent.pt')

    task.close()
    print("Done!")


def run_agent_backtest(env, agent, df, window_size, cash=10000, commission=0.002):
    """
    Run a backtest for the trained PPO agent.

    Args:
        env: Trading environment
        agent: Trained PPO agent
        df: DataFrame with OHLCV data
        window_size: Size of the observation window
        cash: Initial cash amount
        commission: Commission rate for trades
    """
    # Generate signals from agent
    df_signals = generate_signals_from_agent(env, agent, df, window_size)

    # Prepare data for backtesting
    # Convert column names to match backtesting.py requirements
    df_backtest = df_signals.copy()

    # Run backtest
    metrics = run_backtest(
        df=df_backtest,
        cash=cash,
        commission=commission,
        plot=True,
        save_path='agent_backtest_results.png'
    )

    # Print metrics
    print("\nBacktest Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['trades']}")
    print(f"Final Equity: ${metrics['final_equity']:.2f}")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2%}")
    print(f"SQN: {metrics['sqn']:.2f}")

    return metrics


def compare_agent_with_strategies(env, agent, df, window_size, cash=10000, commission=0.002):
    """
    Compare the PPO agent with other trading strategies.

    Args:
        env: Trading environment
        agent: Trained PPO agent
        df: DataFrame with OHLCV data
        window_size: Size of the observation window
        cash: Initial cash amount
        commission: Commission rate for trades
    """
    # Generate signals from agent
    df_signals = generate_signals_from_agent(env, agent, df, window_size)
    agent_signals = df_signals['Signal'].values

    # Create signals for other strategies
    buy_hold_signals = create_buy_hold_signals(df)
    sma_crossover_signals = create_sma_crossover_signals(df, short_window=20, long_window=50)

    # Create strategies dictionary
    strategies = {
        'PPO Agent': agent_signals,
        'Buy & Hold': buy_hold_signals,
        'SMA Crossover': sma_crossover_signals
    }

    # Compare strategies
    results = compare_strategies(
        df=df_signals,
        strategies=strategies,
        cash=cash,
        commission=commission,
        save_path='strategy_comparison.png'
    )

    # Print comparison results
    print("\nStrategy Comparison:")
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Number of Trades: {metrics['trades']}")

    return results


def run_ensemble_backtest(ensemble_model, df, window_size, threshold=0.6, cash=10000, commission=0.002):
    """
    Run a backtest for the ensemble model.

    Args:
        ensemble_model: Trained ensemble model
        df: DataFrame with OHLCV data and features
        window_size: Size of the observation window
        threshold: Probability threshold for buy/sell decisions
        cash: Initial cash amount
        commission: Commission rate for trades
    """
    # Generate signals from ensemble
    df_signals = generate_signals_from_ensemble(ensemble_model, df, window_size, threshold)

    # Run backtest
    metrics = run_backtest(
        df=df_signals,
        cash=cash,
        commission=commission,
        plot=True,
        save_path='ensemble_backtest_results.png'
    )

    # Print metrics
    print("\nEnsemble Backtest Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['trades']}")
    print(f"Final Equity: ${metrics['final_equity']:.2f}")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2%}")
    print(f"SQN: {metrics['sqn']:.2f}")

    return metrics

if __name__ == "__main__":
    main()
