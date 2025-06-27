import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any, Optional, List

class CryptoTradingEnv(gym.Env):
    """
    A cryptocurrency trading environment for OpenAI gym.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 ensemble_model=None, 
                 window_size: int = 60,
                 initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001,
                 reward_scaling: float = 1e-4,
                 use_ensemble: bool = True):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV timeseries and features
            ensemble_model: Ensemble models for price direction prediction
            window_size: Size of the observation window
            initial_balance: Initial account balance
            transaction_fee: Fee for each transaction as a fraction of the transaction amount
            reward_scaling: Scaling factor for rewards
            use_ensemble: Whether to use ensemble predictions in the observation space
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index()
        self.ensemble_model = ensemble_model
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.use_ensemble = use_ensemble
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Calculate observation space dimensions
        # Features from the dataframe (excluding 'timestamp' and 'direction')
        self.feature_cols = [col for col in self.df.columns if col not in ['timestamp', 'direction']]
        self.num_features = len(self.feature_cols)
        
        # If using ensemble, add ensemble prediction to observation
        if self.use_ensemble and self.ensemble_model is not None:
            # Ensemble prediction (probability of price going up)
            self.obs_dim = self.num_features + 1
        else:
            self.obs_dim = self.num_features
        
        # Add account features to observation (balance, crypto_held, portfolio_value)
        self.obs_dim += 3
        
        # Observation space: features + account state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset account state
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.current_step = self.window_size
        self.done = False
        self.history = []
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0 = Hold, 1 = Buy, 2 = Sell)
            
        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Get current price
        current_price = self.df.loc[self.current_step, 'close']
        
        # Get previous portfolio value
        prev_portfolio_value = self.balance + self.crypto_held * current_price
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                # Calculate transaction fee
                fee = self.balance * self.transaction_fee
                # Buy crypto with remaining balance
                self.crypto_held += (self.balance - fee) / current_price
                self.balance = 0
        
        elif action == 2:  # Sell
            if self.crypto_held > 0:
                # Calculate transaction fee
                fee = self.crypto_held * current_price * self.transaction_fee
                # Sell all crypto
                self.balance += self.crypto_held * current_price - fee
                self.crypto_held = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        # Calculate reward
        new_price = self.df.loc[self.current_step, 'close']
        new_portfolio_value = self.balance + self.crypto_held * new_price
        
        # Reward is the change in portfolio value
        reward = (new_portfolio_value - prev_portfolio_value) * self.reward_scaling
        
        # Store history for rendering
        self.history.append({
            'step': self.current_step,
            'price': new_price,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': new_portfolio_value,
            'action': action,
            'reward': reward
        })
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Observation array
        """
        # Get market features
        features = self.df.loc[self.current_step, self.feature_cols].values
        
        # Calculate account state features
        current_price = self.df.loc[self.current_step, 'close']
        balance_feature = self.balance / self.initial_balance  # Normalized balance
        crypto_held_feature = self.crypto_held * current_price / self.initial_balance  # Normalized crypto value
        portfolio_value = (self.balance + self.crypto_held * current_price) / self.initial_balance  # Normalized portfolio value
        
        # Combine features
        obs = list(features)
        
        # Add ensemble prediction if available
        if self.use_ensemble and self.ensemble_model is not None:
            # Get timeseries for the current window
            window_data = self.df.loc[self.current_step - self.window_size:self.current_step - 1, self.feature_cols].values
            window_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Get ensemble prediction
            with torch.no_grad():
                ensemble_pred = self.ensemble_model.predict(window_tensor)[0, 1].item()  # Probability of price going up
            
            obs.append(ensemble_pred)
        
        # Add account state features
        obs.extend([balance_feature, crypto_held_feature, portfolio_value])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information.
        
        Returns:
            Dictionary with additional information
        """
        current_price = self.df.loc[self.current_step, 'close']
        portfolio_value = self.balance + self.crypto_held * current_price
        
        return {
            'step': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': portfolio_value
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            if len(self.history) > 0:
                last_record = self.history[-1]
                print(f"Step: {last_record['step']}, "
                      f"Price: {last_record['price']:.2f}, "
                      f"Balance: {last_record['balance']:.2f}, "
                      f"Crypto: {last_record['crypto_held']:.6f}, "
                      f"Portfolio: {last_record['portfolio_value']:.2f}, "
                      f"Action: {['Hold', 'Buy', 'Sell'][last_record['action']]}, "
                      f"Reward: {last_record['reward']:.6f}")
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio history as a DataFrame.
        
        Returns:
            DataFrame with portfolio history
        """
        if len(self.history) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.history)
    
    def get_buy_and_hold_portfolio_value(self) -> float:
        """
        Calculate the portfolio value if using a buy and hold strategy.
        
        Returns:
            Final portfolio value
        """
        initial_price = self.df.loc[self.window_size, 'close']
        final_price = self.df.loc[len(self.df) - 1, 'close']
        
        # Calculate how much crypto could be bought initially
        crypto_amount = self.initial_balance / initial_price
        
        # Calculate final portfolio value
        final_value = crypto_amount * final_price
        
        return final_value
