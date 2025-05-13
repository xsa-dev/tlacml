import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pytorch_lightning as pl
from typing import Tuple, Dict, List, Any, Optional
import gymnasium as gym

class ActorNetwork(nn.Module):
    """
    Actor network for the PPO agent.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the actor network.
        
        Args:
            input_dim: Dimension of the input (observation space)
            hidden_dim: Dimension of the hidden layers
            output_dim: Dimension of the output (action space)
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Action probabilities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.softmax(x, dim=-1)

class CriticNetwork(nn.Module):
    """
    Critic network for the PPO agent.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize the critic network.
        
        Args:
            input_dim: Dimension of the input (observation space)
            hidden_dim: Dimension of the hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Value estimate
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class PPOMemory:
    """
    Memory buffer for PPO.
    """
    def __init__(self, batch_size: int):
        """
        Initialize the memory buffer.
        
        Args:
            batch_size: Size of batches for training
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def store(self, state, action, prob, val, reward, done):
        """
        Store a transition in the buffer.
        
        Args:
            state: Observation
            action: Action taken
            prob: Probability of the action
            val: Value estimate
            reward: Reward received
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        """
        Clear the buffer.
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self) -> Tuple[List[np.ndarray], ...]:
        """
        Generate batches for training.
        
        Returns:
            Batches of states, actions, old_probs, vals, rewards, dones, and batch indices
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

class PPOAgent(pl.LightningModule):
    """
    PPO agent implemented with PyTorch Lightning.
    """
    def __init__(self, 
                 input_dim: int, 
                 n_actions: int, 
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 policy_clip: float = 0.2,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize the PPO agent.
        
        Args:
            input_dim: Dimension of the input (observation space)
            n_actions: Number of possible actions
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            gae_lambda: Lambda parameter for GAE
            policy_clip: Clipping parameter for PPO
            batch_size: Size of batches for training
            n_epochs: Number of epochs to train on each batch of data
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
        """
        super(PPOAgent, self).__init__()
        self.save_hyperparameters()
        
        self.actor = ActorNetwork(input_dim, hidden_dim, n_actions)
        self.critic = CriticNetwork(input_dim, hidden_dim)
        self.memory = PPOMemory(batch_size)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the networks.
        
        Args:
            state: Input state
            
        Returns:
            Action probabilities and value estimate
        """
        action_probs = self.actor(state)
        value = self.critic(state)
        
        return action_probs, value
    
    def choose_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        """
        Choose an action based on the observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Chosen action, log probability of the action, and value estimate
        """
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        
        action_probs, value = self(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def learn(self):
        """
        Update the networks based on collected experience.
        """
        # Create optimizer if it doesn't exist
        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            # Calculate advantages using GAE
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-done_arr[k]) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            
            advantage = torch.tensor(advantage, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32)
                actions = torch.tensor(action_arr[batch], dtype=torch.int64)
                
                # Get new action probabilities and values
                action_probs, critic_value = self(states)
                critic_value = critic_value.squeeze()
                
                # Get log probabilities of actions
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate probability ratio
                prob_ratio = torch.exp(new_probs - old_probs)
                
                # Calculate surrogate losses
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = advantage[batch] * torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                
                # Calculate actor loss
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Calculate critic loss
                critic_loss = F.mse_loss(critic_value, values[batch])
                
                # Calculate total loss
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # Clear memory after learning
        self.memory.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizers for PyTorch Lightning.
        
        Returns:
            Optimizer for the networks
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        # This is a placeholder for PyTorch Lightning
        # The actual training happens in the learn method
        return {"loss": torch.tensor(0.0, requires_grad=True)}
    
    def save_models(self, path: str):
        """
        Save the models to disk.
        
        Args:
            path: Path to save the models
        """
        torch.save(self.state_dict(), path)
    
    def load_models(self, path: str):
        """
        Load the models from disk.
        
        Args:
            path: Path to load the models from
        """
        self.load_state_dict(torch.load(path))

def train_ppo(env: gym.Env, 
              agent: PPOAgent, 
              n_episodes: int, 
              max_steps: int, 
              update_interval: int = 20,
              log_interval: int = 10,
              save_path: Optional[str] = None) -> List[float]:
    """
    Train the PPO agent.
    
    Args:
        env: Environment to train in
        agent: PPO agent
        n_episodes: Number of episodes to train for
        max_steps: Maximum number of steps per episode
        update_interval: Number of steps between updates
        log_interval: Number of episodes between logs
        save_path: Path to save the agent
        
    Returns:
        List of episode rewards
    """
    best_reward = float('-inf')
    episode_rewards = []
    
    for episode in range(n_episodes):
        observation, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (done or truncated) and step_count < max_steps:
            action, prob, val = agent.choose_action(observation)
            next_observation, reward, done, truncated, _ = env.step(action)
            
            agent.memory.store(observation, action, prob, val, reward, done)
            episode_reward += reward
            
            if step_count % update_interval == 0 and step_count > 0:
                agent.learn()
            
            observation = next_observation
            step_count += 1
        
        # Final update at the end of the episode
        if len(agent.memory.states) > 0:
            agent.learn()
        
        episode_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.2f}")
            
            # Save best models
            if save_path and avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_models(save_path)
                print(f"Saved best model with avg reward: {best_reward:.2f}")
    
    return episode_rewards
