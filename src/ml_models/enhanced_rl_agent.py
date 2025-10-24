"""
Enhanced Reinforcement Learning agent with improved architecture and error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

from src.utils import log, config
from src.utils.exceptions import ModelError, ValidationError
from src.utils.validators import validate_dataframe, validate_positive_number


class TradingEnvironment(gym.Env):
    """
    Enhanced trading environment with better state representation and reward shaping.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.1
    ):
        """
        Initialize trading environment.
        
        Args:
            data: Market data DataFrame
            initial_balance: Initial account balance
            transaction_cost: Transaction cost as fraction
            max_position_size: Maximum position size as fraction of balance
        """
        super().__init__()
        
        try:
            validate_dataframe(data, required_columns=['Close', 'Volume'])
            validate_positive_number(initial_balance, "initial_balance")
            
            self.data = data.reset_index(drop=True)
            self.initial_balance = initial_balance
            self.transaction_cost = transaction_cost
            self.max_position_size = max_position_size
            
            self.n_features = len(self._get_state(0))
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_features,),
                dtype=np.float32
            )
            
            self.action_space = spaces.Discrete(3)
            
            self.reset()
            
            log.info(f"TradingEnvironment initialized with {len(self.data)} steps")
            
        except Exception as e:
            log.error(f"Error initializing TradingEnvironment: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to initialize trading environment: {str(e)}") from e
    
    def _get_state(self, step: int) -> np.ndarray:
        """
        Get state representation at a given step.
        
        Args:
            step: Current step
            
        Returns:
            State vector
        """
        try:
            if step >= len(self.data):
                step = len(self.data) - 1
            
            row = self.data.iloc[step]
            
            price = row['Close']
            volume = row['Volume']
            
            state = [
                self.balance / self.initial_balance,  # Normalized balance
                self.position,  # Current position (-1, 0, 1)
                price / self.initial_price if self.initial_price > 0 else 1.0,  # Normalized price
                volume / self.data['Volume'].mean() if self.data['Volume'].mean() > 0 else 1.0,  # Normalized volume
            ]
            
            for col in ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20', 'SMA_50']:
                if col in self.data.columns:
                    value = row.get(col, 0)
                    if pd.notna(value):
                        state.append(float(value))
                    else:
                        state.append(0.0)
            
            return np.array(state, dtype=np.float32)
            
        except Exception as e:
            log.error(f"Error getting state at step {step}: {str(e)}")
            return np.zeros(self.n_features, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Returns:
            Initial state and info dict
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.trades = []
        self.initial_price = self.data.iloc[0]['Close']
        
        return self._get_state(0), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        try:
            current_price = self.data.iloc[self.current_step]['Close']
            
            reward = 0.0
            trade_executed = False
            
            if action == 1 and self.position <= 0:  # Buy
                if self.position < 0:
                    profit = (self.entry_price - current_price) * abs(self.position)
                    profit -= abs(self.position) * current_price * self.transaction_cost
                    self.balance += profit
                    self.total_profit += profit
                    reward += profit / self.initial_balance
                
                position_size = min(
                    self.balance * self.max_position_size / current_price,
                    self.balance / current_price
                )
                cost = position_size * current_price * (1 + self.transaction_cost)
                
                if cost <= self.balance:
                    self.position = position_size
                    self.entry_price = current_price
                    self.balance -= cost
                    trade_executed = True
            
            elif action == 2 and self.position >= 0:  # Sell
                if self.position > 0:
                    profit = (current_price - self.entry_price) * self.position
                    profit -= self.position * current_price * self.transaction_cost
                    self.balance += self.position * current_price - self.position * current_price * self.transaction_cost
                    self.total_profit += profit
                    reward += profit / self.initial_balance
                    self.position = 0
                    trade_executed = True
            
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position
                reward += unrealized_pnl / self.initial_balance * 0.1  # Small reward for unrealized gains
            
            if self.position == 0:
                reward -= 0.0001
            
            self.current_step += 1
            terminated = self.current_step >= len(self.data) - 1
            truncated = False
            
            next_state = self._get_state(self.current_step)
            
            info = {
                'balance': self.balance,
                'position': self.position,
                'total_profit': self.total_profit,
                'trade_executed': trade_executed,
                'current_price': current_price
            }
            
            return next_state, reward, terminated, truncated, info
            
        except Exception as e:
            log.error(f"Error in environment step: {str(e)}", exc_info=True)
            return self._get_state(self.current_step), 0.0, True, False, {}


class ProgressCallback(BaseCallback):
    """Callback for logging training progress."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            log.info(f"Training step: {self.n_calls}")
        return True


class EnhancedRLAgent:
    """
    Enhanced RL agent with improved training and evaluation.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = 'auto'
    ):
        """
        Initialize the RL agent.
        
        Args:
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.device = device
        self.model = None
        self.env = None
        
        log.info("EnhancedRLAgent initialized")
    
    def create_environment(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0
    ) -> TradingEnvironment:
        """
        Create a trading environment.
        
        Args:
            data: Market data
            initial_balance: Initial balance
            
        Returns:
            Trading environment
        """
        try:
            env = TradingEnvironment(data, initial_balance)
            return env
        except Exception as e:
            log.error(f"Error creating environment: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to create environment: {str(e)}") from e
    
    def train(
        self,
        data: pd.DataFrame,
        total_timesteps: int = 100000,
        initial_balance: float = 100000.0,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            data: Training data
            total_timesteps: Total training timesteps
            initial_balance: Initial balance
            save_path: Path to save model
            
        Returns:
            Training metrics
        """
        try:
            log.info(f"Starting RL agent training for {total_timesteps} timesteps")
            
            env = self.create_environment(data, initial_balance)
            vec_env = DummyVecEnv([lambda: env])
            
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                verbose=1,
                device=self.device
            )
            
            callback = ProgressCallback(check_freq=1000)
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            if save_path:
                self.model.save(save_path)
                log.info(f"Model saved to {save_path}")
            
            log.info("Training completed successfully")
            
            return {
                'total_timesteps': total_timesteps,
                'final_learning_rate': self.learning_rate
            }
            
        except Exception as e:
            log.error(f"Error training RL agent: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to train RL agent: {str(e)}") from e
    
    def predict(
        self,
        state: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action for a given state.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, action_probabilities)
        """
        try:
            if self.model is None:
                raise ModelError("Model not trained or loaded")
            
            action, _states = self.model.predict(state, deterministic=deterministic)
            return int(action), None
            
        except Exception as e:
            log.error(f"Error predicting action: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to predict action: {str(e)}") from e
    
    def evaluate(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        n_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            data: Evaluation data
            initial_balance: Initial balance
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        try:
            if self.model is None:
                raise ModelError("Model not trained or loaded")
            
            log.info(f"Evaluating RL agent for {n_episodes} episodes")
            
            env = self.create_environment(data, initial_balance)
            
            episode_rewards = []
            episode_profits = []
            
            for episode in range(n_episodes):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.predict(state, deterministic=True)
                    state, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                episode_rewards.append(episode_reward)
                episode_profits.append(info['total_profit'])
                
                log.info(f"Episode {episode + 1}/{n_episodes}: "
                        f"Reward={episode_reward:.4f}, Profit=${info['total_profit']:.2f}")
            
            metrics = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_profit': np.mean(episode_profits),
                'std_profit': np.std(episode_profits),
                'min_profit': np.min(episode_profits),
                'max_profit': np.max(episode_profits)
            }
            
            log.info(f"Evaluation complete: Mean Reward={metrics['mean_reward']:.4f}, "
                    f"Mean Profit=${metrics['mean_profit']:.2f}")
            
            return metrics
            
        except Exception as e:
            log.error(f"Error evaluating RL agent: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to evaluate RL agent: {str(e)}") from e
    
    def load(self, path: str) -> None:
        """
        Load a trained model.
        
        Args:
            path: Path to model file
        """
        try:
            self.model = PPO.load(path, device=self.device)
            log.info(f"Model loaded from {path}")
        except Exception as e:
            log.error(f"Error loading model: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to load model: {str(e)}") from e
