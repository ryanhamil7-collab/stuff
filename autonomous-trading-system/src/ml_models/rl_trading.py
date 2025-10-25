"""
Reinforcement Learning for Trading

Implements DQN and PPO algorithms for optimizing trading actions.
Uses stable-baselines3 for robust RL implementations.

Actions: BUY, SELL, HOLD
Rewards: Risk-adjusted returns (Sharpe ratio based)
Environment: Gym-like trading environment

Target: Optimal trading policy through RL
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from src.utils import log, config

class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for RL
    
    State: Market features (price, volume, indicators, sentiment)
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Reward: Risk-adjusted returns
    
    Compatible with stable-baselines3 and OpenAI Gym.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0
    ):
        """
        Initialize trading environment
        
        Args:
            data: Market data with features
            initial_balance: Starting capital
            transaction_cost: Transaction cost per trade
            max_position: Maximum position size (fraction of balance)
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        self.num_features = len(data.columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        
        self.net_worth_history = []
        self.trades_history = []
        
        log.info(f"TradingEnvironment initialized - Data: {len(data)} steps, Features: {self.num_features}")
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.net_worth_history = [self.initial_balance]
        self.trades_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation (state)"""
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        return obs
    
    def step(self, action):
        """
        Execute action and return next state
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            observation, reward, done, info
        """
        current_price = self.data.iloc[self.current_step]['Close']
        
        if action == 1:  # BUY
            shares_to_buy = int((self.balance * self.max_position) / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.trades_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price
                    })
        
        elif action == 2:  # SELL
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': self.shares_held,
                    'price': current_price
                })
                self.shares_held = 0
        
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        reward = self._calculate_reward()
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        obs = self._get_observation() if not done else np.zeros(self.num_features, dtype=np.float32)
        
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'num_trades': len(self.trades_history)
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self):
        """
        Calculate reward based on risk-adjusted returns
        
        Uses Sharpe-like ratio: (return - risk_free) / volatility
        """
        if len(self.net_worth_history) < 2:
            return 0.0
        
        returns = np.diff(self.net_worth_history) / self.net_worth_history[:-1]
        
        mean_return = np.mean(returns)
        
        volatility = np.std(returns) if len(returns) > 1 else 1.0
        
        if volatility > 0:
            reward = mean_return / volatility
        else:
            reward = mean_return
        
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward -= drawdown * 0.5
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state"""
        profit = self.net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        print(f"Step: {self.current_step}/{len(self.data)}")
        print(f"Net Worth: ${self.net_worth:.2f} ({profit_pct:+.2f}%)")
        print(f"Balance: ${self.balance:.2f}, Shares: {self.shares_held}")
        print(f"Trades: {len(self.trades_history)}")


class RLTrader:
    """
    Reinforcement Learning Trader
    
    Supports:
    - PPO (Proximal Policy Optimization)
    - DQN (Deep Q-Network)
    
    Features:
    - GPU acceleration
    - Automatic hyperparameter tuning
    - Model checkpointing
    - Evaluation callbacks
    """
    
    def __init__(
        self,
        algorithm: str = 'ppo',
        use_gpu: bool = True
    ):
        """
        Initialize RL trader
        
        Args:
            algorithm: 'ppo' or 'dqn'
            use_gpu: Use GPU if available
        """
        self.config = config.get('ml.rl', {
            'algorithm': 'ppo',
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'total_timesteps': 100000,
            'use_gpu': True
        })
        
        self.algorithm = algorithm or self.config['algorithm']
        self.use_gpu = use_gpu and self.config['use_gpu']
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.env = None
        
        log.info(f"RLTrader initialized - Algorithm: {self.algorithm.upper()}, Device: {self.device}")
    
    def create_environment(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0
    ) -> TradingEnvironment:
        """Create trading environment"""
        env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance
        )
        
        self.env = DummyVecEnv([lambda: env])
        
        return env
    
    def build_model(self):
        """Build RL model"""
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment() first.")
        
        if self.algorithm == 'ppo':
            self.model = PPO(
                policy='MlpPolicy',
                env=self.env,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=self.config['clip_range'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                device=self.device,
                verbose=1
            )
        elif self.algorithm == 'dqn':
            self.model = DQN(
                policy='MlpPolicy',
                env=self.env,
                learning_rate=self.config['learning_rate'],
                buffer_size=50000,
                learning_starts=1000,
                batch_size=self.config['batch_size'],
                gamma=self.config['gamma'],
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                device=self.device,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        log.info(f"Built {self.algorithm.upper()} model")
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_env: Optional[gym.Env] = None
    ) -> Dict:
        """
        Train RL model
        
        Args:
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            
        Returns:
            Training statistics
        """
        if self.model is None:
            self.build_model()
        
        total_timesteps = total_timesteps or self.config['total_timesteps']
        
        log.info(f"Starting RL training for {total_timesteps} timesteps...")
        
        callbacks = []
        
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./models/rl_best/',
                log_path='./logs/rl/',
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None
        )
        
        log.info("RL training complete")
        
        return {
            'total_timesteps': total_timesteps,
            'algorithm': self.algorithm
        }
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict action for given observation
        
        Args:
            observation: Current state
            
        Returns:
            Action (0=HOLD, 1=BUY, 2=SELL)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(observation, deterministic=True)
        
        return int(action)
    
    def evaluate(
        self,
        eval_env: gym.Env,
        num_episodes: int = 10
    ) -> Dict:
        """
        Evaluate trained model
        
        Args:
            eval_env: Evaluation environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        log.info(f"Evaluating model for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'num_episodes': num_episodes
        }
        
        log.info(f"Evaluation complete - Mean Reward: {metrics['mean_reward']:.4f}")
        
        return metrics
    
    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        log.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        if self.algorithm == 'ppo':
            self.model = PPO.load(path, device=self.device)
        elif self.algorithm == 'dqn':
            self.model = DQN.load(path, device=self.device)
        
        log.info(f"Model loaded from {path}")
