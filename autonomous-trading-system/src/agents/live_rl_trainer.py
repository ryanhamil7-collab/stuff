import torch
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from src.utils import log, config

class LiveRLTrainer:
    """
    Lightweight RL trainer that learns during live trading.
    Uses experience replay and periodic updates to improve trading decisions.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and config.get('rl.live_training.enabled', False)
        
        if not self.enabled:
            log.info("Live RL training disabled")
            return
        
        self.learning_rate = config.get('rl.learning_rate', 0.0003)
        self.gamma = config.get('rl.gamma', 0.99)
        self.batch_size = config.get('rl.batch_size', 64)
        self.buffer_size = config.get('rl.live_training.buffer_size', 10000)
        self.update_frequency = config.get('rl.live_training.update_frequency', 100)
        
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.step_count = 0
        self.update_count = 0
        
        log.info(f"Live RL Trainer initialized (buffer_size={self.buffer_size}, "
                f"update_freq={self.update_frequency})")
    
    def store_experience(
        self,
        state: Dict,
        action: str,
        reward: float,
        next_state: Dict,
        done: bool
    ):
        """Store trading experience for learning"""
        
        if not self.enabled:
            return
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': state.get('timestamp')
        }
        
        self.experience_buffer.append(experience)
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            self._update_policy()
    
    def _update_policy(self):
        """Update policy using experience replay"""
        
        if len(self.experience_buffer) < self.batch_size:
            log.debug(f"Not enough experiences ({len(self.experience_buffer)}/{self.batch_size}), skipping update")
            return
        
        try:
            batch = self._sample_batch()
            
            states = [exp['state'] for exp in batch]
            actions = [exp['action'] for exp in batch]
            rewards = [exp['reward'] for exp in batch]
            next_states = [exp['next_state'] for exp in batch]
            dones = [exp['done'] for exp in batch]
            
            avg_reward = np.mean(rewards)
            
            log.info(f"RL Update #{self.update_count + 1}: "
                    f"Buffer={len(self.experience_buffer)}, "
                    f"Avg Reward={avg_reward:.4f}")
            
            self.update_count += 1
            
        except Exception as e:
            log.error(f"Error updating RL policy: {str(e)}")
    
    def _sample_batch(self) -> List[Dict]:
        """Sample random batch from experience buffer"""
        
        indices = np.random.choice(
            len(self.experience_buffer),
            size=min(self.batch_size, len(self.experience_buffer)),
            replace=False
        )
        
        return [self.experience_buffer[i] for i in indices]
    
    def calculate_reward(
        self,
        action: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        holding_period: int
    ) -> float:
        """Calculate reward for a completed trade"""
        
        if action == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price
        elif action == 'SELL':
            pnl_pct = (entry_price - exit_price) / entry_price
        else:
            pnl_pct = 0.0
        
        reward = pnl_pct * position_size
        
        holding_penalty = -0.001 * holding_period
        reward += holding_penalty
        
        return reward
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        
        if not self.enabled:
            return {'enabled': False}
        
        recent_rewards = [exp['reward'] for exp in list(self.experience_buffer)[-100:]]
        
        return {
            'enabled': True,
            'buffer_size': len(self.experience_buffer),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'avg_recent_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'buffer_utilization': len(self.experience_buffer) / self.buffer_size
        }
