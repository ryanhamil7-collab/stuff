"""Incentive System - Rewards for contributions"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class IncentiveManager:
    def __init__(self, config: dict):
        self.config = config
        self.balances: Dict[str, float] = {}
        
    def reward_contribution(self, node_id: str, contribution_type: str):
        """Reward node for contribution"""
        if not self.config['incentives']['enabled']:
            return
        
        rewards = {
            'gradient': self.config['incentives']['reward_per_gradient'],
            'alpha': self.config['incentives']['reward_per_alpha'],
            'strategy': self.config['incentives']['reward_per_strategy']
        }
        
        reward = rewards.get(contribution_type, 0)
        self.balances[node_id] = self.balances.get(node_id, 0) + reward
        logger.info(f"Rewarded {node_id} with {reward} {self.config['incentives']['token_name']}")
    
    def penalize_node(self, node_id: str):
        """Penalize node for bad contribution"""
        if not self.config['incentives']['enabled']:
            return
        
        penalty = self.config['incentives']['penalty_for_rejected']
        self.balances[node_id] = self.balances.get(node_id, 0) + penalty
        logger.info(f"Penalized {node_id} with {penalty} {self.config['incentives']['token_name']}")
    
    def get_balance(self, node_id: str) -> float:
        """Get node's token balance"""
        return self.balances.get(node_id, 0)
