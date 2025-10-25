"""
Model Sharing Module

Handles sharing of model updates, gradients, alphas, and strategies
across the P2P hive mind network with privacy preservation.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SharedUpdate:
    """Shared model update structure"""
    update_id: str
    update_type: str  # gradient | alpha | strategy | hypothesis
    sender_id: str
    timestamp: float
    
    # Update data (anonymized)
    data: Dict[str, Any]
    
    # Metadata
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    
    # Security
    checksum: str
    signature: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SharedUpdate':
        return cls(**data)
    
    def verify_checksum(self) -> bool:
        """Verify data integrity"""
        data_str = json.dumps(self.data, sort_keys=True)
        computed_checksum = hashlib.sha256(data_str.encode()).hexdigest()
        return computed_checksum == self.checksum


class ModelSharer:
    """
    Model Sharing Manager
    
    Manages sharing of model updates across hive mind network.
    """
    
    def __init__(self, node, config: dict):
        """Initialize model sharer
        
        Args:
            node: P2PNode instance
            config: Sharing configuration
        """
        self.node = node
        self.config = config
        
        # Shared updates cache
        self.shared_updates: Dict[str, SharedUpdate] = {}
        self.received_updates: Dict[str, SharedUpdate] = {}
        
        # Statistics
        self.stats = {
            'updates_shared': 0,
            'updates_received': 0,
            'gradients_shared': 0,
            'alphas_shared': 0,
            'strategies_shared': 0
        }
        
        logger.info("Model sharer initialized")
    
    def share_gradient(self, gradient: np.ndarray, metrics: dict):
        """Share model gradient with network
        
        Args:
            gradient: Model gradient
            metrics: Performance metrics
        """
        if not self.config['sharing']['share_gradients']:
            return
        
        # Anonymize gradient
        anonymized_gradient = self._anonymize_gradient(gradient)
        
        # Create update
        update = self._create_update(
            update_type='gradient',
            data={'gradient': anonymized_gradient.tolist()},
            metrics=metrics
        )
        
        # Share with network
        self._share_update(update)
        self.stats['gradients_shared'] += 1
    
    def share_alpha(self, alpha_formula: str, metrics: dict):
        """Share alpha factor with network
        
        Args:
            alpha_formula: Alpha formula string
            metrics: Performance metrics (RankIC, ICIR, etc.)
        """
        if not self.config['sharing']['share_alphas']:
            return
        
        # Check if meets quality threshold
        if not self._meets_quality_threshold(metrics):
            logger.debug("Alpha doesn't meet quality threshold")
            return
        
        # Create update
        update = self._create_update(
            update_type='alpha',
            data={'formula': alpha_formula},
            metrics=metrics
        )
        
        # Share with network
        self._share_update(update)
        self.stats['alphas_shared'] += 1
    
    def share_strategy(self, strategy: dict, metrics: dict):
        """Share trading strategy with network
        
        Args:
            strategy: Strategy configuration
            metrics: Performance metrics
        """
        if not self.config['sharing']['share_strategies']:
            return
        
        # Check if meets quality threshold
        if not self._meets_quality_threshold(metrics):
            logger.debug("Strategy doesn't meet quality threshold")
            return
        
        # Anonymize strategy
        anonymized_strategy = self._anonymize_strategy(strategy)
        
        # Create update
        update = self._create_update(
            update_type='strategy',
            data=anonymized_strategy,
            metrics=metrics
        )
        
        # Share with network
        self._share_update(update)
        self.stats['strategies_shared'] += 1
    
    def receive_update(self, update_data: dict):
        """Receive and process shared update
        
        Args:
            update_data: Update data dictionary
        """
        try:
            update = SharedUpdate.from_dict(update_data)
            
            # Verify checksum
            if not update.verify_checksum():
                logger.warning(f"Invalid checksum for update {update.update_id}")
                return
            
            # Store update
            self.received_updates[update.update_id] = update
            self.stats['updates_received'] += 1
            
            logger.info(f"Received {update.update_type} update from {update.sender_id}")
            
        except Exception as e:
            logger.error(f"Error receiving update: {e}")
    
    def _create_update(self, update_type: str, data: dict, metrics: dict) -> SharedUpdate:
        """Create shared update
        
        Args:
            update_type: Type of update
            data: Update data
            metrics: Performance metrics
            
        Returns:
            SharedUpdate object
        """
        # Generate update ID
        update_id = hashlib.sha256(
            f"{self.node.node_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Compute checksum
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Create update
        update = SharedUpdate(
            update_id=update_id,
            update_type=update_type,
            sender_id=self.node.node_id,
            timestamp=datetime.now().timestamp(),
            data=data,
            performance_metrics=metrics,
            validation_results={},
            checksum=checksum
        )
        
        return update
    
    def _share_update(self, update: SharedUpdate):
        """Share update with network
        
        Args:
            update: Update to share
        """
        # Store locally
        self.shared_updates[update.update_id] = update
        
        # Broadcast to network
        self.node.broadcast_message('model_update', update.to_dict())
        
        self.stats['updates_shared'] += 1
        logger.info(f"Shared {update.update_type} update: {update.update_id}")
    
    def _anonymize_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Anonymize gradient for privacy
        
        Args:
            gradient: Original gradient
            
        Returns:
            Anonymized gradient
        """
        if not self.config['sharing']['anonymize']:
            return gradient
        
        # Add differential privacy noise
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, gradient.shape)
        return gradient + noise
    
    def _anonymize_strategy(self, strategy: dict) -> dict:
        """Anonymize strategy for privacy
        
        Args:
            strategy: Original strategy
            
        Returns:
            Anonymized strategy
        """
        if not self.config['sharing']['anonymize']:
            return strategy
        
        # Remove sensitive information
        anonymized = strategy.copy()
        anonymized.pop('capital', None)
        anonymized.pop('positions', None)
        anonymized.pop('trades', None)
        
        return anonymized
    
    def _meets_quality_threshold(self, metrics: dict) -> bool:
        """Check if update meets quality threshold
        
        Args:
            metrics: Performance metrics
            
        Returns:
            True if meets threshold
        """
        min_sharpe = self.config['sharing'].get('min_sharpe_to_share', 1.5)
        min_win_rate = self.config['sharing'].get('min_win_rate_to_share', 0.60)
        
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        
        return sharpe >= min_sharpe and win_rate >= min_win_rate
    
    def get_stats(self) -> dict:
        """Get sharing statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'shared_updates_count': len(self.shared_updates),
            'received_updates_count': len(self.received_updates)
        }
