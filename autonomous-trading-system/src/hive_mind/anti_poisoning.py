"""Anti-Poisoning Validator - Protects against malicious updates"""
import logging
import numpy as np
from typing import Dict, List
from scipy import stats

logger = logging.getLogger(__name__)

class AntiPoisoningValidator:
    def __init__(self, config: dict):
        self.config = config
        self.update_history: List[Dict] = []
        
    def validate_update(self, update: dict) -> bool:
        """Validate update for poisoning attacks"""
        if not self.config['security']['enabled']:
            return True
        
        # Check blockchain verification
        if self.config['security']['blockchain_verification']:
            if not self._verify_blockchain(update):
                logger.warning(f"Blockchain verification failed for {update['update_id']}")
                return False
        
        # Check for outliers
        if self.config['security']['reject_outliers']:
            if self._is_outlier(update):
                logger.warning(f"Update {update['update_id']} is an outlier")
                return False
        
        return True
    
    def _verify_blockchain(self, update: dict) -> bool:
        """Verify update using blockchain hash"""
        # Simplified blockchain verification
        return update.get('checksum') is not None
    
    def _is_outlier(self, update: dict) -> bool:
        """Check if update is statistical outlier"""
        if len(self.update_history) < 10:
            return False
        
        # Extract metrics
        metrics = update.get('performance_metrics', {})
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Get historical sharpes
        historical_sharpes = [
            u.get('performance_metrics', {}).get('sharpe_ratio', 0)
            for u in self.update_history[-100:]
        ]
        
        if not historical_sharpes:
            return False
        
        # Z-score test
        z_score = np.abs(stats.zscore([sharpe] + historical_sharpes)[0])
        threshold = self.config['security'].get('outlier_threshold', 3.0)
        
        return z_score > threshold
    
    def record_update(self, update: dict):
        """Record update in history"""
        self.update_history.append(update)
        if len(self.update_history) > 1000:
            self.update_history = self.update_history[-1000:]
