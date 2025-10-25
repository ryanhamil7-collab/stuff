"""
High-Risk Mode - Feature 20

Toggleable aggressive trading mode for users seeking higher returns.

Features:
- 5-10x leverage simulation
- 20-30% position sizing (vs 5-10% normal)
- Prioritizes volatile symbols (>5% volatility)
- 80% intraday allocation for faster trades
- Emergency shutdown at -30% drawdown
- Adjusted delay/slippage (1.5x/1.2x)

Target: 2-3x returns (300-500% annual vs 150-250% baseline)
Risk: 2-3x drawdowns as well
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from src.utils import log, config
from src.strategies.portfolio import Portfolio
from src.strategies.risk_manager import RiskManager

class HighRiskStrategy:
    """
    High-Risk Trading Strategy
    
    WARNING: This mode uses aggressive position sizing and leverage.
    Only suitable for users with high risk tolerance.
    
    Features:
    - Leveraged positions (5-10x)
    - Large position sizes (20-30% of capital)
    - Focus on volatile symbols
    - Fast intraday trading
    - Emergency stop-loss at -30% drawdown
    """
    
    def __init__(
        self,
        enabled: bool = False,
        leverage: float = None,
        position_size_pct: float = None,
        emergency_stop_pct: float = None
    ):
        """
        Initialize high-risk strategy
        
        Args:
            enabled: Enable high-risk mode
            leverage: Leverage multiplier (5-10x)
            position_size_pct: Position size as % of capital (0.2-0.3)
            emergency_stop_pct: Emergency stop at drawdown % (0.3)
        """
        self.high_risk_config = config.get('trading.high_risk', {
            'enabled': False,
            'leverage': 7.0,
            'position_size_pct': 0.25,
            'min_volatility': 0.05,
            'intraday_allocation': 0.8,
            'emergency_stop_pct': 0.30,
            'delay_multiplier': 1.5,
            'slippage_multiplier': 1.2,
            'max_positions': 3,
            'target_return_annual': 4.0  # 400%
        })
        
        if enabled is not None:
            self.high_risk_config['enabled'] = enabled
        if leverage is not None:
            self.high_risk_config['leverage'] = leverage
        if position_size_pct is not None:
            self.high_risk_config['position_size_pct'] = position_size_pct
        if emergency_stop_pct is not None:
            self.high_risk_config['emergency_stop_pct'] = emergency_stop_pct
        
        self.enabled = self.high_risk_config['enabled']
        self.leverage = self.high_risk_config['leverage']
        self.position_size_pct = self.high_risk_config['position_size_pct']
        self.emergency_stop_pct = self.high_risk_config['emergency_stop_pct']
        
        self.emergency_stop_triggered = False
        self.initial_capital = None
        self.peak_equity = None
        
        if self.enabled:
            log.warning("="*70)
            log.warning("HIGH-RISK MODE ENABLED")
            log.warning(f"Leverage: {self.leverage}x")
            log.warning(f"Position Size: {self.position_size_pct:.1%}")
            log.warning(f"Emergency Stop: {self.emergency_stop_pct:.1%} drawdown")
            log.warning("WARNING: This mode can result in significant losses!")
            log.warning("="*70)
        else:
            log.info("High-risk mode disabled (normal trading)")
    
    def filter_symbols(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Filter symbols for high-risk trading
        
        Prioritizes:
        - High volatility (>5%)
        - High volume (liquid)
        - Strong momentum
        - Crypto and options if available
        """
        if not self.enabled:
            return symbols
        
        scored_symbols = []
        min_volatility = self.high_risk_config['min_volatility']
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std()
            
            momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] if len(df) >= 20 else 0
            
            avg_volume = df['Volume'].mean()
            
            score = 0
            
            if volatility > min_volatility:
                score += volatility * 100
            
            score += abs(momentum) * 50
            
            if avg_volume > 1000000:
                score += 10
            
            if '-USD' in symbol or 'BTC' in symbol or 'ETH' in symbol:
                score += 20  # Crypto bonus
            
            scored_symbols.append((symbol, score))
        
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        max_positions = self.high_risk_config['max_positions']
        
        filtered = [s[0] for s in scored_symbols[:max_positions]]
        
        log.info(f"High-risk mode: Selected {len(filtered)} volatile symbols")
        return filtered
    
    def adjust_position_size(
        self,
        base_position_size: float,
        symbol: str,
        current_capital: float
    ) -> float:
        """
        Adjust position size for high-risk mode
        
        Args:
            base_position_size: Normal position size
            symbol: Trading symbol
            current_capital: Current portfolio capital
            
        Returns:
            Adjusted position size with leverage
        """
        if not self.enabled:
            return base_position_size
        
        if self.emergency_stop_triggered:
            log.warning("Emergency stop active - no new positions")
            return 0
        
        leveraged_size = current_capital * self.position_size_pct * self.leverage
        
        log.info(f"High-risk position size for {symbol}: ${leveraged_size:,.2f} ({self.leverage}x leverage)")
        
        return leveraged_size
    
    def check_emergency_stop(
        self,
        current_equity: float,
        initial_capital: float = None
    ) -> bool:
        """
        Check if emergency stop should be triggered
        
        Args:
            current_equity: Current portfolio equity
            initial_capital: Initial capital (for first call)
            
        Returns:
            True if emergency stop triggered
        """
        if not self.enabled:
            return False
        
        if self.emergency_stop_triggered:
            return True
        
        if self.initial_capital is None and initial_capital is not None:
            self.initial_capital = initial_capital
            self.peak_equity = initial_capital
        
        if self.initial_capital is None:
            return False
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown >= self.emergency_stop_pct:
            self.emergency_stop_triggered = True
            log.error("="*70)
            log.error("EMERGENCY STOP TRIGGERED!")
            log.error(f"Drawdown: {drawdown:.1%} (threshold: {self.emergency_stop_pct:.1%})")
            log.error(f"Peak equity: ${self.peak_equity:,.2f}")
            log.error(f"Current equity: ${current_equity:,.2f}")
            log.error("All positions will be closed. High-risk mode disabled.")
            log.error("="*70)
            return True
        
        elif drawdown >= 0.20:
            log.warning(f"High drawdown warning: {drawdown:.1%}")
        
        return False
    
    def adjust_risk_params(self, base_params: Dict) -> Dict:
        """
        Adjust risk parameters for high-risk mode
        
        Args:
            base_params: Normal risk parameters
            
        Returns:
            Adjusted parameters for high-risk
        """
        if not self.enabled:
            return base_params
        
        adjusted = base_params.copy()
        
        if 'stop_loss_pct' in adjusted:
            adjusted['stop_loss_pct'] *= 2.0
        
        if 'take_profit_pct' in adjusted:
            adjusted['take_profit_pct'] *= 2.0
        
        if 'execution_delay_ms' in adjusted:
            adjusted['execution_delay_ms'] *= self.high_risk_config['delay_multiplier']
        
        if 'slippage_bps' in adjusted:
            adjusted['slippage_bps'] *= self.high_risk_config['slippage_multiplier']
        
        return adjusted
    
    def get_mode_info(self) -> Dict:
        """Get high-risk mode information"""
        return {
            'enabled': self.enabled,
            'leverage': self.leverage,
            'position_size_pct': self.position_size_pct,
            'emergency_stop_pct': self.emergency_stop_pct,
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'initial_capital': self.initial_capital,
            'peak_equity': self.peak_equity,
            'current_drawdown': (self.peak_equity - self.initial_capital) / self.peak_equity if self.peak_equity else 0,
            'target_return_annual': self.high_risk_config['target_return_annual']
        }
    
    def reset_emergency_stop(self):
        """Reset emergency stop (use with caution!)"""
        if self.emergency_stop_triggered:
            log.warning("Resetting emergency stop - use with extreme caution!")
            self.emergency_stop_triggered = False
            self.initial_capital = None
            self.peak_equity = None
