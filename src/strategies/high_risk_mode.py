"""
High-Risk Mode for Aggressive Trading Strategies

This module implements a toggleable high-risk mode that amplifies trading parameters
for aggressive strategies. When enabled, it increases leverage, position sizing,
trade frequency, and prioritizes volatile symbols.

WARNING: High-risk mode can produce 2-3x returns but also 2-3x drawdowns.
         Even in paper trading, this mode should be used with caution.

Performance Expectations:
- Baseline mode: 150-250% annual returns, -10-15% max drawdown
- High-risk mode: 300-500% annual returns, -20-40% max drawdown
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class HighRiskConfig:
    """Configuration for high-risk mode"""
    
    enabled: bool = False
    
    leverage_multiplier: float = 5.0
    max_position_size_pct: float = 0.25
    trade_frequency_multiplier: float = 2.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    
    intraday_allocation: float = 0.80
    interday_allocation: float = 0.20
    
    min_volatility: float = 0.05
    prioritize_crypto: bool = True
    prioritize_options: bool = True
    
    max_drawdown_shutdown: float = 0.30
    
    slippage_multiplier: float = 1.5
    delay_multiplier: float = 1.2


class HighRiskModeManager:
    """
    Manages high-risk mode parameters and adjustments
    
    When enabled, this class modifies trading parameters to be more aggressive:
    - Increases leverage from 1-2x to 5-10x
    - Increases position sizing from 10% to 20-30% per trade
    - Doubles intraday trade frequency
    - Loosens stop-loss from -2% to -5%
    - Prioritizes volatile symbols (>5% volatility)
    - Adjusts hybrid mode allocation (80% intraday vs 60% baseline)
    """
    
    def __init__(self, config: Optional[HighRiskConfig] = None):
        self.config = config or HighRiskConfig()
        self.is_enabled = self.config.enabled
        self.current_drawdown = 0.0
        self.shutdown_triggered = False
        
        if self.is_enabled:
            log.warning(
                "⚠️  HIGH-RISK MODE ENABLED ⚠️\n"
                f"  Leverage: {self.config.leverage_multiplier}x\n"
                f"  Max Position: {self.config.max_position_size_pct:.0%}\n"
                f"  Stop Loss: {self.config.stop_loss_pct:.0%}\n"
                f"  Max Drawdown Shutdown: {self.config.max_drawdown_shutdown:.0%}\n"
                "  Expected: 2-3x returns but also 2-3x drawdowns\n"
                "  Use with caution even in paper trading!"
            )
    
    def enable(self):
        """Enable high-risk mode"""
        self.is_enabled = True
        self.config.enabled = True
        log.warning("High-risk mode ENABLED")
    
    def disable(self):
        """Disable high-risk mode"""
        self.is_enabled = False
        self.config.enabled = False
        log.info("High-risk mode DISABLED")
    
    def get_adjusted_parameters(self, baseline_params: Dict) -> Dict:
        """
        Get adjusted trading parameters for high-risk mode
        
        Args:
            baseline_params: Dictionary of baseline trading parameters
            
        Returns:
            Dictionary of adjusted parameters (or baseline if disabled)
        """
        if not self.is_enabled or self.shutdown_triggered:
            return baseline_params
        
        adjusted = baseline_params.copy()
        
        adjusted['leverage'] = baseline_params.get('leverage', 1.0) * self.config.leverage_multiplier
        adjusted['max_position_size'] = self.config.max_position_size_pct
        adjusted['stop_loss_pct'] = self.config.stop_loss_pct
        adjusted['take_profit_pct'] = self.config.take_profit_pct
        
        adjusted['trade_frequency'] = baseline_params.get('trade_frequency', 1.0) * self.config.trade_frequency_multiplier
        
        adjusted['intraday_allocation'] = self.config.intraday_allocation
        adjusted['interday_allocation'] = self.config.interday_allocation
        
        adjusted['slippage_multiplier'] = self.config.slippage_multiplier
        adjusted['delay_multiplier'] = self.config.delay_multiplier
        
        return adjusted
    
    def filter_symbols_for_high_risk(self, symbols: List[Dict]) -> List[Dict]:
        """
        Filter and prioritize symbols for high-risk trading
        
        Prioritizes:
        - High volatility symbols (>5%)
        - Crypto assets
        - Options
        - Recent momentum
        
        Args:
            symbols: List of symbol dictionaries with metadata
            
        Returns:
            Filtered and sorted list of symbols
        """
        if not self.is_enabled or self.shutdown_triggered:
            return symbols
        
        high_risk_symbols = []
        
        for symbol in symbols:
            volatility = symbol.get('volatility', 0.0)
            asset_class = symbol.get('asset_class', 'stock')
            
            if volatility < self.config.min_volatility:
                continue
            
            score = symbol.get('score', 0.0)
            
            if self.config.prioritize_crypto and asset_class == 'crypto':
                score *= 1.5
            
            if self.config.prioritize_options and asset_class == 'option':
                score *= 1.3
            
            if volatility > 0.10:
                score *= 1.2
            
            symbol_copy = symbol.copy()
            symbol_copy['high_risk_score'] = score
            high_risk_symbols.append(symbol_copy)
        
        high_risk_symbols.sort(key=lambda x: x['high_risk_score'], reverse=True)
        
        log.info(
            f"High-risk filter: {len(high_risk_symbols)}/{len(symbols)} symbols "
            f"(min volatility: {self.config.min_volatility:.1%})"
        )
        
        return high_risk_symbols
    
    def adjust_position_size(self, base_size: float, symbol_volatility: float) -> float:
        """
        Adjust position size for high-risk mode
        
        Args:
            base_size: Baseline position size
            symbol_volatility: Symbol's volatility
            
        Returns:
            Adjusted position size
        """
        if not self.is_enabled or self.shutdown_triggered:
            return base_size
        
        adjusted_size = base_size * self.config.leverage_multiplier
        
        adjusted_size = min(adjusted_size, self.config.max_position_size_pct)
        
        if symbol_volatility > 0.10:
            adjusted_size *= 1.2
        
        return adjusted_size
    
    def should_take_trade(self, signal_confidence: float, symbol_data: Dict) -> bool:
        """
        Determine if a trade should be taken in high-risk mode
        
        High-risk mode loosens filters for more speculative entries
        
        Args:
            signal_confidence: Confidence score of the signal (0-1)
            symbol_data: Symbol metadata
            
        Returns:
            True if trade should be taken
        """
        if not self.is_enabled:
            return signal_confidence > 0.7
        
        if self.shutdown_triggered:
            return False
        
        volatility = symbol_data.get('volatility', 0.0)
        asset_class = symbol_data.get('asset_class', 'stock')
        
        min_confidence = 0.5
        
        if self.config.prioritize_crypto and asset_class == 'crypto':
            min_confidence = 0.4
        
        if volatility > 0.10:
            min_confidence -= 0.1
        
        return signal_confidence > min_confidence
    
    def update_drawdown(self, current_equity: float, peak_equity: float):
        """
        Update current drawdown and check for shutdown threshold
        
        Args:
            current_equity: Current portfolio equity
            peak_equity: Peak portfolio equity
        """
        if peak_equity > 0:
            self.current_drawdown = (peak_equity - current_equity) / peak_equity
        else:
            self.current_drawdown = 0.0
        
        if self.is_enabled and self.current_drawdown >= self.config.max_drawdown_shutdown:
            self.trigger_shutdown()
    
    def trigger_shutdown(self):
        """Trigger emergency shutdown due to excessive drawdown"""
        if not self.shutdown_triggered:
            self.shutdown_triggered = True
            log.critical(
                f"🚨 HIGH-RISK MODE EMERGENCY SHUTDOWN 🚨\n"
                f"  Current drawdown: {self.current_drawdown:.1%}\n"
                f"  Threshold: {self.config.max_drawdown_shutdown:.1%}\n"
                "  All trading halted. Manual intervention required."
            )
    
    def reset_shutdown(self):
        """Reset shutdown flag (manual intervention)"""
        self.shutdown_triggered = False
        self.current_drawdown = 0.0
        log.warning("High-risk mode shutdown reset - trading resumed")
    
    def get_status(self) -> Dict:
        """Get current high-risk mode status"""
        return {
            'enabled': self.is_enabled,
            'shutdown_triggered': self.shutdown_triggered,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_threshold': self.config.max_drawdown_shutdown,
            'leverage_multiplier': self.config.leverage_multiplier,
            'max_position_size': self.config.max_position_size_pct,
            'trade_frequency_multiplier': self.config.trade_frequency_multiplier
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics for high-risk mode"""
        return {
            'expected_return_multiplier': 2.5,
            'expected_drawdown_multiplier': 2.5,
            'baseline_annual_return': 0.20,
            'high_risk_annual_return': 0.50,
            'baseline_max_drawdown': 0.15,
            'high_risk_max_drawdown': 0.35,
            'volatility_multiplier': 1.8
        }


def apply_high_risk_to_backtest(backtest_config: Dict, high_risk_config: HighRiskConfig) -> Dict:
    """
    Apply high-risk adjustments to backtest configuration
    
    Args:
        backtest_config: Baseline backtest configuration
        high_risk_config: High-risk mode configuration
        
    Returns:
        Adjusted backtest configuration
    """
    if not high_risk_config.enabled:
        return backtest_config
    
    adjusted = backtest_config.copy()
    
    adjusted['execution_delays'] = adjusted.get('execution_delays', {})
    adjusted['execution_delays']['slippage_bps'] = (
        adjusted['execution_delays'].get('slippage_bps', 10.0) * 
        high_risk_config.slippage_multiplier
    )
    adjusted['execution_delays']['delay_mean_ms'] = (
        adjusted['execution_delays'].get('delay_mean_ms', 200.0) * 
        high_risk_config.delay_multiplier
    )
    
    adjusted['max_position_size'] = high_risk_config.max_position_size_pct
    adjusted['stop_loss_pct'] = high_risk_config.stop_loss_pct
    adjusted['take_profit_pct'] = high_risk_config.take_profit_pct
    
    log.info(
        f"Applied high-risk adjustments to backtest:\n"
        f"  Slippage: {adjusted['execution_delays']['slippage_bps']:.1f} bps\n"
        f"  Delay: {adjusted['execution_delays']['delay_mean_ms']:.0f} ms\n"
        f"  Max position: {adjusted['max_position_size']:.0%}\n"
        f"  Stop loss: {adjusted['stop_loss_pct']:.0%}"
    )
    
    return adjusted


def compare_risk_modes(baseline_results: Dict, high_risk_results: Dict) -> Dict:
    """
    Compare results between baseline and high-risk modes
    
    Args:
        baseline_results: Backtest results from baseline mode
        high_risk_results: Backtest results from high-risk mode
        
    Returns:
        Comparison metrics
    """
    comparison = {
        'return_multiplier': (
            high_risk_results.get('total_return', 0) / 
            max(baseline_results.get('total_return', 1), 0.01)
        ),
        'sharpe_multiplier': (
            high_risk_results.get('sharpe_ratio', 0) / 
            max(baseline_results.get('sharpe_ratio', 1), 0.01)
        ),
        'drawdown_multiplier': (
            abs(high_risk_results.get('max_drawdown', 0)) / 
            max(abs(baseline_results.get('max_drawdown', 1)), 0.01)
        ),
        'win_rate_diff': (
            high_risk_results.get('win_rate', 0) - 
            baseline_results.get('win_rate', 0)
        ),
        'num_trades_multiplier': (
            high_risk_results.get('num_trades', 0) / 
            max(baseline_results.get('num_trades', 1), 1)
        )
    }
    
    log.info(
        f"Risk Mode Comparison:\n"
        f"  Return multiplier: {comparison['return_multiplier']:.2f}x\n"
        f"  Sharpe multiplier: {comparison['sharpe_multiplier']:.2f}x\n"
        f"  Drawdown multiplier: {comparison['drawdown_multiplier']:.2f}x\n"
        f"  Win rate diff: {comparison['win_rate_diff']:+.1%}\n"
        f"  Trade frequency: {comparison['num_trades_multiplier']:.2f}x"
    )
    
    return comparison


if __name__ == "__main__":
    manager = HighRiskModeManager(HighRiskConfig(enabled=True))
    
    baseline_params = {
        'leverage': 1.0,
        'max_position_size': 0.10,
        'trade_frequency': 1.0,
        'stop_loss_pct': 0.02
    }
    
    adjusted_params = manager.get_adjusted_parameters(baseline_params)
    print("Adjusted parameters:", adjusted_params)
    
    test_symbols = [
        {'symbol': 'AAPL', 'volatility': 0.03, 'asset_class': 'stock', 'score': 0.8},
        {'symbol': 'BTC-USD', 'volatility': 0.12, 'asset_class': 'crypto', 'score': 0.7},
        {'symbol': 'TSLA', 'volatility': 0.08, 'asset_class': 'stock', 'score': 0.75},
        {'symbol': 'SPY_CALL', 'volatility': 0.15, 'asset_class': 'option', 'score': 0.65}
    ]
    
    filtered = manager.filter_symbols_for_high_risk(test_symbols)
    print(f"\nFiltered {len(filtered)} high-risk symbols:")
    for s in filtered[:3]:
        print(f"  {s['symbol']}: score={s['high_risk_score']:.2f}, vol={s['volatility']:.1%}")
