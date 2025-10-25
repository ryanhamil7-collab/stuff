"""
Advanced risk management system with multiple risk control mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.utils import log, config
from src.utils.exceptions import ValidationError
from src.utils.validators import validate_positive_number, validate_percentage


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_value: float
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    risk_level: RiskLevel


@dataclass
class PortfolioRisk:
    """Risk metrics for the entire portfolio."""
    total_value: float
    total_exposure: float
    cash: float
    leverage: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    risk_level: RiskLevel


class AdvancedRiskManager:
    """
    Advanced risk management system with comprehensive risk controls.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.10,  # 10% of portfolio
        max_portfolio_risk: float = 0.02,  # 2% max loss per day
        max_drawdown: float = 0.15,  # 15% max drawdown
        max_leverage: float = 1.0,  # No leverage by default
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.05,  # 5% take profit
        var_confidence: float = 0.95,  # 95% VaR confidence
        correlation_threshold: float = 0.7  # Max correlation between positions
    ):
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Initial portfolio capital
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk per day
            max_drawdown: Maximum allowed drawdown
            max_leverage: Maximum leverage allowed
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            var_confidence: Value at Risk confidence level
            correlation_threshold: Maximum correlation between positions
        """
        try:
            validate_positive_number(initial_capital, "initial_capital")
            validate_percentage(max_position_size * 100, "max_position_size")
            validate_percentage(max_portfolio_risk * 100, "max_portfolio_risk")
            validate_percentage(max_drawdown * 100, "max_drawdown")
            
            self.initial_capital = initial_capital
            self.max_position_size = max_position_size
            self.max_portfolio_risk = max_portfolio_risk
            self.max_drawdown = max_drawdown
            self.max_leverage = max_leverage
            self.stop_loss_pct = stop_loss_pct
            self.take_profit_pct = take_profit_pct
            self.var_confidence = var_confidence
            self.correlation_threshold = correlation_threshold
            
            self.positions = {}
            self.historical_values = []
            self.peak_value = initial_capital
            self.daily_pnl = []
            
            log.info(f"AdvancedRiskManager initialized with capital=${initial_capital:,.2f}")
            
        except Exception as e:
            log.error(f"Error initializing AdvancedRiskManager: {str(e)}", exc_info=True)
            raise ValidationError(f"Failed to initialize risk manager: {str(e)}") from e
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        portfolio_value: float,
        volatility: float,
        confidence: float = 0.8
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion and risk constraints.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            portfolio_value: Current portfolio value
            volatility: Asset volatility
            confidence: Confidence in the trade (0-1)
            
        Returns:
            Optimal position size (number of shares)
        """
        try:
            validate_positive_number(entry_price, "entry_price")
            validate_positive_number(portfolio_value, "portfolio_value")
            validate_positive_number(volatility, "volatility")
            
            win_prob = confidence
            loss_prob = 1 - confidence
            win_loss_ratio = self.take_profit_pct / self.stop_loss_pct
            
            kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% Kelly
            
            risk_amount = portfolio_value * self.max_portfolio_risk
            position_risk = entry_price * self.stop_loss_pct
            risk_based_size = risk_amount / position_risk if position_risk > 0 else 0
            
            target_volatility = 0.15  # 15% annual volatility target
            vol_adjustment = target_volatility / volatility if volatility > 0 else 1.0
            vol_adjusted_size = (portfolio_value * self.max_position_size / entry_price) * vol_adjustment
            
            max_size_by_capital = portfolio_value * self.max_position_size / entry_price
            kelly_size = portfolio_value * kelly_fraction / entry_price
            
            position_size = min(
                max_size_by_capital,
                risk_based_size,
                vol_adjusted_size,
                kelly_size
            )
            
            position_size = max(0, position_size)
            
            log.info(f"Calculated position size for {symbol}: {position_size:.2f} shares "
                    f"(Kelly: {kelly_size:.2f}, Risk: {risk_based_size:.2f}, "
                    f"Vol: {vol_adjusted_size:.2f}, Max: {max_size_by_capital:.2f})")
            
            return position_size
            
        except Exception as e:
            log.error(f"Error calculating position size: {str(e)}", exc_info=True)
            return 0.0
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        position_value: float = 1.0
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            position_value: Position value
            
        Returns:
            VaR value
        """
        try:
            if len(returns) < 10:
                return 0.0
            
            var_percentile = (1 - confidence) * 100
            var = np.percentile(returns, var_percentile)
            var_amount = abs(var * position_value)
            
            return var_amount
            
        except Exception as e:
            log.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            log.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sortino ratio
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_std = np.std(downside_returns)
            sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
            
            return sortino
            
        except Exception as e:
            log.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def calculate_max_drawdown(self, values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            values: Historical portfolio values
            
        Returns:
            Maximum drawdown as fraction
        """
        try:
            if len(values) < 2:
                return 0.0
            
            cumulative_max = np.maximum.accumulate(values)
            drawdowns = (values - cumulative_max) / cumulative_max
            max_dd = abs(np.min(drawdowns))
            
            return max_dd
            
        except Exception as e:
            log.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def assess_position_risk(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        historical_prices: pd.Series
    ) -> PositionRisk:
        """
        Assess risk for a single position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price
            current_price: Current price
            historical_prices: Historical price series
            
        Returns:
            PositionRisk object
        """
        try:
            position_value = quantity * current_price
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            returns = historical_prices.pct_change().dropna().values
            
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            var_95 = self.calculate_var(returns, 0.95, position_value)
            var_99 = self.calculate_var(returns, 0.99, position_value)
            sharpe = self.calculate_sharpe_ratio(returns)
            max_dd = self.calculate_max_drawdown(historical_prices.values)
            
            if max_dd > 0.20 or volatility > 0.40 or unrealized_pnl_pct < -0.10:
                risk_level = RiskLevel.CRITICAL
            elif max_dd > 0.15 or volatility > 0.30 or unrealized_pnl_pct < -0.05:
                risk_level = RiskLevel.HIGH
            elif max_dd > 0.10 or volatility > 0.20:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            return PositionRisk(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                position_value=position_value,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                volatility=volatility,
                risk_level=risk_level
            )
            
        except Exception as e:
            log.error(f"Error assessing position risk for {symbol}: {str(e)}", exc_info=True)
            return PositionRisk(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                position_value=0,
                var_95=0,
                var_99=0,
                sharpe_ratio=0,
                max_drawdown=0,
                volatility=0,
                risk_level=RiskLevel.MEDIUM
            )
    
    def assess_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        cash: float,
        historical_values: List[float]
    ) -> PortfolioRisk:
        """
        Assess overall portfolio risk.
        
        Args:
            positions: Dictionary of positions
            cash: Available cash
            historical_values: Historical portfolio values
            
        Returns:
            PortfolioRisk object
        """
        try:
            total_position_value = sum(p['value'] for p in positions.values())
            total_value = total_position_value + cash
            leverage = total_position_value / total_value if total_value > 0 else 0
            
            if len(historical_values) > 1:
                values_array = np.array(historical_values)
                returns = np.diff(values_array) / values_array[:-1]
            else:
                returns = np.array([])
            
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            var_95 = self.calculate_var(returns, 0.95, total_value)
            var_99 = self.calculate_var(returns, 0.99, total_value)
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
            max_dd = self.calculate_max_drawdown(np.array(historical_values)) if historical_values else 0
            
            if total_position_value > 0:
                weights = [p['value'] / total_position_value for p in positions.values()]
                concentration_risk = sum(w**2 for w in weights)
            else:
                concentration_risk = 0
            
            correlation_risk = min(len(positions) / 10, 1.0)  # Simplified estimate
            
            beta = 1.0
            
            if max_dd > self.max_drawdown or leverage > self.max_leverage * 1.2:
                risk_level = RiskLevel.CRITICAL
            elif max_dd > self.max_drawdown * 0.8 or leverage > self.max_leverage:
                risk_level = RiskLevel.HIGH
            elif max_dd > self.max_drawdown * 0.5 or volatility > 0.25:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            return PortfolioRisk(
                total_value=total_value,
                total_exposure=total_position_value,
                cash=cash,
                leverage=leverage,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                volatility=volatility,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                risk_level=risk_level
            )
            
        except Exception as e:
            log.error(f"Error assessing portfolio risk: {str(e)}", exc_info=True)
            return PortfolioRisk(
                total_value=cash,
                total_exposure=0,
                cash=cash,
                leverage=0,
                var_95=0,
                var_99=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                volatility=0,
                beta=1.0,
                correlation_risk=0,
                concentration_risk=0,
                risk_level=RiskLevel.LOW
            )
    
    def should_stop_loss(self, position_risk: PositionRisk) -> bool:
        """
        Determine if a position should be stopped out.
        
        Args:
            position_risk: Position risk metrics
            
        Returns:
            True if position should be closed
        """
        if position_risk.unrealized_pnl_pct <= -self.stop_loss_pct:
            log.warning(f"Stop loss triggered for {position_risk.symbol}: "
                       f"{position_risk.unrealized_pnl_pct:.2%}")
            return True
        
        if position_risk.risk_level == RiskLevel.CRITICAL:
            log.warning(f"Critical risk level for {position_risk.symbol}, closing position")
            return True
        
        if position_risk.max_drawdown > 0.25:
            log.warning(f"Excessive drawdown for {position_risk.symbol}: "
                       f"{position_risk.max_drawdown:.2%}")
            return True
        
        return False
    
    def should_take_profit(self, position_risk: PositionRisk) -> bool:
        """
        Determine if profits should be taken.
        
        Args:
            position_risk: Position risk metrics
            
        Returns:
            True if profits should be taken
        """
        if position_risk.unrealized_pnl_pct >= self.take_profit_pct:
            log.info(f"Take profit triggered for {position_risk.symbol}: "
                    f"{position_risk.unrealized_pnl_pct:.2%}")
            return True
        
        return False
    
    def can_open_position(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        portfolio_risk: PortfolioRisk
    ) -> Tuple[bool, str]:
        """
        Determine if a new position can be opened.
        
        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            entry_price: Entry price
            portfolio_risk: Current portfolio risk
            
        Returns:
            Tuple of (can_open, reason)
        """
        position_value = position_size * entry_price
        
        new_exposure = portfolio_risk.total_exposure + position_value
        new_leverage = new_exposure / portfolio_risk.total_value if portfolio_risk.total_value > 0 else 0
        
        if new_leverage > self.max_leverage:
            return False, f"Leverage limit exceeded: {new_leverage:.2f} > {self.max_leverage:.2f}"
        
        position_pct = position_value / portfolio_risk.total_value if portfolio_risk.total_value > 0 else 0
        if position_pct > self.max_position_size:
            return False, f"Position size too large: {position_pct:.2%} > {self.max_position_size:.2%}"
        
        if portfolio_risk.risk_level == RiskLevel.CRITICAL:
            return False, "Portfolio risk level is critical"
        
        if portfolio_risk.max_drawdown > self.max_drawdown:
            return False, f"Max drawdown exceeded: {portfolio_risk.max_drawdown:.2%}"
        
        if position_value > portfolio_risk.cash:
            return False, f"Insufficient cash: ${position_value:,.2f} > ${portfolio_risk.cash:,.2f}"
        
        return True, "Position approved"
    
    def get_risk_report(
        self,
        positions: Dict[str, PositionRisk],
        portfolio_risk: PortfolioRisk
    ) -> str:
        """
        Generate a comprehensive risk report.
        
        Args:
            positions: Dictionary of position risks
            portfolio_risk: Portfolio risk metrics
            
        Returns:
            Formatted risk report string
        """
        report = []
        report.append("=" * 80)
        report.append("RISK MANAGEMENT REPORT")
        report.append("=" * 80)
        
        report.append(f"\nPortfolio Value: ${portfolio_risk.total_value:,.2f}")
        report.append(f"Cash: ${portfolio_risk.cash:,.2f}")
        report.append(f"Exposure: ${portfolio_risk.total_exposure:,.2f}")
        report.append(f"Leverage: {portfolio_risk.leverage:.2f}x")
        report.append(f"Risk Level: {portfolio_risk.risk_level.value.upper()}")
        
        report.append(f"\nRisk Metrics:")
        report.append(f"  VaR (95%): ${portfolio_risk.var_95:,.2f}")
        report.append(f"  VaR (99%): ${portfolio_risk.var_99:,.2f}")
        report.append(f"  Sharpe Ratio: {portfolio_risk.sharpe_ratio:.2f}")
        report.append(f"  Sortino Ratio: {portfolio_risk.sortino_ratio:.2f}")
        report.append(f"  Max Drawdown: {portfolio_risk.max_drawdown:.2%}")
        report.append(f"  Volatility: {portfolio_risk.volatility:.2%}")
        report.append(f"  Concentration Risk: {portfolio_risk.concentration_risk:.2f}")
        
        if positions:
            report.append(f"\nPosition Risks:")
            for symbol, pos_risk in positions.items():
                report.append(f"\n  {symbol}:")
                report.append(f"    Value: ${pos_risk.position_value:,.2f}")
                report.append(f"    P&L: ${pos_risk.unrealized_pnl:,.2f} ({pos_risk.unrealized_pnl_pct:.2%})")
                report.append(f"    Risk Level: {pos_risk.risk_level.value}")
                report.append(f"    VaR (95%): ${pos_risk.var_95:,.2f}")
                report.append(f"    Volatility: {pos_risk.volatility:.2%}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
