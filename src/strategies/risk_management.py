import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.utils import log, config

class RiskManager:
    
    def __init__(self, initial_capital: float = None):
        if initial_capital is None:
            initial_capital = config.get('trading.initial_capital', 100000.0)
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = config.get('trading.max_position_size', 0.10)
        self.stop_loss_pct = config.get('trading.stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('trading.take_profit_pct', 0.05)
        self.max_positions = config.get('trading.max_positions', 10)
        self.max_drawdown = config.get('risk.max_drawdown', 0.20)
        self.correlation_threshold = config.get('risk.correlation_threshold', 0.7)
        
        log.info(f"RiskManager initialized with capital: ${initial_capital:,.2f}")
    
    def calculate_position_size_fixed(self, price: float) -> int:
        max_investment = self.current_capital * self.max_position_size
        shares = int(max_investment / price)
        return max(shares, 0)
    
    def calculate_position_size_kelly(
        self, 
        price: float, 
        win_rate: float, 
        avg_win: float, 
        avg_loss: float
    ) -> int:
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self.calculate_position_size_fixed(price)
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        kelly_fraction = max(0, min(kelly_fraction, 1))
        
        kelly_fraction *= config.get('risk.kelly_fraction', 0.25)
        
        max_investment = self.current_capital * kelly_fraction
        shares = int(max_investment / price)
        
        max_shares = self.calculate_position_size_fixed(price)
        shares = min(shares, max_shares)
        
        return max(shares, 0)
    
    def calculate_position_size_volatility(
        self, 
        price: float, 
        volatility: float,
        target_risk: float = 0.01
    ) -> int:
        if volatility == 0:
            return self.calculate_position_size_fixed(price)
        
        risk_amount = self.current_capital * target_risk
        
        position_value = risk_amount / volatility
        
        shares = int(position_value / price)
        
        max_shares = self.calculate_position_size_fixed(price)
        shares = min(shares, max_shares)
        
        return max(shares, 0)
    
    def calculate_stop_loss(self, entry_price: float, position_type: str = 'long') -> float:
        if position_type == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, position_type: str = 'long') -> float:
        if position_type == 'long':
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        return take_profit
    
    def check_stop_loss(
        self, 
        current_price: float, 
        entry_price: float, 
        position_type: str = 'long'
    ) -> bool:
        stop_loss = self.calculate_stop_loss(entry_price, position_type)
        
        if position_type == 'long':
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss
    
    def check_take_profit(
        self, 
        current_price: float, 
        entry_price: float, 
        position_type: str = 'long'
    ) -> bool:
        take_profit = self.calculate_take_profit(entry_price, position_type)
        
        if position_type == 'long':
            return current_price >= take_profit
        else:
            return current_price <= take_profit
    
    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        if not positions:
            return {
                'total_exposure': 0.0,
                'num_positions': 0,
                'avg_position_size': 0.0,
                'max_position_exposure': 0.0
            }
        
        total_value = sum(p['shares'] * p['current_price'] for p in positions)
        exposures = [p['shares'] * p['current_price'] / self.current_capital for p in positions]
        
        return {
            'total_exposure': total_value / self.current_capital,
            'num_positions': len(positions),
            'avg_position_size': np.mean(exposures),
            'max_position_exposure': max(exposures) if exposures else 0.0
        }
    
    def check_correlation(self, returns_data: pd.DataFrame, symbols: List[str]) -> bool:
        if len(symbols) < 2:
            return True
        
        try:
            available_symbols = [s for s in symbols if s in returns_data.columns]
            
            if len(available_symbols) < 2:
                return True
            
            corr_matrix = returns_data[available_symbols].corr()
            
            for i in range(len(available_symbols)):
                for j in range(i+1, len(available_symbols)):
                    if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                        log.warning(f"High correlation detected between {available_symbols[i]} and {available_symbols[j]}: {corr_matrix.iloc[i, j]:.3f}")
                        return False
            
            return True
            
        except Exception as e:
            log.error(f"Error checking correlation: {str(e)}")
            return True
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[float, float]:
        cumulative_max = equity_curve.expanding().max()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
        
        return max_drawdown, current_drawdown
    
    def check_circuit_breaker(self, equity_curve: pd.Series) -> bool:
        if len(equity_curve) < 2:
            return False
        
        max_dd, current_dd = self.calculate_drawdown(equity_curve)
        
        if current_dd < -self.max_drawdown:
            log.warning(f"Circuit breaker triggered! Current drawdown: {current_dd:.2%}")
            return True
        
        daily_loss_limit = config.get('safety.max_daily_loss', 0.05)
        daily_return = (equity_curve.iloc[-1] - equity_curve.iloc[-2]) / equity_curve.iloc[-2]
        
        if daily_return < -daily_loss_limit:
            log.warning(f"Daily loss limit exceeded: {daily_return:.2%}")
            return True
        
        return False
    
    def diversification_check(self, positions: List[Dict]) -> bool:
        if len(positions) >= self.max_positions:
            log.warning(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        return True
    
    def validate_trade(
        self, 
        symbol: str, 
        price: float, 
        shares: int, 
        current_positions: List[Dict],
        equity_curve: pd.Series = None
    ) -> Tuple[bool, str]:
        if shares <= 0:
            return False, "Invalid share quantity"
        
        trade_value = price * shares
        if trade_value > self.current_capital * self.max_position_size:
            return False, f"Trade exceeds max position size ({self.max_position_size:.1%})"
        
        if not self.diversification_check(current_positions):
            return False, "Maximum positions reached"
        
        if equity_curve is not None and len(equity_curve) > 0:
            if self.check_circuit_breaker(equity_curve):
                return False, "Circuit breaker activated"
        
        return True, "Trade validated"
    
    def update_capital(self, new_capital: float):
        self.current_capital = new_capital
        log.info(f"Capital updated to: ${new_capital:,.2f}")
    
    def get_risk_metrics(self, positions: List[Dict], equity_curve: pd.Series) -> Dict:
        portfolio_risk = self.calculate_portfolio_risk(positions)
        
        max_dd, current_dd = self.calculate_drawdown(equity_curve) if len(equity_curve) > 0 else (0.0, 0.0)
        
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
        
        metrics = {
            'total_exposure': portfolio_risk['total_exposure'],
            'num_positions': portfolio_risk['num_positions'],
            'max_drawdown': max_dd,
            'current_drawdown': current_dd,
            'portfolio_volatility': volatility,
            'capital': self.current_capital,
            'available_capital': self.current_capital * (1 - portfolio_risk['total_exposure'])
        }
        
        return metrics
