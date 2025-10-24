import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from src.utils import log, config

class Portfolio:
    
    def __init__(self, initial_capital: float = None):
        if initial_capital is None:
            initial_capital = config.get('trading.initial_capital', 100000.0)
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.closed_positions = []
        self.equity_curve = [initial_capital]
        self.equity_dates = [datetime.now()]
        self.trade_history = []
        
        log.info(f"Portfolio initialized with ${initial_capital:,.2f}")
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        positions_value = sum(
            pos['shares'] * current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions
    
    def open_position(
        self, 
        symbol: str, 
        shares: int, 
        price: float, 
        timestamp: datetime = None,
        signal_type: str = 'technical'
    ) -> bool:
        if timestamp is None:
            timestamp = datetime.now()
        
        slippage = config.get('backtest.slippage', 0.0005)
        actual_price = price * (1 + slippage)
        
        cost = shares * actual_price
        commission = cost * config.get('backtest.commission', 0.001)
        total_cost = cost + commission
        
        if total_cost > self.cash:
            log.warning(f"Insufficient cash for {symbol}: need ${total_cost:,.2f}, have ${self.cash:,.2f}")
            return False
        
        if symbol in self.positions:
            log.warning(f"Position already exists for {symbol}")
            return False
        
        self.cash -= total_cost
        
        self.positions[symbol] = {
            'symbol': symbol,
            'shares': shares,
            'entry_price': actual_price,
            'entry_date': timestamp,
            'current_price': actual_price,
            'signal_type': signal_type,
            'commission_paid': commission,
            'slippage_cost': shares * price * slippage
        }
        
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'quoted_price': price,
            'actual_price': actual_price,
            'slippage': slippage,
            'commission': commission,
            'total_cost': total_cost
        })
        
        log.info(f"Opened position: {symbol} - {shares} shares @ ${actual_price:.2f} (slippage: ${actual_price-price:.4f})")
        return True
    
    def close_position(
        self, 
        symbol: str, 
        price: float, 
        timestamp: datetime = None,
        reason: str = 'signal'
    ) -> bool:
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.positions:
            log.warning(f"No position to close for {symbol}")
            return False
        
        position = self.positions[symbol]
        shares = position['shares']
        entry_price = position['entry_price']
        
        slippage = config.get('backtest.slippage', 0.0005)
        actual_price = price * (1 - slippage)
        
        proceeds = shares * actual_price
        commission = proceeds * config.get('backtest.commission', 0.001)
        net_proceeds = proceeds - commission
        
        self.cash += net_proceeds
        
        pnl = net_proceeds - (shares * entry_price)
        pnl_pct = (actual_price - entry_price) / entry_price
        
        holding_period = (timestamp - position['entry_date']).days
        
        total_slippage_cost = position.get('slippage_cost', 0) + (shares * price * slippage)
        total_commission_cost = position.get('commission_paid', 0) + commission
        
        closed_position = {
            **position,
            'exit_price': actual_price,
            'exit_date': timestamp,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_period': holding_period,
            'exit_commission': commission,
            'total_slippage_cost': total_slippage_cost,
            'total_commission_cost': total_commission_cost,
            'reason': reason
        }
        
        self.closed_positions.append(closed_position)
        
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'quoted_price': price,
            'actual_price': actual_price,
            'slippage': slippage,
            'commission': commission,
            'net_proceeds': net_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        del self.positions[symbol]
        
        log.info(f"Closed position: {symbol} - PnL: ${pnl:.2f} ({pnl_pct:.2%}) - Reason: {reason} (slippage: ${price-actual_price:.4f})")
        return True
    
    def update_positions(self, current_prices: Dict[str, float], timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position['current_price'] = current_prices[symbol]
        
        total_value = self.get_total_value(current_prices)
        self.equity_curve.append(total_value)
        self.equity_dates.append(timestamp)
    
    def get_open_positions(self) -> List[Dict]:
        return list(self.positions.values())
    
    def get_positions_list(self) -> List[Dict]:
        positions_list = []
        for symbol, pos in self.positions.items():
            current_value = pos['shares'] * pos['current_price']
            cost_basis = pos['shares'] * pos['entry_price']
            unrealized_pnl = current_value - cost_basis
            unrealized_pnl_pct = (pos['current_price'] - pos['entry_price']) / pos['entry_price']
            
            positions_list.append({
                'symbol': symbol,
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'cost_basis': cost_basis,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'entry_date': pos['entry_date']
            })
        
        return positions_list
    
    def get_equity_curve(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'Date': self.equity_dates,
            'Equity': self.equity_curve
        })
        return df
    
    def get_trade_history(self) -> pd.DataFrame:
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_closed_positions(self) -> pd.DataFrame:
        if not self.closed_positions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.closed_positions)
    
    def calculate_metrics(self) -> Dict:
        if len(self.equity_curve) < 2:
            return {}
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        
        if len(returns) > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            downside_returns = returns[returns < 0]
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        if self.closed_positions:
            winning_trades = [p for p in self.closed_positions if p['pnl'] > 0]
            losing_trades = [p for p in self.closed_positions if p['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(self.closed_positions)
            
            avg_win = np.mean([p['pnl'] for p in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(p['pnl']) for p in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(p['pnl'] for p in winning_trades) / sum(p['pnl'] for p in losing_trades)) if losing_trades and sum(p['pnl'] for p in losing_trades) != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.closed_positions),
            'winning_trades': len([p for p in self.closed_positions if p['pnl'] > 0]),
            'losing_trades': len([p for p in self.closed_positions if p['pnl'] <= 0]),
            'current_equity': equity_series.iloc[-1],
            'cash': self.cash,
            'num_open_positions': len(self.positions)
        }
        
        return metrics
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.closed_positions = []
        self.equity_curve = [self.initial_capital]
        self.equity_dates = [datetime.now()]
        self.trade_history = []
        log.info("Portfolio reset")
