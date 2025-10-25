"""
Dynamic Fee/Slippage Modeling - Feature 15

Implements realistic cost modeling for trading:
1. Liquidity-based cost calculation
2. Market impact modeling
3. Dynamic slippage based on order size
4. Time-of-day adjustments
5. Volatility-based slippage

Target: +10-20% net returns accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, time
from src.utils import log, config

class FeeSlippageModel:
    """
    Dynamic Fee and Slippage Model
    
    Calculates realistic trading costs based on:
    - Order size relative to average volume
    - Market volatility
    - Time of day (market open/close have higher slippage)
    - Liquidity (bid-ask spread)
    - Market impact
    
    This prevents overestimation of returns in backtesting.
    """
    
    def __init__(self):
        """Initialize fee/slippage model"""
        self.config = config.get('trading.fee_slippage', {
            'base_commission_pct': 0.001,  # 0.1% base commission
            'min_commission': 1.0,          # $1 minimum
            'base_slippage_bps': 5,         # 5 basis points base slippage
            'market_impact_factor': 0.1,    # Market impact coefficient
            'volatility_multiplier': 2.0,   # Volatility impact on slippage
            'time_of_day_adjustments': {
                'market_open': 1.5,         # 50% higher at open
                'market_close': 1.3,        # 30% higher at close
                'midday': 1.0               # Normal during midday
            }
        })
        
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        log.info("FeeSlippageModel initialized")
        log.info(f"Base commission: {self.config['base_commission_pct']:.3%}")
        log.info(f"Base slippage: {self.config['base_slippage_bps']} bps")
    
    def calculate_total_cost(
        self,
        symbol: str,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        current_time: datetime = None,
        bid_ask_spread: Optional[float] = None
    ) -> Dict:
        """
        Calculate total trading cost including fees and slippage
        
        Args:
            symbol: Trading symbol
            order_size: Number of shares
            price: Current price
            volume: Average daily volume
            volatility: Recent volatility (std dev of returns)
            current_time: Time of trade (for time-of-day adjustments)
            bid_ask_spread: Current bid-ask spread (optional)
            
        Returns:
            Dict with breakdown of costs
        """
        if current_time is None:
            current_time = datetime.now()
        
        commission = self._calculate_commission(order_size, price)
        
        slippage = self._calculate_slippage(
            order_size=order_size,
            price=price,
            volume=volume,
            volatility=volatility,
            current_time=current_time,
            bid_ask_spread=bid_ask_spread
        )
        
        market_impact = self._calculate_market_impact(
            order_size=order_size,
            price=price,
            volume=volume
        )
        
        total_cost = commission + slippage + market_impact
        total_cost_pct = total_cost / (order_size * price) if order_size * price > 0 else 0
        
        return {
            'commission': commission,
            'commission_pct': commission / (order_size * price) if order_size * price > 0 else 0,
            'slippage': slippage,
            'slippage_pct': slippage / (order_size * price) if order_size * price > 0 else 0,
            'market_impact': market_impact,
            'market_impact_pct': market_impact / (order_size * price) if order_size * price > 0 else 0,
            'total_cost': total_cost,
            'total_cost_pct': total_cost_pct,
            'effective_price': price * (1 + total_cost_pct)
        }
    
    def _calculate_commission(self, order_size: float, price: float) -> float:
        """Calculate commission fees"""
        notional_value = order_size * price
        commission_pct = self.config['base_commission_pct']
        commission = notional_value * commission_pct
        
        min_commission = self.config['min_commission']
        return max(commission, min_commission)
    
    def _calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        current_time: datetime,
        bid_ask_spread: Optional[float] = None
    ) -> float:
        """Calculate slippage cost"""
        base_slippage_bps = self.config['base_slippage_bps']
        
        volume_ratio = order_size / volume if volume > 0 else 0.1
        size_multiplier = 1 + (volume_ratio * 10)  # Larger orders have more slippage
        
        volatility_multiplier = 1 + (volatility * self.config['volatility_multiplier'])
        
        time_multiplier = self._get_time_of_day_multiplier(current_time)
        
        spread_multiplier = 1.0
        if bid_ask_spread is not None:
            spread_pct = bid_ask_spread / price if price > 0 else 0
            spread_multiplier = 1 + (spread_pct * 100)  # Convert to multiplier
        
        total_slippage_bps = (
            base_slippage_bps *
            size_multiplier *
            volatility_multiplier *
            time_multiplier *
            spread_multiplier
        )
        
        slippage = (order_size * price) * (total_slippage_bps / 10000)
        
        return slippage
    
    def _calculate_market_impact(
        self,
        order_size: float,
        price: float,
        volume: float
    ) -> float:
        """
        Calculate market impact cost
        
        Uses square root model: impact ∝ sqrt(order_size / volume)
        """
        if volume <= 0:
            return 0
        
        impact_factor = self.config['market_impact_factor']
        
        volume_ratio = order_size / volume
        impact_pct = impact_factor * np.sqrt(volume_ratio)
        
        market_impact = (order_size * price) * impact_pct
        
        return market_impact
    
    def _get_time_of_day_multiplier(self, current_time: datetime) -> float:
        """Get time-of-day multiplier for slippage"""
        current_time_only = current_time.time()
        
        if self.market_open <= current_time_only < time(10, 30):
            return self.config['time_of_day_adjustments']['market_open']
        
        elif time(15, 0) <= current_time_only <= self.market_close:
            return self.config['time_of_day_adjustments']['market_close']
        
        else:
            return self.config['time_of_day_adjustments']['midday']
    
    def estimate_execution_price(
        self,
        order_type: str,
        current_price: float,
        order_size: float,
        volume: float,
        volatility: float,
        current_time: datetime = None
    ) -> float:
        """
        Estimate actual execution price including all costs
        
        Args:
            order_type: 'BUY' or 'SELL'
            current_price: Current market price
            order_size: Number of shares
            volume: Average daily volume
            volatility: Recent volatility
            current_time: Time of trade
            
        Returns:
            Estimated execution price
        """
        costs = self.calculate_total_cost(
            symbol='',  # Not needed for estimation
            order_size=order_size,
            price=current_price,
            volume=volume,
            volatility=volatility,
            current_time=current_time
        )
        
        total_cost_pct = costs['total_cost_pct']
        
        if order_type == 'BUY':
            return current_price * (1 + total_cost_pct)
        else:  # SELL
            return current_price * (1 - total_cost_pct)
    
    def adjust_backtest_returns(
        self,
        trades: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Adjust backtest returns to account for realistic costs
        
        Args:
            trades: DataFrame with trade history
            market_data: Dict of symbol -> DataFrame with volume and volatility
            
        Returns:
            Adjusted trades DataFrame with realistic costs
        """
        adjusted_trades = trades.copy()
        
        for idx, trade in adjusted_trades.iterrows():
            symbol = trade['symbol']
            
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            trade_date = trade['date']
            
            closest_idx = df.index.get_indexer([trade_date], method='nearest')[0]
            market_row = df.iloc[closest_idx]
            
            costs = self.calculate_total_cost(
                symbol=symbol,
                order_size=trade['shares'],
                price=trade['price'],
                volume=market_row.get('Volume', 1000000),
                volatility=market_row.get('Volatility', 0.02),
                current_time=trade_date
            )
            
            adjusted_trades.at[idx, 'commission'] = costs['commission']
            adjusted_trades.at[idx, 'slippage'] = costs['slippage']
            adjusted_trades.at[idx, 'market_impact'] = costs['market_impact']
            adjusted_trades.at[idx, 'total_cost'] = costs['total_cost']
            adjusted_trades.at[idx, 'effective_price'] = costs['effective_price']
            
            if 'profit' in adjusted_trades.columns:
                adjusted_trades.at[idx, 'profit'] -= costs['total_cost']
        
        return adjusted_trades
    
    def get_cost_summary(self, trades: pd.DataFrame) -> Dict:
        """Get summary statistics of trading costs"""
        if len(trades) == 0:
            return {}
        
        return {
            'total_commission': trades['commission'].sum() if 'commission' in trades.columns else 0,
            'total_slippage': trades['slippage'].sum() if 'slippage' in trades.columns else 0,
            'total_market_impact': trades['market_impact'].sum() if 'market_impact' in trades.columns else 0,
            'total_costs': trades['total_cost'].sum() if 'total_cost' in trades.columns else 0,
            'avg_cost_per_trade': trades['total_cost'].mean() if 'total_cost' in trades.columns else 0,
            'avg_cost_pct': (trades['total_cost'] / (trades['shares'] * trades['price'])).mean() if 'total_cost' in trades.columns else 0,
            'num_trades': len(trades)
        }
