"""
Execution Delay Modeling for Realistic Backtesting

This module simulates unmodeled delays in live trading including:
- Order execution latencies (100-450ms typical)
- Network lag and broker processing time
- Variable delays based on market conditions
- Slippage integration with delays
- Partial fill modeling

These delays can reduce simulated returns by 10-50% in high-frequency strategies,
making backtests more realistic and preventing overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)


@dataclass
class DelayConfig:
    """Configuration for execution delay modeling"""
    
    fixed_delay_ms: float = 200.0
    
    use_variable_delay: bool = True
    delay_mean_ms: float = 200.0
    delay_std_ms: float = 50.0
    delay_distribution: Literal["normal", "poisson", "uniform"] = "normal"
    
    integrate_slippage: bool = True
    slippage_bps: float = 10.0  # 10 basis points = 0.1%
    slippage_std_bps: float = 5.0
    
    model_partial_fills: bool = True
    min_fill_rate: float = 0.8  # 80% minimum fill
    liquidity_threshold: float = 1000000.0  # $1M minimum volume
    
    regime_based_delays: bool = True
    high_volatility_multiplier: float = 1.5
    low_liquidity_multiplier: float = 2.0
    
    data_feed_delay_ms: float = 100.0
    broker_processing_ms: float = 50.0
    network_jitter_ms: float = 20.0


class ExecutionDelaySimulator:
    """
    Simulates realistic execution delays for backtesting
    
    Models multiple types of delays:
    1. Fixed delay: Constant lag (e.g., 200ms)
    2. Variable delay: Stochastic lag with distribution
    3. Slippage: Price impact from delays
    4. Partial fills: Not all orders fill 100%
    5. Regime-based: Higher delays in volatile/illiquid markets
    """
    
    def __init__(self, config: Optional[DelayConfig] = None):
        self.config = config or DelayConfig()
        self.execution_history: List[Dict] = []
        
    def simulate_order_execution(
        self,
        signal_time: pd.Timestamp,
        signal_price: float,
        order_type: Literal["buy", "sell"],
        order_size: float,
        market_data: pd.DataFrame,
        market_regime: Optional[Dict] = None
    ) -> Dict:
        """
        Simulate realistic order execution with delays
        
        Args:
            signal_time: Time when signal was generated
            signal_price: Price when signal was generated
            order_type: "buy" or "sell"
            order_size: Size of order (shares or contracts)
            market_data: Historical price data with columns [timestamp, price, volume, ...]
            market_regime: Optional dict with keys [volatility, liquidity, regime_type]
            
        Returns:
            Dict with execution details:
            {
                'signal_time': original signal timestamp,
                'execution_time': actual execution timestamp,
                'signal_price': price at signal time,
                'execution_price': actual execution price (with slippage),
                'delay_ms': total delay in milliseconds,
                'fill_rate': percentage of order filled (0.0-1.0),
                'slippage_bps': slippage in basis points,
                'order_type': 'buy' or 'sell',
                'order_size': original order size,
                'filled_size': actual filled size
            }
        """
        
        total_delay_ms = self._calculate_total_delay(market_regime)
        
        execution_time, execution_price = self._find_execution_point(
            signal_time, total_delay_ms, market_data
        )
        
        slippage_bps = 0.0
        if self.config.integrate_slippage:
            slippage_bps = self._calculate_slippage(
                signal_price, execution_price, order_type, market_regime
            )
            execution_price = self._apply_slippage(
                execution_price, slippage_bps, order_type
            )
        
        fill_rate = 1.0
        if self.config.model_partial_fills:
            fill_rate = self._calculate_fill_rate(
                order_size, market_data, execution_time, market_regime
            )
        
        filled_size = order_size * fill_rate
        
        execution_result = {
            'signal_time': signal_time,
            'execution_time': execution_time,
            'signal_price': signal_price,
            'execution_price': execution_price,
            'delay_ms': total_delay_ms,
            'fill_rate': fill_rate,
            'slippage_bps': slippage_bps,
            'order_type': order_type,
            'order_size': order_size,
            'filled_size': filled_size,
            'regime': market_regime
        }
        
        self.execution_history.append(execution_result)
        
        log.debug(
            f"Executed {order_type} order: "
            f"delay={total_delay_ms:.1f}ms, "
            f"slippage={slippage_bps:.2f}bps, "
            f"fill_rate={fill_rate:.2%}"
        )
        
        return execution_result
    
    def _calculate_total_delay(self, market_regime: Optional[Dict] = None) -> float:
        """Calculate total execution delay in milliseconds"""
        
        if self.config.use_variable_delay:
            if self.config.delay_distribution == "normal":
                delay = np.random.normal(
                    self.config.delay_mean_ms,
                    self.config.delay_std_ms
                )
            elif self.config.delay_distribution == "poisson":
                delay = np.random.poisson(self.config.delay_mean_ms)
            elif self.config.delay_distribution == "uniform":
                delay = np.random.uniform(
                    self.config.delay_mean_ms - self.config.delay_std_ms,
                    self.config.delay_mean_ms + self.config.delay_std_ms
                )
            else:
                delay = self.config.delay_mean_ms
        else:
            delay = self.config.fixed_delay_ms
        
        delay += self.config.data_feed_delay_ms
        delay += self.config.broker_processing_ms
        delay += np.random.normal(0, self.config.network_jitter_ms)
        
        if self.config.regime_based_delays and market_regime:
            if market_regime.get('high_volatility', False):
                delay *= self.config.high_volatility_multiplier
            if market_regime.get('low_liquidity', False):
                delay *= self.config.low_liquidity_multiplier
        
        return max(0, delay)  # Ensure non-negative
    
    def _find_execution_point(
        self,
        signal_time: pd.Timestamp,
        delay_ms: float,
        market_data: pd.DataFrame
    ) -> Tuple[pd.Timestamp, float]:
        """Find the actual execution time and price after delay"""
        
        execution_time = signal_time + timedelta(milliseconds=delay_ms)
        
        future_data = market_data[market_data['timestamp'] >= execution_time]
        
        if len(future_data) == 0:
            execution_price = market_data.iloc[-1]['price']
            execution_time = market_data.iloc[-1]['timestamp']
            log.warning(
                f"No data available after delay for {signal_time}, "
                f"using last known price"
            )
        else:
            execution_price = future_data.iloc[0]['price']
            execution_time = future_data.iloc[0]['timestamp']
        
        return execution_time, execution_price
    
    def _calculate_slippage(
        self,
        signal_price: float,
        execution_price: float,
        order_type: str,
        market_regime: Optional[Dict] = None
    ) -> float:
        """Calculate slippage in basis points"""
        
        slippage_bps = np.random.normal(
            self.config.slippage_bps,
            self.config.slippage_std_bps
        )
        
        price_change_pct = abs(execution_price - signal_price) / signal_price
        market_impact_bps = price_change_pct * 10000  # Convert to bps
        
        if market_regime:
            if market_regime.get('high_volatility', False):
                slippage_bps *= 1.5
            if market_regime.get('low_liquidity', False):
                slippage_bps *= 2.0
        
        return abs(slippage_bps + market_impact_bps)
    
    def _apply_slippage(
        self,
        execution_price: float,
        slippage_bps: float,
        order_type: str
    ) -> float:
        """Apply slippage to execution price"""
        
        slippage_factor = slippage_bps / 10000  # Convert bps to decimal
        
        if order_type == "buy":
            return execution_price * (1 + slippage_factor)
        else:
            return execution_price * (1 - slippage_factor)
    
    def _calculate_fill_rate(
        self,
        order_size: float,
        market_data: pd.DataFrame,
        execution_time: pd.Timestamp,
        market_regime: Optional[Dict] = None
    ) -> float:
        """Calculate percentage of order that gets filled"""
        
        execution_data = market_data[market_data['timestamp'] == execution_time]
        
        if len(execution_data) == 0 or 'volume' not in execution_data.columns:
            return 1.0
        
        market_volume = execution_data.iloc[0]['volume']
        
        max_fillable = market_volume * 0.1
        
        if order_size <= max_fillable:
            fill_rate = 1.0
        else:
            fill_rate = max_fillable / order_size
        
        fill_rate = max(fill_rate, self.config.min_fill_rate)
        
        if market_regime:
            if market_regime.get('low_liquidity', False):
                fill_rate *= 0.8  # 20% reduction in low liquidity
        
        return min(1.0, fill_rate)
    
    def get_execution_statistics(self) -> Dict:
        """Get statistics on execution history"""
        
        if not self.execution_history:
            return {}
        
        delays = [e['delay_ms'] for e in self.execution_history]
        slippages = [e['slippage_bps'] for e in self.execution_history]
        fill_rates = [e['fill_rate'] for e in self.execution_history]
        
        return {
            'total_executions': len(self.execution_history),
            'avg_delay_ms': np.mean(delays),
            'std_delay_ms': np.std(delays),
            'max_delay_ms': np.max(delays),
            'avg_slippage_bps': np.mean(slippages),
            'std_slippage_bps': np.std(slippages),
            'avg_fill_rate': np.mean(fill_rates),
            'partial_fills': sum(1 for f in fill_rates if f < 1.0)
        }
    
    def reset(self):
        """Reset execution history"""
        self.execution_history = []


class DelayAwareBacktester:
    """
    Backtest engine wrapper that integrates execution delays
    
    Usage:
        backtester = DelayAwareBacktester(delay_config)
        results = backtester.run_backtest(signals, market_data)
    """
    
    def __init__(self, delay_config: Optional[DelayConfig] = None):
        self.delay_simulator = ExecutionDelaySimulator(delay_config)
        
    def apply_delays_to_signals(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
        market_regimes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Apply execution delays to trading signals
        
        Args:
            signals: DataFrame with columns [timestamp, signal, price, size]
                     signal: 1 for buy, -1 for sell, 0 for hold
            market_data: DataFrame with columns [timestamp, price, volume]
            market_regimes: Optional DataFrame with regime information
            
        Returns:
            DataFrame with delayed executions
        """
        
        delayed_executions = []
        
        for idx, row in signals.iterrows():
            if row['signal'] == 0:
                continue  # Skip hold signals
            
            order_type = "buy" if row['signal'] > 0 else "sell"
            
            regime = None
            if market_regimes is not None:
                regime_data = market_regimes[
                    market_regimes['timestamp'] == row['timestamp']
                ]
                if len(regime_data) > 0:
                    regime = regime_data.iloc[0].to_dict()
            
            execution = self.delay_simulator.simulate_order_execution(
                signal_time=row['timestamp'],
                signal_price=row['price'],
                order_type=order_type,
                order_size=row.get('size', 100),
                market_data=market_data,
                market_regime=regime
            )
            
            delayed_executions.append(execution)
        
        return pd.DataFrame(delayed_executions)
    
    def calculate_pnl_with_delays(
        self,
        delayed_executions: pd.DataFrame
    ) -> Dict:
        """Calculate P&L accounting for execution delays"""
        
        total_pnl = 0.0
        trades = []
        
        buys = delayed_executions[delayed_executions['order_type'] == 'buy']
        sells = delayed_executions[delayed_executions['order_type'] == 'sell']
        
        for buy_idx, buy in buys.iterrows():
            future_sells = sells[sells['execution_time'] > buy['execution_time']]
            
            if len(future_sells) == 0:
                continue
            
            sell = future_sells.iloc[0]
            
            buy_cost = buy['execution_price'] * buy['filled_size']
            sell_revenue = sell['execution_price'] * sell['filled_size']
            
            trade_size = min(buy['filled_size'], sell['filled_size'])
            trade_pnl = (sell['execution_price'] - buy['execution_price']) * trade_size
            
            total_pnl += trade_pnl
            
            trades.append({
                'buy_time': buy['execution_time'],
                'sell_time': sell['execution_time'],
                'buy_price': buy['execution_price'],
                'sell_price': sell['execution_price'],
                'size': trade_size,
                'pnl': trade_pnl,
                'buy_delay_ms': buy['delay_ms'],
                'sell_delay_ms': sell['delay_ms'],
                'buy_slippage_bps': buy['slippage_bps'],
                'sell_slippage_bps': sell['slippage_bps']
            })
        
        return {
            'total_pnl': total_pnl,
            'num_trades': len(trades),
            'trades': trades,
            'execution_stats': self.delay_simulator.get_execution_statistics()
        }
    
    def run_monte_carlo_delay_test(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
        num_iterations: int = 1000,
        market_regimes: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run Monte Carlo simulation with varying delays
        
        Tests robustness of strategy to execution delay variability
        """
        
        log.info(f"Running Monte Carlo delay test with {num_iterations} iterations...")
        
        pnl_results = []
        delay_stats = []
        
        for i in range(num_iterations):
            self.delay_simulator.reset()
            
            delayed_executions = self.apply_delays_to_signals(
                signals, market_data, market_regimes
            )
            
            result = self.calculate_pnl_with_delays(delayed_executions)
            pnl_results.append(result['total_pnl'])
            delay_stats.append(result['execution_stats'])
            
            if (i + 1) % 100 == 0:
                log.info(f"Completed {i + 1}/{num_iterations} iterations")
        
        return {
            'pnl_mean': np.mean(pnl_results),
            'pnl_std': np.std(pnl_results),
            'pnl_min': np.min(pnl_results),
            'pnl_max': np.max(pnl_results),
            'pnl_percentiles': {
                '5th': np.percentile(pnl_results, 5),
                '25th': np.percentile(pnl_results, 25),
                '50th': np.percentile(pnl_results, 50),
                '75th': np.percentile(pnl_results, 75),
                '95th': np.percentile(pnl_results, 95)
            },
            'avg_delay_ms': np.mean([s['avg_delay_ms'] for s in delay_stats]),
            'avg_slippage_bps': np.mean([s['avg_slippage_bps'] for s in delay_stats]),
            'avg_fill_rate': np.mean([s['avg_fill_rate'] for s in delay_stats])
        }


def example_usage():
    """Example of how to use the execution delay simulator"""
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    volumes = np.random.randint(10000, 100000, 100)
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volumes
    })
    
    signals = pd.DataFrame({
        'timestamp': [dates[10], dates[50]],
        'signal': [1, -1],  # Buy at t=10, sell at t=50
        'price': [prices[10], prices[50]],
        'size': [100, 100]
    })
    
    config = DelayConfig(
        use_variable_delay=True,
        delay_mean_ms=200,
        delay_std_ms=50,
        integrate_slippage=True,
        slippage_bps=10,
        model_partial_fills=True
    )
    
    backtester = DelayAwareBacktester(config)
    delayed_executions = backtester.apply_delays_to_signals(signals, market_data)
    
    print("Delayed Executions:")
    print(delayed_executions)
    
    result = backtester.calculate_pnl_with_delays(delayed_executions)
    print(f"\nTotal P&L: ${result['total_pnl']:.2f}")
    print(f"Number of trades: {result['num_trades']}")
    print(f"Execution stats: {result['execution_stats']}")
    
    mc_result = backtester.run_monte_carlo_delay_test(
        signals, market_data, num_iterations=100
    )
    print(f"\nMonte Carlo Results (100 iterations):")
    print(f"Mean P&L: ${mc_result['pnl_mean']:.2f}")
    print(f"Std P&L: ${mc_result['pnl_std']:.2f}")
    print(f"5th percentile: ${mc_result['pnl_percentiles']['5th']:.2f}")
    print(f"95th percentile: ${mc_result['pnl_percentiles']['95th']:.2f}")


if __name__ == "__main__":
    example_usage()
