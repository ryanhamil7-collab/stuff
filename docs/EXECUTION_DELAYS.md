# Execution Delay Modeling for Realistic Backtesting

## Overview

Execution delays are unmodeled latencies in live trading that can significantly impact profitability. These delays include order execution latencies (100-450ms typical), network lag, broker processing time, and market impact. Without modeling these delays, backtests can overestimate returns by 10-50%, especially in high-frequency strategies.

This module provides comprehensive execution delay modeling to make backtests more realistic and prevent overfitting.

## Why Model Execution Delays?

### Impact on Performance

Real-world trading involves multiple sources of delay:

1. **Data Feed Delay** (50-150ms): Time for market data to reach your system
2. **Signal Processing** (10-50ms): Time to compute indicators and generate signals
3. **Network Latency** (20-100ms): Time to send orders to broker
4. **Broker Processing** (50-200ms): Time for broker to process and route orders
5. **Exchange Execution** (10-100ms): Time for exchange to match and fill orders

**Total typical delay: 140-600ms** (mean ~200-300ms)

### Performance Impact

| Strategy Type | Delay Impact on Returns |
|--------------|------------------------|
| High-Frequency (< 1 min) | -30% to -50% |
| Intraday (5-60 min) | -15% to -30% |
| Daily | -5% to -15% |
| Weekly+ | -2% to -5% |

### Additional Effects

- **Slippage**: Price moves during delay, causing worse fills
- **Partial Fills**: Large orders may not fill completely in illiquid markets
- **Regime Sensitivity**: Delays increase during high volatility or low liquidity
- **Missed Opportunities**: Fast-moving markets may reverse before execution

## Features

### 1. Fixed Delay Simulation

Constant lag applied to all orders:

```python
from src.backtesting.execution_delays import DelayConfig, ExecutionDelaySimulator

config = DelayConfig(
    use_variable_delay=False,
    fixed_delay_ms=200.0  # 200ms constant delay
)

simulator = ExecutionDelaySimulator(config)
```

**Use case**: Simple baseline for understanding delay impact

### 2. Variable/Stochastic Delay

Random delays with statistical distribution:

```python
config = DelayConfig(
    use_variable_delay=True,
    delay_mean_ms=200.0,
    delay_std_ms=50.0,
    delay_distribution="normal"  # Options: normal, poisson, uniform
)
```

**Distributions**:
- **Normal**: Most realistic for network/processing delays
- **Poisson**: Good for queue-based systems (broker processing)
- **Uniform**: Conservative worst-case testing

### 3. Slippage Integration

Price impact from delays:

```python
config = DelayConfig(
    integrate_slippage=True,
    slippage_bps=10.0,  # 10 basis points = 0.1%
    slippage_std_bps=5.0  # Variability
)
```

**Slippage calculation**:
```
total_slippage = base_slippage + market_impact + regime_adjustment

For buy orders:  execution_price = price * (1 + slippage_factor)
For sell orders: execution_price = price * (1 - slippage_factor)
```

### 4. Partial Fill Modeling

Not all orders fill 100%:

```python
config = DelayConfig(
    model_partial_fills=True,
    min_fill_rate=0.8,  # 80% minimum fill
    liquidity_threshold=1000000.0  # $1M minimum volume
)
```

**Fill rate calculation**:
```
max_fillable = market_volume * 0.1  # Can fill up to 10% of volume
fill_rate = min(1.0, max_fillable / order_size)
fill_rate = max(fill_rate, min_fill_rate)
```

### 5. Market Regime-Based Delays

Delays increase in volatile/illiquid markets:

```python
config = DelayConfig(
    regime_based_delays=True,
    high_volatility_multiplier=1.5,  # 50% longer delays
    low_liquidity_multiplier=2.0  # 100% longer delays
)
```

**Regime detection**:
- **High volatility**: ATR > 2x average
- **Low liquidity**: Volume < 50% average
- **Combined effect**: Multipliers stack

### 6. Component Delays

Breakdown of delay sources:

```python
config = DelayConfig(
    data_feed_delay_ms=100.0,
    broker_processing_ms=50.0,
    network_jitter_ms=20.0
)

# Total delay = base_delay + data_feed + broker + jitter + regime_adjustments
```

## Usage Examples

### Basic Usage

```python
import pandas as pd
from src.backtesting.execution_delays import (
    ExecutionDelaySimulator,
    DelayConfig
)

# Configure delays
config = DelayConfig(
    use_variable_delay=True,
    delay_mean_ms=200.0,
    integrate_slippage=True,
    model_partial_fills=True
)

# Create simulator
simulator = ExecutionDelaySimulator(config)

# Simulate order execution
execution = simulator.simulate_order_execution(
    signal_time=pd.Timestamp('2024-01-01 10:00:00'),
    signal_price=100.0,
    order_type='buy',
    order_size=100,
    market_data=market_df,  # DataFrame with [timestamp, price, volume]
    market_regime={'high_volatility': False, 'low_liquidity': False}
)

print(f"Signal price: ${execution['signal_price']:.2f}")
print(f"Execution price: ${execution['execution_price']:.2f}")
print(f"Delay: {execution['delay_ms']:.1f}ms")
print(f"Slippage: {execution['slippage_bps']:.2f} bps")
print(f"Fill rate: {execution['fill_rate']:.2%}")
```

### Integration with Backtesting

```python
from src.backtesting.execution_delays import DelayAwareBacktester

# Create backtester with delays
backtester = DelayAwareBacktester(delay_config)

# Apply delays to signals
delayed_executions = backtester.apply_delays_to_signals(
    signals=signals_df,  # [timestamp, signal, price, size]
    market_data=market_df,  # [timestamp, price, volume]
    market_regimes=regimes_df  # Optional
)

# Calculate P&L with delays
result = backtester.calculate_pnl_with_delays(delayed_executions)

print(f"Total P&L: ${result['total_pnl']:,.2f}")
print(f"Number of trades: {result['num_trades']}")
print(f"Average delay: {result['execution_stats']['avg_delay_ms']:.1f}ms")
print(f"Average slippage: {result['execution_stats']['avg_slippage_bps']:.2f} bps")
```

### Monte Carlo Delay Testing

Test strategy robustness to delay variability:

```python
# Run 1000 simulations with varying delays
mc_result = backtester.run_monte_carlo_delay_test(
    signals=signals_df,
    market_data=market_df,
    num_iterations=1000
)

print(f"Mean P&L: ${mc_result['pnl_mean']:,.2f}")
print(f"Std P&L: ${mc_result['pnl_std']:,.2f}")
print(f"5th percentile: ${mc_result['pnl_percentiles']['5th']:,.2f}")
print(f"95th percentile: ${mc_result['pnl_percentiles']['95th']:,.2f}")
print(f"Average delay: {mc_result['avg_delay_ms']:.1f}ms")
```

### Using with BacktestEngine

```python
from src.backtesting.backtest_engine import BacktestEngine

# Enable execution delays in backtest
engine = BacktestEngine(
    initial_capital=100000,
    enable_execution_delays=True  # Loads config from config.yaml
)

# Or provide custom config
from src.backtesting.execution_delays import DelayConfig

custom_config = DelayConfig(
    delay_mean_ms=300.0,  # Higher delays
    slippage_bps=15.0  # More slippage
)

engine = BacktestEngine(
    initial_capital=100000,
    enable_execution_delays=True,
    delay_config=custom_config
)

# Run backtest (delays applied automatically)
results = engine.run_backtest(data)
```

## Configuration

### In config.yaml

```yaml
backtest:
  execution_delays:
    enabled: true
    
    # Fixed delay
    fixed_delay_ms: 200.0
    
    # Variable delay
    use_variable_delay: true
    delay_mean_ms: 200.0
    delay_std_ms: 50.0
    delay_distribution: "normal"  # normal | poisson | uniform
    
    # Slippage
    integrate_slippage: true
    slippage_bps: 10.0  # 10 basis points = 0.1%
    slippage_std_bps: 5.0
    
    # Partial fills
    model_partial_fills: true
    min_fill_rate: 0.8  # 80% minimum
    liquidity_threshold: 1000000.0  # $1M
    
    # Regime-based adjustments
    regime_based_delays: true
    high_volatility_multiplier: 1.5
    low_liquidity_multiplier: 2.0
    
    # Component delays
    data_feed_delay_ms: 100.0
    broker_processing_ms: 50.0
    network_jitter_ms: 20.0
```

### Recommended Settings by Strategy Type

#### High-Frequency Trading (< 1 min)
```yaml
delay_mean_ms: 150.0
delay_std_ms: 30.0
slippage_bps: 15.0
model_partial_fills: true
regime_based_delays: true
```

#### Intraday Trading (5-60 min)
```yaml
delay_mean_ms: 200.0
delay_std_ms: 50.0
slippage_bps: 10.0
model_partial_fills: true
regime_based_delays: true
```

#### Daily/Swing Trading
```yaml
delay_mean_ms: 250.0
delay_std_ms: 75.0
slippage_bps: 8.0
model_partial_fills: false
regime_based_delays: false
```

#### Position Trading (weekly+)
```yaml
delay_mean_ms: 300.0
delay_std_ms: 100.0
slippage_bps: 5.0
model_partial_fills: false
regime_based_delays: false
```

## API Reference

### DelayConfig

Configuration dataclass for execution delays.

**Parameters**:
- `fixed_delay_ms` (float): Fixed delay in milliseconds (default: 200.0)
- `use_variable_delay` (bool): Use stochastic delays (default: True)
- `delay_mean_ms` (float): Mean delay for variable delays (default: 200.0)
- `delay_std_ms` (float): Standard deviation of delays (default: 50.0)
- `delay_distribution` (str): Distribution type - "normal", "poisson", "uniform" (default: "normal")
- `integrate_slippage` (bool): Include slippage modeling (default: True)
- `slippage_bps` (float): Base slippage in basis points (default: 10.0)
- `slippage_std_bps` (float): Slippage variability (default: 5.0)
- `model_partial_fills` (bool): Model incomplete order fills (default: True)
- `min_fill_rate` (float): Minimum fill percentage (default: 0.8)
- `liquidity_threshold` (float): Minimum volume for full fills (default: 1000000.0)
- `regime_based_delays` (bool): Adjust delays by market regime (default: True)
- `high_volatility_multiplier` (float): Delay multiplier in high volatility (default: 1.5)
- `low_liquidity_multiplier` (float): Delay multiplier in low liquidity (default: 2.0)
- `data_feed_delay_ms` (float): Data feed latency (default: 100.0)
- `broker_processing_ms` (float): Broker processing time (default: 50.0)
- `network_jitter_ms` (float): Network jitter standard deviation (default: 20.0)

### ExecutionDelaySimulator

Main class for simulating execution delays.

#### Methods

##### `simulate_order_execution()`

Simulate realistic order execution with delays.

**Parameters**:
- `signal_time` (pd.Timestamp): Time when signal was generated
- `signal_price` (float): Price when signal was generated
- `order_type` (str): "buy" or "sell"
- `order_size` (float): Size of order (shares/contracts)
- `market_data` (pd.DataFrame): Historical price data with [timestamp, price, volume]
- `market_regime` (dict, optional): Market regime info with keys [volatility, liquidity, regime_type]

**Returns**: Dict with execution details:
```python
{
    'signal_time': pd.Timestamp,
    'execution_time': pd.Timestamp,
    'signal_price': float,
    'execution_price': float,
    'delay_ms': float,
    'fill_rate': float,
    'slippage_bps': float,
    'order_type': str,
    'order_size': float,
    'filled_size': float,
    'regime': dict
}
```

##### `get_execution_statistics()`

Get statistics on execution history.

**Returns**: Dict with statistics:
```python
{
    'total_executions': int,
    'avg_delay_ms': float,
    'std_delay_ms': float,
    'max_delay_ms': float,
    'avg_slippage_bps': float,
    'std_slippage_bps': float,
    'avg_fill_rate': float,
    'partial_fills': int
}
```

##### `reset()`

Reset execution history.

### DelayAwareBacktester

Backtest engine wrapper that integrates execution delays.

#### Methods

##### `apply_delays_to_signals()`

Apply execution delays to trading signals.

**Parameters**:
- `signals` (pd.DataFrame): Signals with [timestamp, signal, price, size]
- `market_data` (pd.DataFrame): Market data with [timestamp, price, volume]
- `market_regimes` (pd.DataFrame, optional): Regime information

**Returns**: pd.DataFrame with delayed executions

##### `calculate_pnl_with_delays()`

Calculate P&L accounting for execution delays.

**Parameters**:
- `delayed_executions` (pd.DataFrame): Output from apply_delays_to_signals()

**Returns**: Dict with P&L and trade details

##### `run_monte_carlo_delay_test()`

Run Monte Carlo simulation with varying delays.

**Parameters**:
- `signals` (pd.DataFrame): Trading signals
- `market_data` (pd.DataFrame): Market data
- `num_iterations` (int): Number of simulations (default: 1000)
- `market_regimes` (pd.DataFrame, optional): Regime information

**Returns**: Dict with Monte Carlo results including percentiles

## Best Practices

### 1. Always Enable for Realistic Backtests

```python
# ❌ Bad: No delay modeling
engine = BacktestEngine(enable_execution_delays=False)

# ✅ Good: Realistic delays
engine = BacktestEngine(enable_execution_delays=True)
```

### 2. Calibrate to Your Broker

Measure actual execution delays from your broker:

```python
# Collect real execution data
real_delays = []
for trade in historical_trades:
    delay = trade.execution_time - trade.signal_time
    real_delays.append(delay.total_seconds() * 1000)

# Use measured delays
config = DelayConfig(
    delay_mean_ms=np.mean(real_delays),
    delay_std_ms=np.std(real_delays)
)
```

### 3. Test Multiple Delay Scenarios

```python
# Conservative (high delays)
conservative_config = DelayConfig(delay_mean_ms=300.0, slippage_bps=15.0)

# Realistic (measured delays)
realistic_config = DelayConfig(delay_mean_ms=200.0, slippage_bps=10.0)

# Optimistic (low delays)
optimistic_config = DelayConfig(delay_mean_ms=150.0, slippage_bps=8.0)

# Test all scenarios
for config in [conservative_config, realistic_config, optimistic_config]:
    results = run_backtest_with_config(config)
```

### 4. Use Monte Carlo for Robustness

```python
# Test strategy robustness to delay variability
mc_result = backtester.run_monte_carlo_delay_test(
    signals, market_data, num_iterations=1000
)

# Strategy should be profitable in 95th percentile
if mc_result['pnl_percentiles']['5th'] > 0:
    print("Strategy is robust to delay variability")
else:
    print("Strategy fails with high delays - needs improvement")
```

### 5. Monitor Execution Statistics

```python
stats = simulator.get_execution_statistics()

# Check for issues
if stats['avg_delay_ms'] > 500:
    print("WARNING: Very high average delays")

if stats['avg_fill_rate'] < 0.9:
    print("WARNING: Many partial fills - check liquidity")

if stats['avg_slippage_bps'] > 20:
    print("WARNING: High slippage - reduce order sizes")
```

## Performance Impact Examples

### Example 1: Intraday Strategy

**Without delays**:
- Sharpe: 2.1
- Annual return: 45%
- Max drawdown: -12%

**With realistic delays (200ms mean)**:
- Sharpe: 1.6 (-24%)
- Annual return: 32% (-29%)
- Max drawdown: -15% (+25%)

### Example 2: High-Frequency Strategy

**Without delays**:
- Sharpe: 3.2
- Annual return: 85%
- Max drawdown: -8%

**With realistic delays (150ms mean)**:
- Sharpe: 1.8 (-44%)
- Annual return: 42% (-51%)
- Max drawdown: -14% (+75%)

### Example 3: Daily Strategy

**Without delays**:
- Sharpe: 1.8
- Annual return: 28%
- Max drawdown: -18%

**With realistic delays (250ms mean)**:
- Sharpe: 1.6 (-11%)
- Annual return: 24% (-14%)
- Max drawdown: -20% (+11%)

## Troubleshooting

### Issue: Delays Too High

**Symptoms**: Average delay > 500ms

**Solutions**:
1. Check `data_feed_delay_ms` and `broker_processing_ms` settings
2. Reduce `delay_mean_ms` if using simulated delays
3. Verify market data timestamps are correct

### Issue: Too Many Partial Fills

**Symptoms**: `avg_fill_rate` < 0.9

**Solutions**:
1. Reduce order sizes relative to market volume
2. Adjust `liquidity_threshold` setting
3. Filter out low-volume symbols
4. Increase `min_fill_rate` for conservative testing

### Issue: Excessive Slippage

**Symptoms**: `avg_slippage_bps` > 20

**Solutions**:
1. Reduce `slippage_bps` if unrealistic
2. Check if `regime_based_delays` is causing excessive multipliers
3. Verify market data quality (no gaps or errors)
4. Consider using limit orders instead of market orders

### Issue: Monte Carlo Results Too Variable

**Symptoms**: Large difference between 5th and 95th percentiles

**Solutions**:
1. Increase `num_iterations` to 5000+
2. Reduce `delay_std_ms` for more consistent delays
3. Strategy may be too sensitive to timing - needs improvement
4. Consider adding more robust entry/exit rules

## References

- [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/1119800064) - Chapter on execution costs
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089) - Section on backtesting pitfalls
- [Trading and Exchanges: Market Microstructure for Practitioners](https://www.amazon.com/Trading-Exchanges-Market-Microstructure-Practitioners/dp/0195144708) - Market microstructure and execution
- [High-Frequency Trading: A Practical Guide](https://www.amazon.com/High-Frequency-Trading-Practical-Algorithmic-Strategies/dp/1118343506) - Latency and execution modeling

## See Also

- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Critical improvements for production
- [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) - Advanced trading features
- [HYBRID_AND_OFFLINE.md](HYBRID_AND_OFFLINE.md) - Hybrid modes and offline research
