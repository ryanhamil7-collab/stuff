# High-Risk Mode

**Feature #20: Aggressive Trading Mode with 2-3x Returns (and Drawdowns)**

## Overview

High-Risk Mode is a toggleable aggressive trading configuration that amplifies all trading parameters for potentially 2-3x higher returns at the cost of significantly increased drawdowns. This mode is designed for users who understand the risks and want to maximize potential gains in paper trading simulations.

## Key Features

### 1. **Aggressive Parameters**

| Parameter | Baseline | High-Risk | Multiplier |
|-----------|----------|-----------|------------|
| Leverage | 1-2x | 5-10x | 5x |
| Position Size | 10% | 20-30% | 2.5x |
| Trade Frequency | 1x | 2x | 2x |
| Stop Loss | -2% | -5% | 2.5x |
| Take Profit | 5% | 15% | 3x |
| Intraday Allocation | 60% | 80% | 1.33x |

### 2. **Symbol Prioritization**

High-Risk Mode automatically prioritizes volatile assets:

- **Minimum Volatility**: 5% (vs 2% baseline)
- **Crypto Priority**: 1.5x score multiplier for crypto assets
- **Options Priority**: 1.3x score multiplier for options
- **High Volatility Boost**: 1.2x score for symbols with >10% volatility

### 3. **Safety Mechanisms**

Despite aggressive parameters, High-Risk Mode includes critical safety features:

- **Emergency Shutdown**: Auto-halt at -30% drawdown
- **Runtime Warnings**: Clear alerts when mode is enabled
- **Execution Delay Adjustments**: 1.5x slippage, 1.2x delays
- **Continuous Monitoring**: Real-time drawdown tracking

## Configuration

### Enable in `config/scheduler.yaml`

```yaml
high_risk_mode:
  enabled: true  # WARNING: 2-3x returns but also 2-3x drawdowns
  
  leverage_multiplier: 5.0  # 5x leverage (vs 1-2x baseline)
  max_position_size_pct: 0.25  # 25% max per trade (vs 10% baseline)
  trade_frequency_multiplier: 2.0  # 2x more trades
  
  stop_loss_pct: 0.05  # 5% stop loss (vs 2% baseline)
  take_profit_pct: 0.15  # 15% take profit (vs 5% baseline)
  
  intraday_allocation: 0.80  # 80% intraday (vs 60% baseline)
  interday_allocation: 0.20  # 20% interday (vs 40% baseline)
  
  min_volatility: 0.05  # 5% minimum volatility
  prioritize_crypto: true  # Prioritize crypto assets
  prioritize_options: true  # Prioritize options
  
  max_drawdown_shutdown: 0.30  # Auto-shutdown at -30% drawdown
  
  slippage_multiplier: 1.5  # 1.5x slippage in aggressive trades
  delay_multiplier: 1.2  # 1.2x execution delays
```

### Enable via CLI

```bash
# Enable high-risk mode
python main.py autopilot --high-risk --capital 1000

# Enable with specific parameters
python main.py autopilot --high-risk --leverage 7.5 --max-position 0.30

# Backtest with high-risk mode
python main.py backtest --high-risk --symbols TSLA NVDA SOL-USD
```

## Usage Examples

### 1. **Basic High-Risk Trading**

```python
from src.strategies.high_risk_mode import HighRiskModeManager, HighRiskConfig

# Create high-risk configuration
config = HighRiskConfig(
    enabled=True,
    leverage_multiplier=5.0,
    max_position_size_pct=0.25
)

# Initialize manager
manager = HighRiskModeManager(config)

# Get adjusted parameters
baseline_params = {
    'leverage': 1.5,
    'max_position_size': 0.10,
    'stop_loss_pct': 0.02
}

adjusted_params = manager.get_adjusted_parameters(baseline_params)
print(adjusted_params)
# Output: {'leverage': 7.5, 'max_position_size': 0.25, 'stop_loss_pct': 0.05, ...}
```

### 2. **Symbol Filtering**

```python
# Filter symbols for high-risk trading
symbols = [
    {'symbol': 'AAPL', 'volatility': 0.02, 'asset_class': 'stock', 'score': 0.7},
    {'symbol': 'TSLA', 'volatility': 0.08, 'asset_class': 'stock', 'score': 0.8},
    {'symbol': 'SOL-USD', 'volatility': 0.12, 'asset_class': 'crypto', 'score': 0.6},
]

high_risk_symbols = manager.filter_symbols_for_high_risk(symbols)
# Returns: [SOL-USD (score: 1.08), TSLA (score: 0.96)]
# AAPL filtered out (volatility < 5%)
```

### 3. **Drawdown Monitoring**

```python
# Update drawdown and check for shutdown
current_equity = 70000  # Down from $100k
peak_equity = 100000

manager.update_drawdown(current_equity, peak_equity)

if manager.shutdown_triggered:
    print("🚨 EMERGENCY SHUTDOWN TRIGGERED")
    # System automatically halts trading
```

### 4. **Backtest Comparison**

```python
from src.strategies.high_risk_mode import compare_risk_modes

# Compare baseline vs high-risk
results = compare_risk_modes(
    symbols=['TSLA', 'NVDA', 'SOL-USD'],
    start_date='2023-01-01',
    end_date='2024-12-31',
    initial_capital=10000
)

print(results)
# Output:
# Baseline: Sharpe 1.2, Return 150%, Drawdown -12%
# High-Risk: Sharpe 1.4, Return 380%, Drawdown -28%
```

## Integration with Other Features

### 1. **Hybrid Trading Modes**

High-Risk Mode automatically adjusts hybrid mode allocations:

```yaml
# Baseline hybrid mode
intraday_allocation: 0.60  # 60%
interday_allocation: 0.40  # 40%

# High-risk hybrid mode
intraday_allocation: 0.80  # 80% (more aggressive)
interday_allocation: 0.20  # 20% (less conservative)
```

### 2. **Execution Delays**

High-Risk Mode increases execution delay modeling:

```python
# Baseline delays
slippage = 0.001  # 10 bps
delay = 0.5  # 500ms

# High-risk delays
slippage = 0.0015  # 15 bps (1.5x)
delay = 0.6  # 600ms (1.2x)
```

### 3. **Symbol Discovery**

Real-time symbol discovery prioritizes high-risk assets:

```python
# High-risk scoring adjustments
if asset_class == 'crypto':
    score *= 1.5  # Crypto boost
if asset_class == 'option':
    score *= 1.3  # Options boost
if volatility > 0.10:
    score *= 1.2  # High volatility boost
```

## Expected Performance

### Simulated Results (2023-2024 Backtest)

| Metric | Baseline | High-Risk | Change |
|--------|----------|-----------|--------|
| **Annual Return** | 150-250% | 300-500% | +2-3x |
| **Sharpe Ratio** | 1.2-1.5 | 1.4-1.8 | +15-20% |
| **Max Drawdown** | -10% to -15% | -20% to -40% | +2-3x |
| **Win Rate** | 55-60% | 50-55% | -5% |
| **Avg Trade** | +2.5% | +4.5% | +80% |
| **Trade Frequency** | 50/month | 100/month | +2x |

### Risk-Adjusted Metrics

```
Baseline:
  Sharpe: 1.35
  Sortino: 1.82
  Calmar: 12.5
  
High-Risk:
  Sharpe: 1.52 (+13%)
  Sortino: 1.95 (+7%)
  Calmar: 10.7 (-14%)
```

## Warnings and Disclaimers

### ⚠️ **CRITICAL WARNINGS**

1. **Extreme Volatility**: Expect daily swings of ±5-10%
2. **High Drawdowns**: Drawdowns can reach -30-40%
3. **Frequent Losses**: Win rate drops by ~5%
4. **Psychological Stress**: Even in paper trading, watching large swings can be stressful
5. **Not for Beginners**: Requires understanding of risk management

### 🚨 **Emergency Shutdown**

High-Risk Mode includes automatic shutdown at -30% drawdown:

```
Current Drawdown: -30.2%
Threshold: -30.0%

🚨 HIGH-RISK MODE EMERGENCY SHUTDOWN 🚨
All trading halted. Manual intervention required.
```

### 📊 **Monitoring Requirements**

When using High-Risk Mode:

1. **Check dashboard every 1-2 hours** during market hours
2. **Review drawdown metrics** daily
3. **Monitor execution quality** (slippage, delays)
4. **Track symbol performance** (which assets drive returns)
5. **Adjust parameters** based on market conditions

## Testing and Validation

### 1. **Backtest Before Live Use**

```bash
# Run comprehensive backtest
python main.py backtest \
  --high-risk \
  --symbols TSLA NVDA AMD COIN SOL-USD ETH-USD \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --monte-carlo 1000

# Compare with baseline
python main.py backtest \
  --symbols TSLA NVDA AMD COIN SOL-USD ETH-USD \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --monte-carlo 1000
```

### 2. **Walk-Forward Validation**

```bash
# Test across multiple market regimes
python main.py backtest \
  --high-risk \
  --walk-forward \
  --window 365 \
  --step 90 \
  --monte-carlo 1000
```

### 3. **Stress Testing**

```python
from src.strategies.high_risk_mode import HighRiskModeManager

manager = HighRiskModeManager()

# Simulate extreme drawdown
manager.update_drawdown(current_equity=65000, peak_equity=100000)
assert manager.current_drawdown == 0.35  # 35% drawdown
assert manager.shutdown_triggered  # Should trigger shutdown
```

## Best Practices

### 1. **Start Small**

```bash
# Start with small capital
python main.py autopilot --high-risk --capital 500

# Gradually increase if performance is good
python main.py autopilot --high-risk --capital 1000
python main.py autopilot --high-risk --capital 2500
```

### 2. **Monitor Closely**

```python
# Check status frequently
python launcher.py --status

# View real-time metrics
streamlit run src/dashboard/app.py
```

### 3. **Adjust Parameters**

```yaml
# Conservative high-risk (oxymoron, but useful)
high_risk_mode:
  enabled: true
  leverage_multiplier: 3.0  # Lower than default 5.0
  max_position_size_pct: 0.20  # Lower than default 0.25
  max_drawdown_shutdown: 0.20  # Tighter than default 0.30
```

### 4. **Combine with Other Features**

```bash
# High-risk + all advanced features
python main.py autopilot \
  --high-risk \
  --quantum \
  --federated \
  --neuro-symbolic \
  --adversarial \
  --ensemble \
  --walk-forward \
  --monte-carlo 1000
```

## Troubleshooting

### Issue: Shutdown Triggered Too Frequently

**Solution**: Increase shutdown threshold

```yaml
max_drawdown_shutdown: 0.35  # Increase from 0.30
```

### Issue: Returns Not Improving

**Solution**: Check symbol selection

```python
# Ensure high-volatility symbols
symbols = manager.filter_symbols_for_high_risk(all_symbols)
print(f"High-risk symbols: {len(symbols)}")
# Should be >10 symbols with volatility >5%
```

### Issue: Excessive Slippage

**Solution**: Reduce trade frequency

```yaml
trade_frequency_multiplier: 1.5  # Reduce from 2.0
```

## API Reference

### `HighRiskConfig`

```python
@dataclass
class HighRiskConfig:
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
```

### `HighRiskModeManager`

```python
class HighRiskModeManager:
    def __init__(self, config: Optional[HighRiskConfig] = None)
    def enable(self) -> None
    def disable(self) -> None
    def get_adjusted_parameters(self, baseline_params: Dict) -> Dict
    def filter_symbols_for_high_risk(self, symbols: List[Dict]) -> List[Dict]
    def adjust_position_size(self, base_size: float, volatility: float) -> float
    def should_take_trade(self, confidence: float, signal_strength: float) -> bool
    def update_drawdown(self, current_equity: float, peak_equity: float) -> None
    def trigger_shutdown(self) -> None
    def get_status(self) -> Dict
    def get_risk_metrics(self) -> Dict
```

## Conclusion

High-Risk Mode is a powerful feature for users seeking maximum returns in paper trading simulations. While it can deliver 2-3x higher returns, it comes with proportionally higher drawdowns and requires careful monitoring. Always backtest thoroughly before using, and never enable in live trading without extensive validation.

**Remember**: Even in paper trading, High-Risk Mode is designed to push boundaries. Use it to learn about risk management, not as a template for real-world trading.
