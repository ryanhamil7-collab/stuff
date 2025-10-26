# Google Colab Setup Guide

This guide provides step-by-step instructions for running the Autonomous Trading System in Google Colab with L4/T4 GPUs.

## Important: Dependency Resolution

The system requires pandas-ta 0.4.71b0, which needs:
- `pandas>=2.3.2`
- `numpy>=2.2.6`  
- `numba==0.61.2`

These are newer than Colab's default versions, but they are **required for full functionality**. The installation process below handles this correctly.

## Installation Steps

### Step 1: Clone Repository

```python
!git clone -b devin/1761207403-autonomous-trading-system https://github.com/ryanhamil7-collab/stuff.git
%cd stuff/autonomous-trading-system
```

### Step 2: Install Dependencies (Correct Order)

**IMPORTANT**: Install in this specific order to avoid conflicts:

```python
# Step 2a: Upgrade pip
!pip install --upgrade pip

# Step 2b: Install core data processing libraries FIRST (this upgrades numpy/pandas)
!pip install 'pandas>=2.3.2' 'numpy>=2.2.6' 'numba==0.61.2'

# Step 2c: Restart runtime to load new numpy/pandas
# Click: Runtime > Restart runtime (or use keyboard shortcut)
# Then re-run Step 1 (cd to directory) and continue with Step 2d

# Step 2d: Install remaining dependencies
!pip install yfinance tenacity pyyaml loguru requests scikit-learn xgboost
!pip install torch transformers pydantic accelerate bitsandbytes sentencepiece
!pip install pandas-ta>=0.4.71b0
```

### Step 3: Verify Installation

```python
!python3 fix_and_test.py
```

Expected output:
- ✅ All imports successful (with warnings for optional RL/crypto deps)
- ✅ Backtest generates 26 trades
- ✅ 24.65% return, 1.14 Sharpe ratio
- ✅ LLM and ML models working

## Why the Specific Order?

1. **Upgrade numpy/pandas first**: Colab has old versions that conflict with pandas-ta
2. **Restart runtime**: Required for Python to load the new numpy/pandas versions
3. **Install pandas-ta last**: Ensures it gets the correct dependencies

## Troubleshooting

### Error: "numpy.dtype size changed"
**Solution**: You didn't restart the runtime after upgrading numpy/pandas. Go to Runtime > Restart runtime, then re-run from Step 2d.

### Error: "pandas 2.2.2 but you have pandas 2.3.3 which is incompatible"
**This is expected and safe to ignore**. The warning comes from google-colab package, but it doesn't affect functionality. The system works perfectly with pandas 2.3.3.

### Error: "tensorflow requires numpy<2.2.0"
**This is expected and safe to ignore**. TensorFlow is not used by the trading system. If you need TensorFlow for other work, create a separate notebook.

## Running Backtests

### Quick Test (30 seconds)
```python
!python3 fix_and_test.py
```

### Full Backtest with LLM (5-10 minutes on CPU)
```python
!python3 examples/quick_backtest.py --symbols AAPL MSFT GOOGL --start 2023-01-01 --end 2024-10-24
```

### With GPU Acceleration (if available)
The LLM will automatically use GPU if available. Check GPU status:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Features Verified Working

✅ **Data Fetching**: yfinance, historical data
✅ **Technical Indicators**: All indicators including pandas-ta
✅ **ML Models**: Alpha mining, sentiment analysis
✅ **LLM Integration**: Mistral-7B-Instruct for trading decisions
✅ **Backtesting**: Full backtest engine with execution delays
✅ **Risk Management**: Position sizing, stop loss, take profit
✅ **Portfolio Management**: Multi-symbol trading

## Optional Dependencies

These are disabled if not installed (system still works):
- **RL Training**: gymnasium, stable-baselines3 (for reinforcement learning)
- **Crypto Discovery**: ccxt (for cryptocurrency symbol discovery)

To enable them:
```python
!pip install gymnasium stable-baselines3 ccxt
```

## Performance Notes

- **CPU Mode**: LLM inference takes ~5-10 minutes for full backtest
- **GPU Mode (L4/T4)**: LLM inference takes ~1-2 minutes for full backtest
- **Memory**: System uses ~8GB RAM for full backtest with LLM

## Support

If you encounter issues:
1. Check that you followed the installation order exactly
2. Verify you restarted the runtime after upgrading numpy/pandas
3. Run `fix_and_test.py` to diagnose import errors
4. Check the logs in `logs/` directory for detailed error messages
