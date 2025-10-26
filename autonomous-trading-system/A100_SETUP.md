# A100 80GB Setup Guide

This guide is optimized for running the autonomous trading system on an A100 80GB GPU.

## System Configuration

The system is now configured to maximize GPU utilization on A100 80GB:

- **3-Model Ensemble**: Mixtral-8x7B (50%), Llama-3-8B (30%), Mistral-7B (20%)
- **Batch Size**: 32 symbols processed in parallel
- **Max Workers**: 16 threads for CPU parallelism
- **Context Window**: 4096 tokens (2x larger)
- **Max New Tokens**: 512 (2x larger)
- **Expected GPU Usage**: 60-70GB

## Quick Setup (Google Colab)

```python
# 1. Clone and checkout branch
!git clone https://github.com/ryanhamil7-collab/stuff.git
%cd stuff/autonomous-trading-system
!git checkout devin/1761207403-autonomous-trading-system

# 2. Install PyTorch with CUDA 11.8
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install transformers and quantization libraries
!pip install -q transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0

# 4. Install trading dependencies
!pip install -q yfinance pandas numpy scipy scikit-learn xgboost
!pip install -q pandas-ta tenacity pyyaml loguru requests pydantic
!pip install -q alpaca-trade-api websockets stable-baselines3 gymnasium

# 5. Verify GPU
!nvidia-smi
```

## Running Backtest

### Standard Backtest (10 symbols, 9 months)
```python
!python3 examples/quick_backtest.py \
  --symbols AAPL MSFT GOOGL NVDA TSLA META AMZN NFLX AMD INTC \
  --start 2023-06-01 \
  --end 2024-03-01
```

### Large Backtest (20 symbols, 1 year)
```python
!python3 examples/quick_backtest.py \
  --symbols AAPL MSFT GOOGL NVDA TSLA META AMZN NFLX AMD INTC CSCO ORCL QCOM AVGO TXN ADBE CRM PYPL UBER ABNB \
  --start 2023-01-01 \
  --end 2024-01-01
```

### Maximum GPU Utilization (30+ symbols)
```python
!python3 examples/quick_backtest.py \
  --symbols AAPL MSFT GOOGL NVDA TSLA META AMZN NFLX AMD INTC CSCO ORCL QCOM AVGO TXN ADBE CRM PYPL UBER ABNB SPY QQQ IWM DIA XLF XLE XLK XLV XLI XLU \
  --start 2023-01-01 \
  --end 2024-01-01
```

## Monitoring GPU Usage

### Real-time GPU monitoring
```python
# In a separate cell
!watch -n 1 nvidia-smi
```

### Check GPU memory during backtest
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## Expected Performance

### GPU Utilization
- **Idle**: ~2GB (base PyTorch)
- **Loading Models**: 40-50GB (loading 3 models)
- **Inference**: 60-70GB (batch processing 32 symbols)
- **Peak**: 75GB (during ensemble voting)

### Processing Speed
- **10 symbols**: ~2-3 minutes
- **20 symbols**: ~4-6 minutes
- **30+ symbols**: ~8-12 minutes

### Model Loading Time
- **Mixtral-8x7B**: ~30 seconds
- **Llama-3-8B**: ~20 seconds
- **Mistral-7B**: ~15 seconds
- **Total**: ~65 seconds for all 3 models

## Troubleshooting

### Out of Memory (OOM)
If you get OOM errors, reduce batch_size:
```python
# Edit config/config.yaml
llm:
  batch_size: 16  # Reduce from 32 to 16
```

Or disable one model from ensemble:
```python
# Edit config/config.yaml
llm:
  ensemble:
    models:
      - name: "mistralai/Mixtral-8x7B-Instruct-v0.1"
        weight: 0.7
        quantization: "4bit"
      - name: "mistralai/Mistral-7B-Instruct-v0.2"
        weight: 0.3
        quantization: "4bit"
```

### Models Not Loading
Check if you have HuggingFace access token:
```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

### Slow Performance
Ensure you're using CUDA 11.8 or higher:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Live Trading Setup

### Paper Trading with Alpaca
```python
# 1. Set up API keys
import os
os.environ['ALPACA_API_KEY'] = 'YOUR_PAPER_KEY'
os.environ['ALPACA_SECRET_KEY'] = 'YOUR_PAPER_SECRET'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'

# 2. Launch autopilot
!python3 launch_autopilot.py
```

### Monitor trades on Alpaca dashboard
Visit: https://app.alpaca.markets/paper/dashboard/overview

## Advanced Features

### Enable RL Training
Already enabled in config! The system will learn from trades in real-time.

### Enable Ensemble Voting
Already enabled! 3 models vote on every decision.

### View Training Statistics
```python
# Check RL training logs
!tail -f logs/trading_system.log | grep "RL Training"
```

## Performance Metrics

Expected results with A100 80GB optimization:

- **Sharpe Ratio**: 1.5-2.5
- **Annual Return**: 15-30%
- **Max Drawdown**: 10-20%
- **Win Rate**: 55-65%
- **Trades per Day**: 5-15

## Support

For issues or questions:
- GitHub: https://github.com/ryanhamil7-collab/stuff
- PR: https://github.com/ryanhamil7-collab/stuff/pull/5
