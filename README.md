# Autonomous Trading System v2.0

A comprehensive algorithmic trading platform with 19 advanced features, optimized for Kaggle/Colab deployment with zero-setup-errors.

## Features (All 19 Implemented)

1. **Multi-Strategy Support** - Run multiple trading strategies simultaneously
2. **Hive Mind Consensus** - Aggregate signals from multiple strategies
3. **Dynamic Position Sizing** - Adaptive position sizing based on risk
4. **Adaptive Stop Loss** - Dynamic stop-loss adjustments
5. **Backtesting Engine** - Historical performance testing
6. **Paper Trading** - Risk-free trading simulation with Alpaca
7. **Live Trading** - Real-time trading capabilities (disabled by default)
8. **Risk Management** - Comprehensive risk controls
9. **Portfolio Optimization** - Automated portfolio balancing
10. **Technical Indicators** - 19 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, OBV, VWAP, CCI, Williams %R, Momentum, ROC, Ichimoku, Parabolic SAR, Supertrend, Keltner Channels, Donchian Channels)
11. **Pattern Recognition** - Chart pattern detection
12. **Machine Learning** - ML-based prediction models
13. **Automated Rebalancing** - Portfolio rebalancing
14. **Performance Analytics** - Detailed performance metrics
15. **Real-time Monitoring** - Live system monitoring
16. **Symbol Discovery** - Automated symbol screening
17. **Sentiment Analysis** - Market sentiment integration (optional)
18. **News Integration** - News-based trading signals (optional)
19. **Options Trading** - Options strategy support (optional)

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd autonomous-trading-system

# Run automated setup
./setup.sh

# Or use Python installer
python3 install.py

# Configure API keys
nano .env  # Add your Alpaca API keys

# Launch the system
python3 launcher.py
```

### Kaggle Installation

See [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md) for detailed Kaggle-specific instructions.

**Quick Kaggle Setup:**

```python
# In a Kaggle notebook (Internet ON, GPU T4 x2 recommended)

# Clone repository
!git clone <repository-url>
%cd autonomous-trading-system

# Install dependencies
!python3 install.py

# Configure (optional - works with defaults)
import os
os.environ['KAGGLE_MODE'] = 'true'
os.environ['KEEP_ALIVE'] = 'true'

# Launch
!python3 launcher.py
```

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **OS**: Linux, macOS, Windows (WSL recommended)
- **Memory**: 2GB+ RAM
- **Internet**: Required for data fetching

## Dependencies

All dependencies are automatically installed via `setup.sh` or `install.py`:

- numpy (1.24.0+, <2.0.0)
- pandas (2.0.0+)
- pandas-ta (0.3.14b0) - Fixed version for Python 3.10 compatibility
- yfinance (0.2.28+)
- alpaca-py (0.15.0+)
- apscheduler (3.10.0+) - Now included
- flask (3.0.0+)
- pyyaml (6.0+)
- python-dotenv (1.0.0+)
- loguru (0.7.0+)

## Configuration

### Environment Variables (.env)

```env
# Alpaca API (Paper Trading)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here

# Trading Settings
TRADING_MODE=paper
INITIAL_CAPITAL=1000

# Kaggle/Colab Settings
KAGGLE_MODE=true
KEEP_ALIVE=true
```

### Main Configuration (config/config.yaml)

```yaml
trading:
  mode: paper
  symbols:
    - AAPL
    - MSFT
    - GOOGL

features:
  backtesting: true
  paper_trading: true
  risk_management: true
```

### Scheduler Configuration (config/scheduler.yaml)

```yaml
capital:
  initial: 1000

risk_management:
  max_daily_loss: 0.02
  stop_loss_percentage: 0.02

symbol_discovery:
  enabled: true
  max_symbols: 50
```

## Project Structure

```
autonomous-trading-system/
├── launcher.py              # Robust launcher with error handling
├── main.py                  # Main trading system
├── install.py               # Python-based installer
├── setup.sh                 # Bash setup script
├── requirements.txt         # Fixed dependencies
├── .env.example             # Environment template
├── config/
│   ├── config.yaml          # Main configuration
│   └── scheduler.yaml       # Scheduler settings
├── src/
│   ├── indicators/          # Technical indicators
│   ├── strategies/          # Trading strategies
│   ├── risk_management/     # Risk controls
│   ├── data/                # Data management
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── docs/
│   ├── KAGGLE_SETUP.md      # Kaggle deployment guide
│   └── API.md               # API documentation
├── data/                    # Data storage
├── logs/                    # Log files
└── backtest_results/        # Backtest outputs
```

## Usage

### Running Backtests

```python
python3 launcher.py
```

The system will automatically run backtests on configured symbols with all 19 technical indicators.

### Paper Trading

1. Get free Alpaca API keys: https://alpaca.markets/
2. Update `.env` with your keys
3. Run: `python3 launcher.py`

### Viewing Results

- **Logs**: `logs/trading.log`
- **Backtest Results**: `backtest_results/`
- **Dashboard**: http://localhost:5000 (if enabled)

## Troubleshooting

### Common Issues

#### 1. `[Errno 2] No such file or directory: 'stuff/autonomous-trading-system'`

**Solution**: Ensure you're in the correct directory:
```bash
cd autonomous-trading-system  # Enter the subfolder
python3 launcher.py
```

#### 2. `ERROR: Could not find a version that satisfies the requirement pandas-ta>=0.3.14b`

**Solution**: Fixed in requirements.txt with `pandas-ta==0.3.14b0`. Run:
```bash
pip install pandas-ta==0.3.14b0
```

#### 3. `ModuleNotFoundError: No module named 'apscheduler'`

**Solution**: Now included in requirements.txt. Run:
```bash
pip install apscheduler>=3.10.0
```

#### 4. `sed: can't read config/scheduler.yaml`

**Solution**: The launcher automatically creates missing config files. If issues persist:
```bash
python3 launcher.py  # Auto-creates missing configs
```

#### 5. Python Version Issues

**Solution**: Ensure Python 3.10+:
```bash
python3 --version  # Should be 3.10, 3.11, or 3.12
```

### Kaggle-Specific Issues

#### Session Timeout

**Solution**: Enable keep-alive in `.env`:
```env
KEEP_ALIVE=true
KEEP_ALIVE_INTERVAL=300
```

#### Internet Access

**Solution**: Enable Internet in Kaggle notebook settings:
- Settings → Internet → ON

#### GPU Selection

**Recommended**: T4 x2 or P100
- Settings → Accelerator → GPU T4 x2

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_indicators.py

# With coverage
pytest --cov=src tests/
```

## Development

### Adding New Indicators

1. Add indicator to `src/indicators/technical_indicators.py`
2. Update `config/config.yaml` with indicator config
3. Test with backtesting

### Adding New Strategies

1. Create strategy in `src/strategies/`
2. Register in `config/config.yaml`
3. Test with paper trading

## Performance

- **Backtest Speed**: ~1000 bars/second
- **Memory Usage**: ~500MB for 50 symbols
- **Indicator Calculation**: <100ms per symbol

## Security

- **API Keys**: Never commit `.env` to version control
- **Paper Trading**: Default mode (safe)
- **Live Trading**: Disabled by default (requires explicit enable)

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Open an issue on GitHub
- **Kaggle Guide**: See `docs/KAGGLE_SETUP.md`

## License

MIT License - See LICENSE file

## Changelog

### v2.0.0 (Current)

- ✅ Fixed all Kaggle/Colab deployment issues
- ✅ Fixed `pandas-ta` version compatibility (0.3.14b0)
- ✅ Added `apscheduler` to dependencies
- ✅ Fixed directory structure issues
- ✅ Added automatic config file creation
- ✅ Improved error handling in launcher
- ✅ Added Python 3.10-3.12 compatibility
- ✅ Added comprehensive Kaggle documentation
- ✅ Added automated setup scripts
- ✅ Implemented all 19 features
- ✅ Zero-setup-errors deployment

### v1.0.0

- Initial release

## Credits

Developed by the Autonomous Trading Team

## Disclaimer

This software is for educational purposes only. Trading involves risk. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real capital.
