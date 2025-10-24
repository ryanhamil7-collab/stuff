# Kaggle/Colab Setup Guide

Complete guide for deploying the Autonomous Trading System on Kaggle with zero errors.

## Prerequisites

### Kaggle Account Setup

1. Create a Kaggle account at https://www.kaggle.com
2. Verify your phone number (required for Internet access)
3. Navigate to "Code" → "New Notebook"

### Required Settings

**IMPORTANT**: Configure these settings before running:

1. **Internet Access**: Settings → Internet → **ON**
2. **Accelerator**: Settings → Accelerator → **GPU T4 x2** (recommended)
3. **Persistence**: Settings → Persistence → **Files only** (optional)

## Installation Methods

### Method 1: Quick Install (Recommended)

```python
# Cell 1: Clone and Install
!git clone https://github.com/your-username/autonomous-trading-system.git
%cd autonomous-trading-system

# Run Python installer
!python3 install.py
```

### Method 2: Manual Install

```python
# Cell 1: Clone Repository
!git clone https://github.com/your-username/autonomous-trading-system.git
%cd autonomous-trading-system

# Cell 2: Install Dependencies
!pip install -q numpy>=1.24.0,<2.0.0
!pip install -q pandas>=2.0.0
!pip install -q pandas-ta==0.3.14b0
!pip install -q yfinance>=0.2.28
!pip install -q alpaca-py>=0.15.0
!pip install -q python-dotenv>=1.0.0
!pip install -q pyyaml>=6.0
!pip install -q apscheduler>=3.10.0
!pip install -q flask>=3.0.0
!pip install -q loguru>=0.7.0

# Cell 3: Verify Installation
import pandas, numpy, yfinance, yaml, apscheduler
print("✓ All modules imported successfully")
```

### Method 3: Bash Setup Script

```bash
# Cell 1: Clone and Setup
!git clone https://github.com/your-username/autonomous-trading-system.git
%cd autonomous-trading-system

# Run bash installer
!bash setup.sh
```

## Configuration

### Basic Configuration (No API Keys)

The system works out-of-the-box for backtesting without API keys:

```python
# Cell: Configure Environment
import os

# Set Kaggle mode
os.environ['KAGGLE_MODE'] = 'true'
os.environ['KEEP_ALIVE'] = 'true'
os.environ['TRADING_MODE'] = 'paper'
os.environ['LOG_LEVEL'] = 'INFO'

print("✓ Environment configured")
```

### Advanced Configuration (With Alpaca API)

For paper trading, add your Alpaca API keys:

```python
# Cell: Configure with API Keys
import os

# Alpaca API (get free keys at https://alpaca.markets)
os.environ['ALPACA_API_KEY'] = 'your_api_key_here'
os.environ['ALPACA_SECRET_KEY'] = 'your_secret_key_here'

# Trading settings
os.environ['TRADING_MODE'] = 'paper'
os.environ['INITIAL_CAPITAL'] = '1000'

# Kaggle settings
os.environ['KAGGLE_MODE'] = 'true'
os.environ['KEEP_ALIVE'] = 'true'

print("✓ API keys configured")
```

### Using Kaggle Secrets (Recommended)

1. Go to Kaggle Account Settings → API
2. Add secrets:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
3. Enable secrets in notebook settings

```python
# Cell: Load Kaggle Secrets
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

import os
os.environ['ALPACA_API_KEY'] = user_secrets.get_secret("ALPACA_API_KEY")
os.environ['ALPACA_SECRET_KEY'] = user_secrets.get_secret("ALPACA_SECRET_KEY")

print("✓ Secrets loaded from Kaggle")
```

## Running the System

### Basic Run (Backtesting)

```python
# Cell: Launch System
!python3 launcher.py
```

### Run with Keep-Alive (Prevent Timeout)

```python
# Cell: Launch with Keep-Alive
import subprocess
import time
import os

# Set keep-alive
os.environ['KEEP_ALIVE'] = 'true'
os.environ['KEEP_ALIVE_INTERVAL'] = '300'  # 5 minutes

# Launch in background
proc = subprocess.Popen(['python3', 'launcher.py'])

# Keep-alive loop
try:
    while True:
        time.sleep(300)  # 5 minutes
        print("Keep-alive ping...")
except KeyboardInterrupt:
    proc.terminate()
    print("System stopped")
```

### Run Specific Features

```python
# Cell: Run Backtest Only
import sys
sys.path.insert(0, '/kaggle/working/autonomous-trading-system')

from main import TradingSystem

system = TradingSystem()
system.run_backtest()
```

## Complete Kaggle Notebook Example

```python
# ============================================
# CELL 1: Setup and Installation
# ============================================

# Clone repository
!git clone https://github.com/your-username/autonomous-trading-system.git
%cd autonomous-trading-system

# Install dependencies
!python3 install.py

# ============================================
# CELL 2: Configuration
# ============================================

import os

# Configure environment
os.environ['KAGGLE_MODE'] = 'true'
os.environ['KEEP_ALIVE'] = 'true'
os.environ['TRADING_MODE'] = 'paper'
os.environ['LOG_LEVEL'] = 'INFO'

print("✓ Configuration complete")

# ============================================
# CELL 3: Launch System
# ============================================

!python3 launcher.py

# ============================================
# CELL 4: View Results (Optional)
# ============================================

# View logs
!tail -n 50 logs/trading.log

# View backtest results
import os
if os.path.exists('backtest_results'):
    !ls -lh backtest_results/
```

## Troubleshooting

### Issue 1: Directory Not Found

**Error**: `[Errno 2] No such file or directory: 'stuff/autonomous-trading-system'`

**Solution**:
```python
# Ensure you're in the correct directory
%cd /kaggle/working/autonomous-trading-system
!pwd  # Verify path
!python3 launcher.py
```

### Issue 2: pandas-ta Installation Failed

**Error**: `ERROR: Could not find a version that satisfies the requirement pandas-ta>=0.3.14b`

**Solution**:
```python
# Install specific version
!pip install pandas-ta==0.3.14b0

# Or try latest
!pip install pandas-ta
```

### Issue 3: Missing apscheduler

**Error**: `ModuleNotFoundError: No module named 'apscheduler'`

**Solution**:
```python
!pip install apscheduler>=3.10.0
```

### Issue 4: Config Files Missing

**Error**: `sed: can't read config/scheduler.yaml`

**Solution**: The launcher auto-creates missing configs. If issues persist:
```python
# Manually create config directory
!mkdir -p config

# Run launcher (auto-creates configs)
!python3 launcher.py
```

### Issue 5: Session Timeout

**Problem**: Kaggle notebook times out after 12 hours

**Solution**: Enable keep-alive mode:
```python
import os
os.environ['KEEP_ALIVE'] = 'true'
os.environ['KEEP_ALIVE_INTERVAL'] = '300'

!python3 launcher.py
```

### Issue 6: Internet Access Denied

**Error**: `URLError` or connection errors

**Solution**:
1. Go to notebook Settings
2. Enable "Internet" toggle
3. Restart notebook
4. Re-run cells

### Issue 7: GPU Not Available

**Warning**: System runs slower without GPU

**Solution**:
1. Settings → Accelerator → GPU T4 x2
2. Save and restart notebook

### Issue 8: Import Errors

**Error**: `ImportError: No module named 'src'`

**Solution**:
```python
import sys
sys.path.insert(0, '/kaggle/working/autonomous-trading-system')

# Now import
from main import TradingSystem
```

## Performance Optimization

### Memory Management

```python
# Clear memory between runs
import gc
gc.collect()

# Limit symbols for faster processing
os.environ['MAX_SYMBOLS'] = '10'
```

### Speed Optimization

```python
# Use GPU acceleration (if available)
os.environ['USE_GPU'] = 'true'

# Reduce indicator calculations
os.environ['FAST_MODE'] = 'true'
```

## Verification Tests

### Test 1: Installation Verification

```python
# Verify all modules
import pandas
import numpy
import yfinance
import yaml
import apscheduler
from dotenv import load_dotenv

print("✓ All core modules available")

# Verify pandas-ta
try:
    import pandas_ta
    print("✓ pandas-ta available")
except ImportError:
    print("⚠ pandas-ta not available (optional)")
```

### Test 2: Indicator Calculation

```python
# Test technical indicators
import sys
sys.path.insert(0, '/kaggle/working/autonomous-trading-system')

from src.indicators.technical_indicators import TechnicalIndicators
import yfinance as yf

# Fetch test data
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1mo")

# Calculate indicators
indicators = TechnicalIndicators()
result = indicators.calculate_all(data)

print(f"✓ Calculated indicators for {len(result)} bars")
print(f"✓ Columns: {list(result.columns)}")
```

### Test 3: Backtest Run

```python
# Run quick backtest
import sys
sys.path.insert(0, '/kaggle/working/autonomous-trading-system')

from main import TradingSystem

system = TradingSystem()
system.run_backtest()

print("✓ Backtest completed successfully")
```

## Best Practices

### 1. Save Work Regularly

```python
# Save important results
import shutil
shutil.copy('logs/trading.log', '/kaggle/working/trading_log_backup.txt')
```

### 2. Use Version Control

```python
# Commit changes
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"
!git add .
!git commit -m "Kaggle run results"
```

### 3. Monitor Resources

```python
# Check memory usage
!free -h

# Check disk usage
!df -h

# Check running processes
!ps aux | grep python
```

### 4. Log Everything

```python
# Enable verbose logging
os.environ['LOG_LEVEL'] = 'DEBUG'

# View logs in real-time
!tail -f logs/trading.log
```

## Sample Kaggle Notebook Script

Save this as a complete notebook:

```python
# ============================================
# Autonomous Trading System - Kaggle Notebook
# ============================================

# CELL 1: Installation
print("Installing Autonomous Trading System...")
!git clone https://github.com/your-username/autonomous-trading-system.git
%cd autonomous-trading-system
!python3 install.py

# CELL 2: Configuration
import os
os.environ['KAGGLE_MODE'] = 'true'
os.environ['KEEP_ALIVE'] = 'true'
os.environ['TRADING_MODE'] = 'paper'
print("✓ Configured")

# CELL 3: Verification
import pandas, numpy, yfinance, yaml, apscheduler
print("✓ All modules available")

# CELL 4: Launch
print("Launching system...")
!python3 launcher.py

# CELL 5: View Results
print("\n=== Trading Log ===")
!tail -n 100 logs/trading.log

print("\n=== Backtest Results ===")
!ls -lh backtest_results/ 2>/dev/null || echo "No backtest results yet"
```

## Additional Resources

- **Main README**: See `README.md` for general documentation
- **API Documentation**: See `docs/API.md` for API details
- **Alpaca API**: https://alpaca.markets/docs/
- **Kaggle Documentation**: https://www.kaggle.com/docs/

## Support

If you encounter issues not covered here:

1. Check the main README.md troubleshooting section
2. Review logs in `logs/trading.log`
3. Open an issue on GitHub with:
   - Error message
   - Kaggle notebook link
   - Python version (`!python3 --version`)
   - Installed packages (`!pip list`)

## Success Checklist

Before running, verify:

- [ ] Internet access enabled in Kaggle settings
- [ ] GPU accelerator selected (T4 x2 recommended)
- [ ] Repository cloned successfully
- [ ] Dependencies installed without errors
- [ ] Configuration set (KAGGLE_MODE=true)
- [ ] Directory is `autonomous-trading-system/`
- [ ] Python version is 3.10+ (`!python3 --version`)

## Expected Output

Successful run should show:

```
==========================================================
Autonomous Trading System Launcher
==========================================================
Python version: 3.10.x
✓ Python version OK

[1/5] Creating directories...
✓ Directory ensured: config/
✓ Directory ensured: data/
✓ Directory ensured: logs/

[2/5] Checking configuration files...
✓ Config files present

[3/5] Checking dependencies...
✓ pandas is installed
✓ numpy is installed
✓ yfinance is installed
✓ apscheduler is installed

[4/5] Setting up environment...
Kaggle mode detected - applying optimizations...

[5/5] Launching main application...
==========================================================
AUTONOMOUS TRADING SYSTEM
==========================================================
Mode: paper
Symbols: AAPL, MSFT, GOOGL
==========================================================

Running initial backtest...
Analyzing AAPL...
✓ Fetched 30 bars for AAPL
✓ Analysis complete for AAPL
Price: $XXX.XX
RSI: XX.XX
MACD: X.XX

✓ System is running
```

## Conclusion

This guide covers all common scenarios for running the Autonomous Trading System on Kaggle. The system is designed to work out-of-the-box with zero configuration errors. All dependencies are fixed, configs are auto-created, and error handling is comprehensive.

Happy trading! 🚀
