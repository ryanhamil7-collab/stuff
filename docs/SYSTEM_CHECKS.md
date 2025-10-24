# System Checks Documentation

Comprehensive pre-flight validation for the Autonomous Trading System.

## Overview

The system checks module (`src/utils/system_checks.py`) performs 10 critical validations before launching the trading system. This ensures:
- All dependencies are properly installed
- API keys are configured
- System resources are sufficient
- Environment is properly set up
- No runtime failures due to missing configuration

## Running System Checks

### Method 1: Standalone Checks

```bash
# Run all checks
python3 src/utils/system_checks.py

# Run checks only (don't launch system)
python3 run_with_checks.py --checks-only
```

### Method 2: With Launcher

```bash
# Launch with automatic pre-flight checks (recommended)
python3 run_with_checks.py --set-and-forget --capital 100000

# Skip checks (not recommended)
python3 run_with_checks.py --skip-checks --set-and-forget --capital 100000

# Strict mode (treat warnings as errors)
python3 run_with_checks.py --strict --set-and-forget --capital 100000
```

## Check Details

### Check 1: Python Version

**What it checks:**
- Python version is 3.10 or higher

**Why it matters:**
- The system uses modern Python features (match/case, type hints, etc.)
- Dependencies require Python 3.10+

**How to fix:**
```bash
# Check your Python version
python3 --version

# If < 3.10, upgrade Python
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3.11

# macOS (Homebrew):
brew install python@3.11
```

### Check 2: API Keys

**What it checks:**
- Required API keys: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- Optional API keys: `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `HUGGINGFACE_TOKEN`

**Why it matters:**
- Alpaca keys required for paper trading
- Optional keys enable additional data sources and LLM models

**How to fix:**
```bash
# Copy example .env file
cp .env.example .env

# Edit .env and add your keys
nano .env  # or use your preferred editor

# Get free API keys:
# - Alpaca: https://alpaca.markets (paper trading)
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
# - Finnhub: https://finnhub.io/register
# - Hugging Face: https://huggingface.co/settings/tokens
```

### Check 3: GPU/CUDA Availability

**What it checks:**
- PyTorch installation
- CUDA availability
- GPU device count and names
- CUDA version

**Why it matters:**
- GPU acceleration speeds up LLM inference 10-50x
- Some features (auto-gptq) require CUDA

**Expected output:**
```
✅ GPU detected: NVIDIA Tesla T4 (CUDA 11.8)
ℹ️  Using GPU acceleration for LLM inference
```

**If no GPU:**
```
⚠️  No GPU detected. System will use CPU (slower performance).
    For Kaggle, enable GPU in notebook settings.
```

**How to enable GPU on Kaggle:**
1. Open notebook settings
2. Accelerator → GPU T4 x2
3. Save and restart notebook

### Check 4: System Resources

**What it checks:**
- Available RAM (recommended: 16GB+)
- CPU cores (recommended: 4+)
- Free disk space (minimum: 10GB)

**Why it matters:**
- LLM models require significant RAM (8-16GB)
- Multi-agent architecture benefits from multiple CPU cores
- Models and data require disk space

**Expected output:**
```
✅ RAM: 20.5GB available
✅ CPU: 8 cores
✅ Disk: 45.2GB free
```

**If insufficient resources:**
```
⚠️  RAM: 7.2GB available (recommended: 16GB+).
    May have issues with large models.
```

**How to fix:**
- **Kaggle**: Use notebooks with more RAM (20GB available)
- **Local**: Close other applications or upgrade RAM
- **Cloud**: Use larger instance types

### Check 5: Internet Connectivity

**What it checks:**
- Connection to google.com
- Connection to api.alpaca.markets
- Connection to www.alphavantage.co

**Why it matters:**
- Required for fetching market data
- Required for downloading LLM models
- Required for API calls

**How to fix on Kaggle:**
1. Notebook settings → Internet → ON
2. Save and restart notebook

### Check 6: Directory Structure

**What it checks:**
- Required directories exist: `data/`, `logs/`, `config/`, `backtest_results/`, `models/`

**Why it matters:**
- System writes logs, data, and results to these directories
- Missing directories cause file write errors

**Auto-fix:**
The check automatically creates missing directories.

### Check 7: Configuration Files

**What it checks:**
- `.env` file exists
- `config/config.yaml` exists
- `config/scheduler.yaml` exists

**Why it matters:**
- Configuration files control system behavior
- Missing files cause startup errors

**Auto-fix:**
The launcher automatically creates missing config files with defaults.

### Check 8: Package Versions

**What it checks:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- torch >= 2.1.0
- transformers >= 4.40.0

**Why it matters:**
- Ensures compatibility with Python 3.11+
- Prevents API incompatibilities

**How to fix:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Check 9: Model Access

**What it checks:**
- Models directory exists
- Cached models present
- Hugging Face token configured

**Why it matters:**
- LLM models are large (2-8GB)
- First run downloads models (can take 10-30 minutes)
- Some models require authentication

**Expected output:**
```
✅ Found 3 cached models
✅ Hugging Face token configured
```

**If no models:**
```
ℹ️  No cached models found (will download on first run)
```

### Check 10: Kaggle Environment

**What it checks:**
- `KAGGLE_KERNEL_RUN_TYPE` environment variable
- Kernel type (interactive/batch)
- Internet enabled
- GPU enabled

**Why it matters:**
- Kaggle has specific limitations and optimizations
- Enables Kaggle-specific features

**Expected output on Kaggle:**
```
✅ Running in Kaggle (Interactive)
✅ Internet connection verified
✅ Kaggle GPU enabled
```

**Expected output locally:**
```
ℹ️  Not running in Kaggle environment
```

## Check Results

### Success (All Checks Pass)

```
======================================================================
SUMMARY
======================================================================

✅ All critical checks passed!
System is ready to launch.
```

### Warnings (Non-Critical Issues)

```
======================================================================
SUMMARY
======================================================================

⚠️  Warnings: 3
  - PyTorch not installed, GPU detection skipped
  - torch not installed (optional)
  - Hugging Face token not set. May be required for some models.

✅ All critical checks passed!
System is ready to launch.
```

### Errors (Critical Issues)

```
======================================================================
SUMMARY
======================================================================

❌ Errors: 1
  - Missing required API keys: ALPACA_API_KEY, ALPACA_SECRET_KEY. Set them in .env file.

❌ Some checks failed. Please fix errors before launching.
```

## Strict Mode

In strict mode, warnings are treated as errors:

```bash
python3 run_with_checks.py --strict --checks-only
```

Use strict mode when:
- Deploying to production
- Running automated tests
- Ensuring maximum reliability

## Skipping Checks

You can skip checks (not recommended):

```bash
python3 run_with_checks.py --skip-checks --set-and-forget --capital 100000
```

Skip checks only when:
- You've already verified the environment
- Running in a controlled environment
- Debugging specific issues

## Integration with CI/CD

Add system checks to your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
- name: Run System Checks
  run: |
    python3 src/utils/system_checks.py
    if [ $? -ne 0 ]; then
      echo "System checks failed"
      exit 1
    fi
```

## Troubleshooting

### Check Fails: "psutil not installed"

```bash
pip install psutil>=5.9.0
```

### Check Fails: "requests library not available"

```bash
pip install requests>=2.31.0
```

### Check Fails: "PyTorch not installed"

```bash
# CPU-only
pip install torch>=2.1.0

# With CUDA 11.8
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Check Fails: "No internet connection"

**Kaggle:**
1. Settings → Internet → ON
2. Restart notebook

**Local:**
1. Check firewall settings
2. Check proxy configuration
3. Test with: `ping google.com`

### Check Fails: "Insufficient disk space"

**Kaggle:**
- Kaggle notebooks have 20GB disk space
- Clear old outputs and data files

**Local:**
```bash
# Check disk usage
df -h

# Clear pip cache
pip cache purge

# Clear old models
rm -rf models/*.old
```

## API Reference

### SystemChecker Class

```python
from src.utils.system_checks import SystemChecker

# Create checker
checker = SystemChecker(strict_mode=False)

# Run all checks
success = checker.run_all_checks(skip_optional=False)

# Access results
print(f"Errors: {checker.errors}")
print(f"Warnings: {checker.warnings}")
print(f"Info: {checker.info}")
```

### Convenience Function

```python
from src.utils.system_checks import run_system_checks

# Run all checks
success = run_system_checks(
    strict_mode=False,
    skip_optional=False
)
```

## Best Practices

1. **Always run checks before deployment**
   - Prevents runtime failures
   - Catches configuration issues early

2. **Review warnings**
   - Warnings indicate suboptimal configuration
   - May impact performance or features

3. **Use strict mode for production**
   - Ensures highest reliability
   - Catches all potential issues

4. **Automate checks in CI/CD**
   - Prevents deploying broken configurations
   - Maintains code quality

5. **Keep dependencies updated**
   - Run `pip install -r requirements.txt --upgrade` regularly
   - Check for security updates

## Related Documentation

- [Installation Guide](../README.md#installation)
- [Kaggle Setup](KAGGLE_SETUP.md)
- [Python 3.11 Fixes](../PYTHON311_FIXES.md)
- [Troubleshooting](../README.md#troubleshooting)
