# Migration Guide: Repository Restructure

## Overview

The repository has been restructured to fix directory navigation issues. All project files have been moved into an `autonomous-trading-system/` subfolder.

## What Changed

### Before (Old Structure)
```
stuff/
├── README.md
├── launcher.py
├── main.py
├── src/
├── config/
├── docs/
└── ...
```

### After (New Structure)
```
stuff/
├── README.md                    # Quick start guide (NEW)
├── setup.sh                     # One-command setup (NEW)
└── autonomous-trading-system/   # All project files (MOVED)
    ├── README.md                # Full documentation
    ├── launcher.py              # Main launcher
    ├── main.py                  # Entry point
    ├── src/                     # Source code
    ├── config/                  # Configuration
    └── docs/                    # Documentation
```

## Why This Change?

**Problem**: Users cloning the repository got all files at the root level, but documentation said to run `cd stuff/autonomous-trading-system`, which didn't exist. This caused:
- `[Errno 2] No such file or directory: 'stuff/autonomous-trading-system'`
- Confusion about where to run commands
- Kaggle/Colab setup failures

**Solution**: All project files are now in the `autonomous-trading-system/` subfolder, matching the documented instructions.

## Migration for Existing Users

If you have an existing clone of the repository, follow these steps:

### Option 1: Pull Latest Changes (Recommended)

```bash
# Navigate to your existing clone
cd /path/to/your/stuff

# Pull latest changes
git pull origin devin/1761207403-autonomous-trading-system

# Navigate to the new subfolder
cd autonomous-trading-system

# Continue working as normal
python launcher.py --set-and-forget --capital 100000
```

### Option 2: Fresh Clone (Clean Start)

```bash
# Remove old clone
rm -rf /path/to/your/stuff

# Clone fresh
git clone https://github.com/ryanhamil7-collab/stuff.git
cd stuff/autonomous-trading-system

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Launch
python launcher.py --set-and-forget --capital 100000
```

### Option 3: Use Setup Script (Easiest)

```bash
# Clone repository
git clone https://github.com/ryanhamil7-collab/stuff.git
cd stuff

# Run automated setup
bash setup.sh
```

## Updated Commands

All commands remain the same, but you must run them from the `autonomous-trading-system/` directory:

### Before
```bash
cd stuff
python launcher.py --set-and-forget
```

### After
```bash
cd stuff/autonomous-trading-system
python launcher.py --set-and-forget
```

## Kaggle/Colab Users

The new structure fixes the Kaggle/Colab setup issues:

### Old Instructions (Broken)
```python
!git clone https://github.com/ryanhamil7-collab/stuff.git
%cd stuff/autonomous-trading-system  # ❌ This failed
```

### New Instructions (Working)
```python
!git clone https://github.com/ryanhamil7-collab/stuff.git
%cd stuff/autonomous-trading-system  # ✅ This works now!
!pip install -r requirements.txt
!python launcher.py --set-and-forget --capital 1000
```

## What Hasn't Changed

- **All functionality**: The system works exactly the same
- **All 21 features**: Every feature is preserved
- **Configuration**: All config files are in the same relative locations
- **Commands**: All commands are identical (just run from subfolder)
- **File paths**: All internal relative paths remain unchanged

## Troubleshooting

### Issue: "No such file or directory: autonomous-trading-system"

**Cause**: You haven't pulled the latest changes or you're in the wrong directory.

**Solution**:
```bash
# Make sure you're in the stuff repo root
cd /path/to/stuff

# Pull latest changes
git pull origin devin/1761207403-autonomous-trading-system

# Navigate to subfolder
cd autonomous-trading-system
```

### Issue: "Module not found" errors

**Cause**: You're running commands from the wrong directory.

**Solution**: Always run commands from `stuff/autonomous-trading-system/`:
```bash
cd stuff/autonomous-trading-system
python main.py backtest --symbols AAPL
```

### Issue: Virtual environment not working

**Cause**: Virtual environment was created in old location.

**Solution**: Recreate virtual environment in new location:
```bash
cd stuff/autonomous-trading-system
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Config files not found

**Cause**: Config files are now in `autonomous-trading-system/config/`.

**Solution**: Make sure you're in the right directory:
```bash
cd stuff/autonomous-trading-system
ls config/  # Should show config.yaml and scheduler.yaml
```

## Benefits of New Structure

1. **Matches Documentation**: Instructions now match actual directory structure
2. **Cleaner Root**: Root directory is clean with just README and setup script
3. **Easier Setup**: One-command setup with `setup.sh`
4. **Better Organization**: Clear separation between root-level docs and project files
5. **Kaggle/Colab Compatible**: Works seamlessly in notebook environments

## Questions?

If you encounter any issues with the migration:

1. Check this guide first
2. Try a fresh clone (Option 2 above)
3. Use the automated setup script (Option 3 above)
4. Open an issue on GitHub: https://github.com/ryanhamil7-collab/stuff/issues

## Python 3.10 Compatibility Fix

### pandas-ta Dependency Issue

**Problem**: Newer versions of pandas-ta (>=0.4.67b0) require Python 3.12+, causing installation errors on Python 3.10 environments (Kaggle, Colab):
```
ERROR: Could not find a version that satisfies the requirement pandas-ta>=0.3.14b
```

**Solution**: The system now uses `pandas-ta==0.3.14b0` (pinned version) for Python 3.10 compatibility.

### Automatic Handling

The `setup.sh` script automatically detects your Python version and installs the correct pandas-ta version:

```bash
# For Python 3.10-3.11
pip install pandas-ta==0.3.14b0

# For Python 3.12+
pip install pandas-ta  # Latest version
```

If `pandas-ta==0.3.14b0` fails, the script automatically tries the fallback:
```bash
pip install pandas-ta-openbb==0.4.22  # Compatible fork
```

### Manual Installation

If you encounter pandas-ta installation errors:

```bash
# Check Python version
python --version

# For Python 3.10-3.11, use pinned version
pip install pandas-ta==0.3.14b0

# Or use the compatible fork
pip install pandas-ta-openbb==0.4.22
```

### Kaggle/Colab Specific

For Kaggle/Colab notebooks (Python 3.10):

```python
# Install compatible version
!pip install pandas-ta==0.3.14b0

# Or use fallback
!pip install pandas-ta-openbb==0.4.22

# Then install other dependencies
!pip install -r requirements.txt
```

### Verification

Test that pandas-ta is working:

```python
import pandas_ta as ta
print(f"pandas-ta version: {ta.__version__}")

# Test RSI calculation
import pandas as pd
df = pd.DataFrame({'Close': [100, 102, 101, 103, 105]})
rsi = ta.rsi(df['Close'], length=14)
print(f"RSI calculation works: {rsi is not None}")
```

### Technical Details

- **Compatible Version**: `pandas-ta==0.3.14b0` (Python 3.10-3.11)
- **Fallback Version**: `pandas-ta-openbb==0.4.22` (NumPy 2 compatible fork)
- **Latest Version**: `pandas-ta>=0.4.67b0` (Python 3.12+ only)

All 21 features work correctly with both pandas-ta versions. The system includes error handling in `src/utils/indicators.py` to gracefully handle pandas-ta import failures.

## Summary

**TL;DR**: After pulling latest changes, run `cd autonomous-trading-system` before running any commands. Everything else stays the same.

```bash
# Quick migration
cd /path/to/your/stuff
git pull
cd autonomous-trading-system
python launcher.py --set-and-forget
```

**Python 3.10 Users**: The pandas-ta dependency is now fixed. Use `pandas-ta==0.3.14b0` or run `bash setup.sh` for automatic installation.

That's it! 🚀
