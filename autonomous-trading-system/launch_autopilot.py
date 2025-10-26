#!/usr/bin/env python3
"""
Quick launcher for Autopilot mode with Alpaca Paper Trading

This script simplifies launching the autopilot with Alpaca integration.
Just run: python3 launch_autopilot.py

Requirements:
- .env file with Alpaca API keys
- pip install alpaca-trade-api python-dotenv
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from src.utils import log

load_dotenv()

def check_env():
    """Check if required environment variables are set"""
    required = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    missing = [key for key in required if not os.getenv(key)]
    
    if missing:
        log.error("Missing required environment variables:")
        for key in missing:
            log.error(f"  - {key}")
        log.error("\nPlease create a .env file with your Alpaca API keys:")
        log.error("  ALPACA_API_KEY=your_key_here")
        log.error("  ALPACA_SECRET_KEY=your_secret_here")
        log.error("  ALPACA_BASE_URL=https://paper-api.alpaca.markets")
        return False
    
    return True

def main():
    log.info("=" * 80)
    log.info("AUTONOMOUS TRADING SYSTEM - AUTOPILOT LAUNCHER")
    log.info("=" * 80)
    
    if not check_env():
        sys.exit(1)
    
    log.info("✓ Environment variables loaded")
    log.info("🚀 Launching Autopilot with Alpaca Paper Trading...")
    log.info("=" * 80)
    log.info("")
    log.info("What will happen:")
    log.info("  1. Connect to Alpaca Paper Trading API")
    log.info("  2. Fetch market data every 5 minutes")
    log.info("  3. Analyze symbols and generate trading signals")
    log.info("  4. Execute trades through Alpaca")
    log.info("  5. Display portfolio status after each cycle")
    log.info("")
    log.info("You can watch trades in real-time at:")
    log.info("  https://app.alpaca.markets/paper/dashboard/overview")
    log.info("")
    log.info("Press Ctrl+C to stop")
    log.info("=" * 80)
    log.info("")
    
    from src.autopilot import AutopilotDaemon
    
    autopilot = AutopilotDaemon(use_alpaca=True)
    autopilot.run_continuous()

if __name__ == "__main__":
    main()
