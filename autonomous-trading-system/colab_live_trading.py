#!/usr/bin/env python3
"""
Colab-friendly Live Paper Trading Launcher

This script is optimized for Google Colab with A100 GPU.
It will:
1. Test Alpaca connection
2. Display current portfolio
3. Launch autopilot in paper trading mode
4. Show real-time trading activity

Usage in Colab:
!python3 colab_live_trading.py
"""

import sys
import os
from pathlib import Path

os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('CUDA_HOME', '')}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['PATH'] = f"{os.environ.get('CUDA_HOME', '')}/bin:{os.environ.get('PATH', '')}"

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from src.utils import log

load_dotenv()

def test_alpaca_connection():
    """Test connection to Alpaca Paper Trading API"""
    log.info("=" * 80)
    log.info("TESTING ALPACA CONNECTION")
    log.info("=" * 80)
    
    try:
        import alpaca_trade_api as tradeapi
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            log.error("❌ Missing Alpaca API keys in .env file")
            return False
        
        log.info(f"API Key: {api_key[:8]}...")
        log.info(f"Base URL: {base_url}")
        
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        account = api.get_account()
        
        log.info("✓ Connection successful!")
        log.info("")
        log.info("ACCOUNT DETAILS:")
        log.info(f"  Account Number: {account.account_number}")
        log.info(f"  Status: {account.status}")
        log.info(f"  Cash: ${float(account.cash):,.2f}")
        log.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        log.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
        log.info(f"  Day Trade Count: {account.daytrade_count}")
        log.info("")
        
        positions = api.list_positions()
        
        if positions:
            log.info(f"CURRENT POSITIONS ({len(positions)}):")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                log.info(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f} "
                        f"(P&L: ${pnl:,.2f} / {pnl_pct:.2f}%)")
        else:
            log.info("CURRENT POSITIONS: None")
        
        log.info("")
        log.info("=" * 80)
        return True
        
    except Exception as e:
        log.error(f"❌ Connection failed: {str(e)}")
        return False

def launch_autopilot():
    """Launch autopilot in paper trading mode"""
    log.info("=" * 80)
    log.info("LAUNCHING AUTOPILOT - PAPER TRADING MODE")
    log.info("=" * 80)
    log.info("")
    log.info("Configuration:")
    log.info("  • Mode: Paper Trading (Alpaca)")
    log.info("  • Check Interval: 5 minutes")
    log.info("  • GPU: A100 80GB")
    log.info("  • Models: 3-model ensemble (Mixtral-8x7B, Llama-3-8B, Mistral-7B)")
    log.info("  • Batch Size: 32 symbols")
    log.info("  • Live RL Training: Enabled")
    log.info("")
    log.info("What will happen:")
    log.info("  1. Fetch market data every 5 minutes")
    log.info("  2. Analyze symbols using 3 LLM models")
    log.info("  3. Generate trading signals with ensemble voting")
    log.info("  4. Execute trades through Alpaca Paper API")
    log.info("  5. Learn from trade outcomes in real-time")
    log.info("")
    log.info("Monitor trades at:")
    log.info("  https://app.alpaca.markets/paper/dashboard/overview")
    log.info("")
    log.info("Press Ctrl+C to stop")
    log.info("=" * 80)
    log.info("")
    
    try:
        from src.autopilot import AutopilotDaemon
        
        autopilot = AutopilotDaemon(use_alpaca=True)
        
        autopilot.run_continuous()
        
    except KeyboardInterrupt:
        log.info("")
        log.info("=" * 80)
        log.info("AUTOPILOT STOPPED BY USER")
        log.info("=" * 80)
    except Exception as e:
        log.error(f"Error running autopilot: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    log.info("")
    log.info("=" * 80)
    log.info("AUTONOMOUS TRADING SYSTEM - LIVE PAPER TRADING")
    log.info("=" * 80)
    log.info("")
    
    if not test_alpaca_connection():
        log.error("Cannot proceed without valid Alpaca connection")
        sys.exit(1)
    
    launch_autopilot()

if __name__ == "__main__":
    main()
