"""
Autonomous Trading System - Kaggle Notebook Script
Complete script for running on Kaggle with zero errors.

Copy this entire file into a Kaggle notebook or run cell by cell.
"""


def install_system():
    """Install the autonomous trading system."""
    print("=" * 60)
    print("AUTONOMOUS TRADING SYSTEM - KAGGLE INSTALLATION")
    print("=" * 60)
    
    import subprocess
    import os
    
    repo_url = "https://github.com/your-username/autonomous-trading-system.git"
    
    print("\n[1/3] Cloning repository...")
    try:
        subprocess.run(['git', 'clone', repo_url], check=True)
        print("✓ Repository cloned")
    except subprocess.CalledProcessError:
        print("⚠ Repository already exists or clone failed")
    
    os.chdir('/kaggle/working/autonomous-trading-system')
    print(f"✓ Changed to: {os.getcwd()}")
    
    print("\n[2/3] Installing dependencies...")
    result = subprocess.run(['python3', 'install.py'], capture_output=False)
    
    if result.returncode == 0:
        print("✓ Installation complete")
    else:
        print("⚠ Installation had warnings (may still work)")
    
    print("\n[3/3] Verifying installation...")
    try:
        import pandas
        import numpy
        import yfinance
        import yaml
        import apscheduler
        print("✓ All core modules available")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False



def configure_environment(api_key=None, secret_key=None):
    """Configure environment variables."""
    import os
    
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    
    os.environ['KAGGLE_MODE'] = 'true'
    os.environ['KEEP_ALIVE'] = 'true'
    os.environ['TRADING_MODE'] = 'paper'
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['INITIAL_CAPITAL'] = '1000'
    
    print("✓ Basic configuration set")
    
    if api_key and secret_key:
        os.environ['ALPACA_API_KEY'] = api_key
        os.environ['ALPACA_SECRET_KEY'] = secret_key
        print("✓ API keys configured")
    else:
        print("⚠ No API keys provided (backtesting only)")
    
    print("\nEnvironment:")
    print(f"  - Mode: {os.environ['TRADING_MODE']}")
    print(f"  - Capital: ${os.environ['INITIAL_CAPITAL']}")
    print(f"  - Kaggle Mode: {os.environ['KAGGLE_MODE']}")
    print(f"  - Keep Alive: {os.environ['KEEP_ALIVE']}")



def launch_system(mode='backtest'):
    """Launch the trading system."""
    import sys
    import os
    
    print("\n" + "=" * 60)
    print("LAUNCHING SYSTEM")
    print("=" * 60)
    
    os.chdir('/kaggle/working/autonomous-trading-system')
    sys.path.insert(0, os.getcwd())
    
    if mode == 'backtest':
        print("\nRunning in BACKTEST mode...")
        from main import TradingSystem
        
        system = TradingSystem()
        system.run_backtest()
        
    elif mode == 'paper':
        print("\nRunning in PAPER TRADING mode...")
        from main import TradingSystem
        
        system = TradingSystem()
        system.start_paper_trading()
        
    elif mode == 'full':
        print("\nRunning FULL system...")
        import subprocess
        subprocess.run(['python3', 'launcher.py'])
    
    else:
        print(f"Unknown mode: {mode}")
        return False
    
    return True



def view_results():
    """View system results and logs."""
    import os
    import subprocess
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    log_file = 'logs/trading.log'
    if os.path.exists(log_file):
        print("\n=== Recent Log Entries ===")
        subprocess.run(['tail', '-n', '50', log_file])
    else:
        print("⚠ No log file found")
    
    results_dir = 'backtest_results'
    if os.path.exists(results_dir):
        print("\n=== Backtest Results ===")
        subprocess.run(['ls', '-lh', results_dir])
    else:
        print("⚠ No backtest results found")
    
    print("\n=== System Status ===")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python Version: {subprocess.run(['python3', '--version'], capture_output=True, text=True).stdout.strip()}")



def quick_test():
    """Run a quick test of the system."""
    import sys
    import os
    
    print("\n" + "=" * 60)
    print("QUICK TEST")
    print("=" * 60)
    
    os.chdir('/kaggle/working/autonomous-trading-system')
    sys.path.insert(0, os.getcwd())
    
    try:
        print("\n[Test 1] Importing modules...")
        from src.indicators.technical_indicators import TechnicalIndicators
        print("✓ Indicators module imported")
        
        print("\n[Test 2] Fetching market data...")
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo")
        print(f"✓ Fetched {len(data)} bars for AAPL")
        
        print("\n[Test 3] Calculating indicators...")
        indicators = TechnicalIndicators()
        result = indicators.calculate_all(data)
        print(f"✓ Calculated {len(result.columns)} indicators")
        
        print("\n[Test 4] Latest data:")
        latest = result.iloc[-1]
        print(f"  Close: ${latest['Close']:.2f}")
        if 'RSI' in result.columns:
            print(f"  RSI: {latest['RSI']:.2f}")
        if 'MACD' in result.columns:
            print(f"  MACD: {latest['MACD']:.2f}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False



def keep_alive_loop(interval=300):
    """Keep Kaggle session alive."""
    import time
    
    print("\n" + "=" * 60)
    print("KEEP-ALIVE MODE")
    print("=" * 60)
    print(f"Pinging every {interval} seconds...")
    print("Press Ctrl+C to stop")
    
    try:
        counter = 0
        while True:
            time.sleep(interval)
            counter += 1
            print(f"[{counter}] Keep-alive ping at {time.strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("\nKeep-alive stopped")



def main():
    """Main execution function."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   AUTONOMOUS TRADING SYSTEM - KAGGLE NOTEBOOK           ║
    ║   Version 2.0 - Zero Setup Errors                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if not install_system():
        print("\n❌ Installation failed. Please check errors above.")
        return
    
    configure_environment()
    
    
    if not quick_test():
        print("\n⚠ Quick test failed, but continuing...")
    
    launch_system(mode='backtest')
    
    view_results()
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review results above")
    print("2. Check logs: !cat logs/trading.log")
    print("3. Run keep_alive_loop() to prevent timeout")
    print("4. Customize config files in config/")



"""
USAGE IN KAGGLE NOTEBOOK:

main()

install_system()
configure_environment()
quick_test()
launch_system(mode='backtest')
view_results()

install_system()
configure_environment(
    api_key='your_key',
    secret_key='your_secret'
)
launch_system(mode='paper')

main()
keep_alive_loop(interval=300)

install_system()
configure_environment()
quick_test()
"""



if __name__ == "__main__":
    
    print("Kaggle notebook script loaded!")
    print("Run: main() to start")
    print("Or run functions individually:")
    print("  - install_system()")
    print("  - configure_environment()")
    print("  - quick_test()")
    print("  - launch_system(mode='backtest')")
    print("  - view_results()")
    print("  - keep_alive_loop()")
