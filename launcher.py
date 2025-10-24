#!/usr/bin/env python3
"""
Autonomous Trading System Launcher
Robust launcher with error handling for Kaggle/Colab compatibility.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemLauncher:
    """Launcher with comprehensive error handling and environment setup."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.config_dir = self.base_dir / "config"
        self.src_dir = self.base_dir / "src"
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            logger.error("Python 3.10 or higher is required!")
            logger.error(f"Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        
        if version.major == 3 and version.minor > 12:
            logger.warning(f"Python {version.major}.{version.minor} is newer than tested. May have compatibility issues.")
        
        return True
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.base_dir / "backtest_results"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
    
    def check_config_files(self) -> bool:
        """Check and create config files if missing."""
        config_file = self.config_dir / "config.yaml"
        scheduler_file = self.config_dir / "scheduler.yaml"
        env_file = self.base_dir / ".env"
        
        all_exist = True
        
        if not config_file.exists():
            logger.warning(f"Missing config file: {config_file}")
            logger.info("Creating default config.yaml...")
            self.create_default_config()
            all_exist = False
        
        if not scheduler_file.exists():
            logger.warning(f"Missing scheduler file: {scheduler_file}")
            logger.info("Creating default scheduler.yaml...")
            self.create_default_scheduler()
            all_exist = False
        
        if not env_file.exists():
            logger.warning(f"Missing .env file: {env_file}")
            logger.info("Creating .env from .env.example...")
            self.create_env_file()
            all_exist = False
        
        return all_exist
    
    def create_default_config(self) -> None:
        """Create default config.yaml if missing."""
        default_config = """# Main Configuration for Autonomous Trading System

api:
  alpaca:
    base_url: "https://paper-api.alpaca.markets"
    api_key: "${ALPACA_API_KEY}"
    secret_key: "${ALPACA_SECRET_KEY}"
    paper_trading: true

trading:
  mode: "paper"
  symbols:
    - "AAPL"
    - "MSFT"
    - "GOOGL"

dashboard:
  enabled: true
  port: 5000
  host: "0.0.0.0"

features:
  backtesting: true
  paper_trading: true
  risk_management: true
"""
        config_file = self.config_dir / "config.yaml"
        config_file.write_text(default_config)
        logger.info(f"Created default config: {config_file}")
    
    def create_default_scheduler(self) -> None:
        """Create default scheduler.yaml if missing."""
        default_scheduler = """# Scheduler Configuration

market_hours:
  start: "09:30"
  end: "16:00"
  timezone: "America/New_York"

hive_mind:
  enabled: false

capital:
  initial: 1000

symbol_discovery:
  enabled: true
  max_symbols: 50

risk_management:
  max_daily_loss: 0.02
  stop_loss_percentage: 0.02

kaggle:
  keep_alive: true
  keep_alive_interval: 300
"""
        scheduler_file = self.config_dir / "scheduler.yaml"
        scheduler_file.write_text(default_scheduler)
        logger.info(f"Created default scheduler: {scheduler_file}")
    
    def create_env_file(self) -> None:
        """Create .env file from example."""
        env_example = self.base_dir / ".env.example"
        env_file = self.base_dir / ".env"
        
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            logger.info(f"Created .env from .env.example")
        else:
            default_env = """ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
TRADING_MODE=paper
INITIAL_CAPITAL=1000
LOG_LEVEL=INFO
KAGGLE_MODE=true
"""
            env_file.write_text(default_env)
            logger.info(f"Created default .env file")
        
        logger.warning("Please update .env with your actual API keys!")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_modules = [
            'pandas',
            'numpy',
            'yfinance',
            'yaml',
            'dotenv',
            'flask',
            'apscheduler',
        ]
        
        missing_modules = []
        
        for module in required_modules:
            try:
                if module == 'yaml':
                    __import__('yaml')
                elif module == 'dotenv':
                    __import__('dotenv')
                else:
                    __import__(module)
                logger.info(f"✓ {module} is installed")
            except ImportError:
                logger.error(f"✗ {module} is NOT installed")
                missing_modules.append(module)
        
        if missing_modules:
            logger.error(f"\nMissing dependencies: {', '.join(missing_modules)}")
            logger.error("Please run: pip install -r requirements.txt")
            return False
        
        try:
            import pandas_ta
            logger.info("✓ pandas_ta is installed")
        except ImportError:
            logger.warning("✗ pandas_ta is NOT installed (optional but recommended)")
            logger.warning("Install with: pip install pandas-ta==0.3.14b0")
        
        return True
    
    def setup_kaggle_environment(self) -> None:
        """Setup Kaggle-specific environment settings."""
        if os.getenv('KAGGLE_MODE', 'false').lower() == 'true':
            logger.info("Kaggle mode detected - applying optimizations...")
            
            os.environ['MPLBACKEND'] = 'Agg'
            
            if os.getenv('KEEP_ALIVE', 'false').lower() == 'true':
                logger.info("Keep-alive mode enabled")
    
    def keep_alive_loop(self) -> None:
        """Keep Kaggle/Colab session alive."""
        if os.getenv('KEEP_ALIVE', 'false').lower() == 'true':
            interval = int(os.getenv('KEEP_ALIVE_INTERVAL', '300'))
            logger.info(f"Starting keep-alive loop (interval: {interval}s)")
            
            try:
                while True:
                    time.sleep(interval)
                    logger.info("Keep-alive ping...")
            except KeyboardInterrupt:
                logger.info("Keep-alive loop stopped")
    
    def launch(self) -> bool:
        """Main launch sequence."""
        logger.info("=" * 60)
        logger.info("Autonomous Trading System Launcher")
        logger.info("=" * 60)
        
        if not self.check_python_version():
            return False
        
        logger.info("\n[1/5] Creating directories...")
        self.create_directories()
        
        logger.info("\n[2/5] Checking configuration files...")
        self.check_config_files()
        
        logger.info("\n[3/5] Checking dependencies...")
        if not self.check_dependencies():
            logger.error("\nSetup incomplete. Please install missing dependencies.")
            return False
        
        logger.info("\n[4/5] Setting up environment...")
        self.setup_kaggle_environment()
        
        logger.info("\n[5/5] Launching main application...")
        
        try:
            sys.path.insert(0, str(self.base_dir))
            
            from main import TradingSystem
            
            logger.info("Initializing trading system...")
            system = TradingSystem()
            
            logger.info("Starting trading system...")
            system.start()
            
            if os.getenv('KEEP_ALIVE', 'false').lower() == 'true':
                self.keep_alive_loop()
            
            return True
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.error("Make sure all source files are present in the src/ directory")
            return False
        except Exception as e:
            logger.error(f"Launch error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Entry point."""
    launcher = SystemLauncher()
    success = launcher.launch()
    
    if not success:
        logger.error("\nLaunch failed. Please check the errors above.")
        sys.exit(1)
    
    logger.info("\nSystem launched successfully!")


if __name__ == "__main__":
    main()
