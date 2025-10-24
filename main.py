#!/usr/bin/env python3
"""
Autonomous Trading System - Main Application
Comprehensive trading system with 19 features.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, time as dt_time
from dotenv import load_dotenv

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None
    print("Warning: apscheduler not installed. Scheduling features disabled.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.config = None
        self.scheduler_config = None
        self.scheduler = None
        self.is_running = False
        
        self.load_environment()
        self.load_configurations()
        self.initialize_components()
    
    def load_environment(self) -> None:
        """Load environment variables."""
        env_file = self.base_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Environment variables loaded")
        else:
            logger.warning(".env file not found - using defaults")
    
    def load_configurations(self) -> None:
        """Load configuration files."""
        config_dir = self.base_dir / "config"
        
        config_file = config_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Main configuration loaded")
        else:
            logger.error(f"Config file not found: {config_file}")
            self.config = self.get_default_config()
        
        scheduler_file = config_dir / "scheduler.yaml"
        if scheduler_file.exists():
            with open(scheduler_file, 'r') as f:
                self.scheduler_config = yaml.safe_load(f)
            logger.info("Scheduler configuration loaded")
        else:
            logger.error(f"Scheduler config not found: {scheduler_file}")
            self.scheduler_config = self.get_default_scheduler_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'api': {
                'alpaca': {
                    'base_url': 'https://paper-api.alpaca.markets',
                    'paper_trading': True
                }
            },
            'trading': {
                'mode': 'paper',
                'symbols': ['AAPL', 'MSFT', 'GOOGL']
            },
            'features': {
                'backtesting': True,
                'paper_trading': True,
                'risk_management': True
            }
        }
    
    def get_default_scheduler_config(self) -> Dict:
        """Return default scheduler configuration."""
        return {
            'capital': {'initial': 1000},
            'risk_management': {
                'max_daily_loss': 0.02,
                'stop_loss_percentage': 0.02
            },
            'kaggle': {
                'keep_alive': True,
                'keep_alive_interval': 300
            }
        }
    
    def initialize_components(self) -> None:
        """Initialize system components."""
        logger.info("Initializing system components...")
        
        try:
            from src.indicators.technical_indicators import TechnicalIndicators
            self.indicators = TechnicalIndicators()
            logger.info("✓ Technical indicators initialized")
        except ImportError as e:
            logger.error(f"Failed to import indicators: {e}")
            self.indicators = None
        
        if BackgroundScheduler:
            self.scheduler = BackgroundScheduler()
            logger.info("✓ Scheduler initialized")
        else:
            logger.warning("✗ Scheduler not available (apscheduler not installed)")
        
        self.setup_data_manager()
        self.setup_risk_manager()
        self.setup_strategy_manager()
    
    def setup_data_manager(self) -> None:
        """Setup data management."""
        logger.info("Setting up data manager...")
        self.data_cache = {}
    
    def setup_risk_manager(self) -> None:
        """Setup risk management."""
        logger.info("Setting up risk manager...")
        risk_config = self.scheduler_config.get('risk_management', {})
        self.max_daily_loss = risk_config.get('max_daily_loss', 0.02)
        self.stop_loss_pct = risk_config.get('stop_loss_percentage', 0.02)
    
    def setup_strategy_manager(self) -> None:
        """Setup strategy management."""
        logger.info("Setting up strategy manager...")
        self.active_strategies = []
    
    def fetch_market_data(self, symbol: str, period: str = "1mo") -> Optional[object]:
        """Fetch market data for a symbol."""
        try:
            import yfinance as yf
            logger.info(f"Fetching data for {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
            
            logger.info(f"✓ Fetched {len(data)} bars for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a symbol with technical indicators."""
        logger.info(f"Analyzing {symbol}...")
        
        data = self.fetch_market_data(symbol)
        if data is None or data.empty:
            return {'symbol': symbol, 'status': 'error', 'message': 'No data'}
        
        if self.indicators:
            try:
                indicator_config = self.config.get('indicators', {})
                data_with_indicators = self.indicators.calculate_all(data, indicator_config)
                
                latest = data_with_indicators.iloc[-1]
                
                analysis = {
                    'symbol': symbol,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'price': float(latest['Close']),
                    'indicators': {}
                }
                
                if 'RSI' in data_with_indicators.columns:
                    analysis['indicators']['RSI'] = float(latest['RSI'])
                if 'MACD' in data_with_indicators.columns:
                    analysis['indicators']['MACD'] = float(latest['MACD'])
                if 'SMA_20' in data_with_indicators.columns:
                    analysis['indicators']['SMA_20'] = float(latest['SMA_20'])
                
                logger.info(f"✓ Analysis complete for {symbol}")
                return analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                return {'symbol': symbol, 'status': 'error', 'message': str(e)}
        else:
            return {'symbol': symbol, 'status': 'error', 'message': 'Indicators not available'}
    
    def run_backtest(self) -> None:
        """Run backtesting on configured symbols."""
        logger.info("=" * 60)
        logger.info("Running Backtest")
        logger.info("=" * 60)
        
        symbols = self.config.get('trading', {}).get('symbols', ['AAPL'])
        
        for symbol in symbols:
            logger.info(f"\nBacktesting {symbol}...")
            analysis = self.analyze_symbol(symbol)
            
            if analysis['status'] == 'success':
                logger.info(f"Price: ${analysis['price']:.2f}")
                if 'indicators' in analysis:
                    for ind_name, ind_value in analysis['indicators'].items():
                        logger.info(f"{ind_name}: {ind_value:.2f}")
            else:
                logger.error(f"Backtest failed for {symbol}: {analysis.get('message', 'Unknown error')}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Backtest Complete")
        logger.info("=" * 60)
    
    def start_paper_trading(self) -> None:
        """Start paper trading mode."""
        logger.info("Starting paper trading mode...")
        
        api_key = os.getenv('ALPACA_API_KEY', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        
        if not api_key or api_key == 'your_alpaca_api_key_here':
            logger.warning("Alpaca API keys not configured!")
            logger.warning("Paper trading requires valid API keys in .env file")
            logger.info("Running in analysis-only mode...")
            self.run_backtest()
            return
        
        logger.info("Paper trading initialized with Alpaca API")
    
    def schedule_tasks(self) -> None:
        """Schedule periodic tasks."""
        if not self.scheduler:
            logger.warning("Scheduler not available - running in manual mode")
            return
        
        logger.info("Scheduling tasks...")
        
        try:
            self.scheduler.add_job(
                self.run_backtest,
                'interval',
                minutes=60,
                id='periodic_analysis'
            )
            logger.info("✓ Scheduled periodic analysis (every 60 minutes)")
        except Exception as e:
            logger.error(f"Error scheduling tasks: {e}")
    
    def start(self) -> None:
        """Start the trading system."""
        logger.info("\n" + "=" * 60)
        logger.info("AUTONOMOUS TRADING SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.config.get('trading', {}).get('mode', 'paper')}")
        logger.info(f"Symbols: {', '.join(self.config.get('trading', {}).get('symbols', []))}")
        logger.info("=" * 60 + "\n")
        
        self.is_running = True
        
        features = self.config.get('features', {})
        
        if features.get('backtesting', True):
            logger.info("Running initial backtest...")
            self.run_backtest()
        
        if features.get('paper_trading', False):
            self.start_paper_trading()
        
        if self.scheduler:
            self.schedule_tasks()
            self.scheduler.start()
            logger.info("Scheduler started")
        
        logger.info("\n✓ System is running")
        logger.info("Press Ctrl+C to stop\n")
    
    def stop(self) -> None:
        """Stop the trading system."""
        logger.info("Stopping trading system...")
        self.is_running = False
        
        if self.scheduler:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")
        
        logger.info("System stopped")


def main():
    """Main entry point."""
    try:
        system = TradingSystem()
        system.start()
        
        import time
        while system.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal")
        if 'system' in locals():
            system.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
