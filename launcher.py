#!/usr/bin/env python3
"""
Set-and-Forget Launcher for Autonomous Trading System

Fully automated 24/7 launcher with market-aware scheduling:
- Wakes up at 9:25 AM ET Monday-Friday for trading
- Stops trading at 4:00 PM ET (market close)
- Runs offline research 6:00 PM - 6:00 AM ET
- Zero human input after initial launch

Usage:
    python launcher.py --set-and-forget --capital 500
    python launcher.py --daemon  # Run as background daemon
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional

import pytz
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/launcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingSystemLauncher:
    """Automated launcher for 24/7 trading system operation"""
    
    def __init__(self, config_path: str = "config/scheduler.yaml", capital: float = 100000.0):
        """Initialize launcher with configuration
        
        Args:
            config_path: Path to scheduler configuration file
            capital: Initial trading capital
        """
        self.config_path = config_path
        self.capital = capital
        self.config = self._load_config()
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('America/New_York'))
        self.trading_process: Optional[subprocess.Popen] = None
        self.research_process: Optional[subprocess.Popen] = None
        self.dashboard_process: Optional[subprocess.Popen] = None
        self.discovery_engine = None
        self.running = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Launcher initialized with capital: ${capital:,.2f}")
    
    def _load_config(self) -> dict:
        """Load scheduler configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration if file not found"""
        return {
            'scheduler': {
                'timezone': 'America/New_York',
                'trading': {
                    'enabled': True,
                    'wake_time': '09:25',  # 5 min before market open
                    'close_time': '16:00',  # Market close
                    'days': 'mon-fri'
                },
                'offline_research': {
                    'enabled': True,
                    'start_time': '18:00',  # 6 PM ET
                    'end_time': '06:00',    # 6 AM ET
                    'quantum_enabled': True,
                    'evolve_alphas': 2000,
                    'monte_carlo_runs': 5000,
                    'adversarial_enabled': True
                },
                'dashboard': {
                    'enabled': True,
                    'refresh_interval': 300  # 5 minutes
                }
            },
            'symbol_discovery': {
                'enabled': True,
                'mode': 'realtime',
                'frequency_minutes': 15,
                'max_symbols': 20,
                'include_crypto': True,
                'include_options': True,
                'hive_broadcast': True
            },
            'hive_mind': {
                'enabled': False,
                'peers': [],
                'port': 50051,
                'discovery_enabled': False
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start_trading(self):
        """Start autopilot trading with all 15 advanced features"""
        if self.trading_process and self.trading_process.poll() is None:
            logger.warning("Trading already running, skipping start")
            return
        
        logger.info("🚀 Starting trading autopilot with all features...")
        
        cmd = [
            sys.executable, 'main.py', 'autopilot',
            '--strategy-horizon', 'hybrid',
            '--capital', str(self.capital),
            '--paper',
            '--quantum',
            '--federated',
            '--neuro-symbolic',
            '--external-data',
            '--adversarial',
            '--ensemble',
            '--dynamic-costs',
            '--walk-forward',
            '--monte-carlo', '1000'
        ]
        
        try:
            self.trading_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            logger.info(f"Trading process started (PID: {self.trading_process.pid})")
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
    
    def stop_trading(self):
        """Gracefully stop trading at market close"""
        if not self.trading_process or self.trading_process.poll() is not None:
            logger.info("Trading not running, nothing to stop")
            return
        
        logger.info("🛑 Stopping trading at market close...")
        
        try:
            self.trading_process.terminate()
            
            try:
                self.trading_process.wait(timeout=30)
                logger.info("Trading stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Trading didn't stop gracefully, forcing kill")
                self.trading_process.kill()
                self.trading_process.wait()
            
            self.trading_process = None
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
    
    def start_offline_research(self):
        """Start offline research with quantum optimization"""
        if self.research_process and self.research_process.poll() is None:
            logger.warning("Research already running, skipping start")
            return
        
        config = self.config['scheduler']['offline_research']
        
        logger.info("🌙 Starting offline research (sleep mode)...")
        
        cmd = [
            sys.executable, 'main.py', 'offline-research',
            '--quantum',
            '--evolve-alphas', str(config['evolve_alphas']),
            '--monte-carlo', str(config['monte_carlo_runs'])
        ]
        
        if config['adversarial_enabled']:
            cmd.append('--adversarial')
        
        if self.config['hive_mind']['enabled']:
            cmd.extend(['--hive-mind', '--sync-peers'])
        
        try:
            self.research_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            logger.info(f"Research process started (PID: {self.research_process.pid})")
        except Exception as e:
            logger.error(f"Failed to start research: {e}")
    
    def stop_offline_research(self):
        """Stop offline research in the morning"""
        if not self.research_process or self.research_process.poll() is not None:
            logger.info("Research not running, nothing to stop")
            return
        
        logger.info("☀️ Stopping offline research (wake up)...")
        
        try:
            self.research_process.terminate()
            
            try:
                self.research_process.wait(timeout=60)
                logger.info("Research stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Research didn't stop gracefully, forcing kill")
                self.research_process.kill()
                self.research_process.wait()
            
            self.research_process = None
        except Exception as e:
            logger.error(f"Error stopping research: {e}")
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("Dashboard already running")
            return
        
        if not self.config['scheduler']['dashboard']['enabled']:
            return
        
        logger.info("📊 Starting dashboard...")
        
        cmd = [sys.executable, 'main.py', 'dashboard']
        
        try:
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Dashboard started (PID: {self.dashboard_process.pid})")
            logger.info("Dashboard available at http://localhost:8501")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
    
    def refresh_dashboard(self):
        """Refresh dashboard (restart if needed)"""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            return  # Already running
        
        logger.info("Refreshing dashboard...")
        self.start_dashboard()
    
    def run_symbol_discovery(self):
        """Run real-time symbol discovery"""
        if not self.config.get('symbol_discovery', {}).get('enabled', False):
            return
        
        if not self.check_market_hours():
            logger.debug("Outside market hours, skipping symbol discovery")
            return
        
        try:
            if not self.discovery_engine:
                from src.agents.symbol_discovery import SymbolDiscoveryV2
                
                hive_mind = None
                if self.config.get('hive_mind', {}).get('enabled', False):
                    from src.hive_mind import P2PNode
                    hive_mind = P2PNode(self.config['hive_mind'])
                
                self.discovery_engine = SymbolDiscoveryV2(hive_mind=hive_mind)
                logger.info("Symbol discovery engine initialized")
            
            logger.info("🔍 Running real-time symbol discovery...")
            discoveries = self.discovery_engine.discover_symbols_realtime()
            
            if discoveries:
                logger.info(f"Discovered {len(discoveries)} high-potential symbols:")
                for i, d in enumerate(discoveries[:5], 1):
                    logger.info(f"  {i}. {d['symbol']} (score: {d['score']:.2f}, class: {d['asset_class']})")
            else:
                logger.info("No high-potential symbols discovered this cycle")
                
        except Exception as e:
            logger.error(f"Symbol discovery failed: {e}")
    
    def check_market_hours(self) -> bool:
        """Check if currently in market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""
        now = datetime.now(pytz.timezone('America/New_York'))
        
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def check_research_hours(self) -> bool:
        """Check if currently in research hours (6:00 PM - 6:00 AM ET)"""
        now = datetime.now(pytz.timezone('America/New_York'))
        current_time = now.time()
        
        research_start = dt_time(18, 0)  # 6 PM
        research_end = dt_time(6, 0)     # 6 AM
        
        if current_time >= research_start or current_time <= research_end:
            return True
        return False
    
    def setup_schedule(self):
        """Setup automated schedule for trading and research"""
        config = self.config['scheduler']
        
        if config['trading']['enabled']:
            wake_time = config['trading']['wake_time']
            self.scheduler.add_job(
                self.start_trading,
                CronTrigger(
                    day_of_week='mon-fri',
                    hour=int(wake_time.split(':')[0]),
                    minute=int(wake_time.split(':')[1]),
                    timezone='America/New_York'
                ),
                id='start_trading',
                name='Start Trading',
                replace_existing=True
            )
            logger.info(f"Scheduled trading start: {wake_time} ET (Mon-Fri)")
            
            close_time = config['trading']['close_time']
            self.scheduler.add_job(
                self.stop_trading,
                CronTrigger(
                    day_of_week='mon-fri',
                    hour=int(close_time.split(':')[0]),
                    minute=int(close_time.split(':')[1]),
                    timezone='America/New_York'
                ),
                id='stop_trading',
                name='Stop Trading',
                replace_existing=True
            )
            logger.info(f"Scheduled trading stop: {close_time} ET (Mon-Fri)")
        
        if config['offline_research']['enabled']:
            start_time = config['offline_research']['start_time']
            self.scheduler.add_job(
                self.start_offline_research,
                CronTrigger(
                    hour=int(start_time.split(':')[0]),
                    minute=int(start_time.split(':')[1]),
                    timezone='America/New_York'
                ),
                id='start_research',
                name='Start Offline Research',
                replace_existing=True
            )
            logger.info(f"Scheduled research start: {start_time} ET (daily)")
            
            end_time = config['offline_research']['end_time']
            self.scheduler.add_job(
                self.stop_offline_research,
                CronTrigger(
                    hour=int(end_time.split(':')[0]),
                    minute=int(end_time.split(':')[1]),
                    timezone='America/New_York'
                ),
                id='stop_research',
                name='Stop Offline Research',
                replace_existing=True
            )
            logger.info(f"Scheduled research stop: {end_time} ET (daily)")
        
        if config['dashboard']['enabled']:
            refresh_interval = config['dashboard']['refresh_interval']
            self.scheduler.add_job(
                self.refresh_dashboard,
                'interval',
                seconds=refresh_interval,
                id='refresh_dashboard',
                name='Refresh Dashboard',
                replace_existing=True
            )
            logger.info(f"Scheduled dashboard refresh: every {refresh_interval}s")
        
        if self.config.get('symbol_discovery', {}).get('enabled', False):
            discovery_config = self.config['symbol_discovery']
            frequency_minutes = discovery_config.get('frequency_minutes', 15)
            self.scheduler.add_job(
                self.run_symbol_discovery,
                'interval',
                minutes=frequency_minutes,
                id='symbol_discovery',
                name='Symbol Discovery',
                replace_existing=True
            )
            logger.info(f"Scheduled symbol discovery: every {frequency_minutes} minutes")
    
    def start(self):
        """Start the automated launcher"""
        logger.info("=" * 80)
        logger.info("🤖 AUTONOMOUS TRADING SYSTEM - SET AND FORGET LAUNCHER")
        logger.info("=" * 80)
        logger.info(f"Capital: ${self.capital:,.2f}")
        logger.info(f"Timezone: America/New_York (ET)")
        logger.info(f"Hive Mind: {'Enabled' if self.config['hive_mind']['enabled'] else 'Disabled'}")
        logger.info("=" * 80)
        
        Path('logs').mkdir(exist_ok=True)
        
        self.setup_schedule()
        
        self.start_dashboard()
        
        if self.check_market_hours():
            logger.info("Currently in market hours, starting trading...")
            self.start_trading()
        elif self.check_research_hours():
            logger.info("Currently in research hours, starting research...")
            self.start_offline_research()
        else:
            logger.info("Outside trading/research hours, waiting for next scheduled event...")
        
        self.scheduler.start()
        self.running = True
        
        logger.info("✅ Launcher started successfully!")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()
    
    def stop(self):
        """Stop all processes and scheduler"""
        logger.info("Stopping launcher...")
        
        self.running = False
        
        self.stop_trading()
        self.stop_offline_research()
        
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=10)
            except:
                pass
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        
        logger.info("✅ Launcher stopped")
    
    def status(self):
        """Print current status"""
        now = datetime.now(pytz.timezone('America/New_York'))
        
        print("\n" + "=" * 80)
        print("📊 LAUNCHER STATUS")
        print("=" * 80)
        print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Capital: ${self.capital:,.2f}")
        print()
        
        trading_running = self.trading_process and self.trading_process.poll() is None
        print(f"Trading: {'🟢 RUNNING' if trading_running else '🔴 STOPPED'}")
        if trading_running:
            print(f"  PID: {self.trading_process.pid}")
        
        research_running = self.research_process and self.research_process.poll() is None
        print(f"Research: {'🟢 RUNNING' if research_running else '🔴 STOPPED'}")
        if research_running:
            print(f"  PID: {self.research_process.pid}")
        
        dashboard_running = self.dashboard_process and self.dashboard_process.poll() is None
        print(f"Dashboard: {'🟢 RUNNING' if dashboard_running else '🔴 STOPPED'}")
        if dashboard_running:
            print(f"  PID: {self.dashboard_process.pid}")
            print(f"  URL: http://localhost:8501")
        
        print()
        
        print("Scheduled Jobs:")
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            if next_run:
                print(f"  {job.name}: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        print("=" * 80)
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Set-and-Forget Launcher for Autonomous Trading System'
    )
    parser.add_argument(
        '--set-and-forget',
        action='store_true',
        help='Run in fully automated mode (24/7)'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as background daemon'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial trading capital (default: 100000)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/scheduler.yaml',
        help='Path to scheduler config file'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status and exit'
    )
    
    args = parser.parse_args()
    
    launcher = TradingSystemLauncher(
        config_path=args.config,
        capital=args.capital
    )
    
    if args.status:
        launcher.status()
        return
    
    if args.daemon:
        logger.info("Running in daemon mode...")
    
    launcher.start()


if __name__ == '__main__':
    main()
