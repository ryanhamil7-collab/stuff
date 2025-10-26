import time
import os
from datetime import datetime, time as dt_time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from typing import Dict, Optional
from dotenv import load_dotenv
from src.agents import DataAgent, AnalysisAgent, DecisionAgent
from src.agents.offline_research import OfflineResearchEngine
from src.strategies.portfolio import Portfolio
from src.strategies import RiskManager
from src.utils import log, config

load_dotenv()

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    log.warning("alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")

class AutopilotDaemon:
    
    def __init__(self, use_alpaca: bool = True, enable_research: bool = True):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.decision_agent = DecisionAgent()
        
        self.enable_research = enable_research
        if self.enable_research:
            self.research_engine = OfflineResearchEngine()
            log.info("Offline research engine enabled")
        else:
            self.research_engine = None
        
        self.use_alpaca = use_alpaca and ALPACA_AVAILABLE
        self.alpaca_api: Optional[tradeapi.REST] = None
        
        if self.use_alpaca:
            self._init_alpaca()
        
        if self.alpaca_api:
            account = self.alpaca_api.get_account()
            initial_capital = float(account.portfolio_value)
            log.info(f"Using Alpaca account capital: ${initial_capital:,.2f}")
        else:
            initial_capital = config.get('trading.initial_capital', 100000.0)
            log.info(f"Using simulated capital: ${initial_capital:,.2f}")
        
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(initial_capital)
        
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        
        self.trading_hours_start = config.get('trading.trading_hours.start', '09:30')
        self.trading_hours_end = config.get('trading.trading_hours.end', '16:00')
        self.timezone = pytz.timezone(config.get('trading.trading_hours.timezone', 'America/New_York'))
        
        log.info("AutopilotDaemon initialized")
    
    def _init_alpaca(self):
        """Initialize Alpaca API connection"""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            log.warning("Alpaca API keys not found in environment")
            log.warning("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
            self.use_alpaca = False
            return
        
        try:
            log.info("Connecting to Alpaca Paper Trading API...")
            self.alpaca_api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            account = self.alpaca_api.get_account()
            log.info(f"✓ Connected to Alpaca Paper Trading")
            log.info(f"  Account Status: {account.status}")
            log.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
            log.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
            log.info(f"  Cash: ${float(account.cash):,.2f}")
        except Exception as e:
            log.error(f"Failed to connect to Alpaca: {str(e)}")
            self.use_alpaca = False
            self.alpaca_api = None
    
    def _place_alpaca_order(self, symbol: str, qty: int, side: str) -> bool:
        """Place an order on Alpaca"""
        if not self.alpaca_api:
            return False
        
        try:
            log.info(f"Placing {side.upper()} order on Alpaca: {qty} shares of {symbol}")
            
            order = self.alpaca_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            log.info(f"✓ Alpaca order placed: {order.id}")
            log.info(f"  Symbol: {order.symbol}")
            log.info(f"  Qty: {order.qty}")
            log.info(f"  Side: {order.side}")
            log.info(f"  Status: {order.status}")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to place Alpaca order: {str(e)}")
            return False
    
    def _sync_alpaca_positions(self):
        """Sync positions from Alpaca"""
        if not self.alpaca_api:
            return
        
        try:
            positions = self.alpaca_api.list_positions()
            log.info(f"Alpaca positions: {len(positions)}")
            for pos in positions:
                log.info(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} "
                        f"(P&L: ${float(pos.unrealized_pl):,.2f} / {float(pos.unrealized_plpc):.2%})")
        except Exception as e:
            log.error(f"Failed to sync Alpaca positions: {str(e)}")
    
    def is_trading_hours(self) -> bool:
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        start_time = dt_time.fromisoformat(self.trading_hours_start)
        end_time = dt_time.fromisoformat(self.trading_hours_end)
        
        if now.weekday() >= 5:
            return False
        
        return start_time <= current_time <= end_time
    
    def trading_cycle(self):
        try:
            log.info("=" * 80)
            log.info(f"Starting cycle at {datetime.now()}")
            
            if not self.is_trading_hours():
                log.info("Outside trading hours")
                if self.enable_research and self.research_engine:
                    log.info("Running offline research and training...")
                    try:
                        self.research_engine.run_offline_research()
                        log.info("✓ Offline research completed")
                    except Exception as e:
                        log.error(f"Error in offline research: {str(e)}")
                else:
                    log.info("Offline research disabled, skipping cycle")
                return
            
            if self.alpaca_api:
                log.info("Step 0: Syncing Alpaca account")
                self._sync_alpaca_positions()
            
            log.info("Step 1: Collecting market data")
            data_result = self.data_agent.run()
            
            if not data_result or not data_result.get('processed_data'):
                log.warning("No data collected, skipping cycle")
                return
            
            processed_data = data_result['processed_data']
            news_data = data_result.get('news', {})
            
            log.info(f"Step 2: Analyzing {len(processed_data)} symbols")
            analysis_result = self.analysis_agent.run(processed_data, news_data)
            
            signals = analysis_result.get('signals', {})
            
            log.info("Step 3: Making trading decisions")
            current_prices = {}
            for symbol, df in processed_data.items():
                if len(df) > 0:
                    current_prices[symbol] = df.iloc[-1]['Close']
            
            decision_result = self.decision_agent.run(
                signals,
                self.portfolio,
                current_prices
            )
            
            log.info("Step 4: Executing trades")
            decisions = decision_result.get('decisions', [])
            executed_count = 0
            
            for decision in decisions:
                symbol = decision['symbol']
                action = decision['action']
                shares = decision.get('shares', 0)
                
                if shares > 0:
                    if self.alpaca_api:
                        side = 'buy' if action == 'BUY' else 'sell'
                        if self._place_alpaca_order(symbol, shares, side):
                            executed_count += 1
                            time.sleep(1)
                    else:
                        log.info(f"Simulated {action}: {shares} shares of {symbol}")
                        executed_count += 1
            
            log.info(f"Executed {executed_count} trades this cycle")
            
            log.info("Step 5: Updating portfolio")
            self.portfolio.update_positions(current_prices)
            self.risk_manager.update_capital(self.portfolio.get_total_value(current_prices))
            
            if self.alpaca_api:
                account = self.alpaca_api.get_account()
                log.info(f"Alpaca Account Status:")
                log.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
                log.info(f"  Cash: ${float(account.cash):,.2f}")
                log.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
            
            metrics = self.portfolio.calculate_metrics()
            log.info(f"Local Portfolio Status:")
            log.info(f"  Equity: ${metrics.get('current_equity', 0):,.2f}")
            log.info(f"  Cash: ${metrics.get('cash', 0):,.2f}")
            log.info(f"  Open Positions: {metrics.get('num_open_positions', 0)}")
            log.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            log.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            log.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            log.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            log.info("Trading cycle completed successfully")
            log.info("=" * 80)
            
        except Exception as e:
            log.error(f"Error in trading cycle: {str(e)}", exc_info=True)
    
    def start(self):
        if self.is_running:
            log.warning("Autopilot is already running")
            return
        
        log.info("Starting Autopilot Daemon")
        
        check_interval = config.get('autopilot.check_interval', 300)
        
        self.scheduler.add_job(
            self.trading_cycle,
            'interval',
            seconds=check_interval,
            id='trading_cycle',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        
        log.info(f"Autopilot started - checking every {check_interval} seconds")
        log.info(f"Trading hours: {self.trading_hours_start} - {self.trading_hours_end} {self.timezone}")
        
        self.trading_cycle()
    
    def stop(self):
        if not self.is_running:
            log.warning("Autopilot is not running")
            return
        
        log.info("Stopping Autopilot Daemon")
        
        self.scheduler.shutdown()
        self.is_running = False
        
        log.info("Autopilot stopped")
    
    def run_continuous(self):
        self.start()
        
        try:
            while True:
                time.sleep(60)
                
                if config.get('safety.circuit_breaker', True):
                    equity_series = self.portfolio.get_equity_curve()['Equity']
                    if self.risk_manager.check_circuit_breaker(equity_series):
                        log.error("Circuit breaker triggered! Stopping autopilot")
                        self.stop()
                        break
                
        except KeyboardInterrupt:
            log.info("Received interrupt signal")
            self.stop()
        except Exception as e:
            log.error(f"Error in continuous run: {str(e)}")
            self.stop()
    
    def get_status(self) -> Dict:
        metrics = self.portfolio.calculate_metrics()
        
        status = {
            'is_running': self.is_running,
            'is_trading_hours': self.is_trading_hours(),
            'portfolio_metrics': metrics,
            'open_positions': self.portfolio.get_positions_list(),
            'timestamp': datetime.now()
        }
        
        return status

if __name__ == "__main__":
    log.info("Starting Autonomous Trading System in Autopilot Mode")
    
    autopilot = AutopilotDaemon()
    
    mode = config.get('autopilot.mode', 'paper')
    if mode != 'paper':
        log.error("SAFETY CHECK: Only paper trading mode is allowed!")
        log.error("Please set autopilot.mode to 'paper' in config.yaml")
        exit(1)
    
    log.info("PAPER TRADING MODE - No real money at risk")
    log.info("=" * 80)
    
    autopilot.run_continuous()
