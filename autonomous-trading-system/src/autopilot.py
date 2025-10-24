import time
from datetime import datetime, time as dt_time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from typing import Dict
from src.agents import DataAgent, AnalysisAgent, DecisionAgent
from src.strategies.portfolio import Portfolio
from src.strategies import RiskManager
from src.utils import log, config

class AutopilotDaemon:
    
    def __init__(self):
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent()
        self.decision_agent = DecisionAgent()
        
        initial_capital = config.get('trading.initial_capital', 100000.0)
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(initial_capital)
        
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        
        self.trading_hours_start = config.get('trading.trading_hours.start', '09:30')
        self.trading_hours_end = config.get('trading.trading_hours.end', '16:00')
        self.timezone = pytz.timezone(config.get('trading.trading_hours.timezone', 'America/New_York'))
        
        log.info("AutopilotDaemon initialized")
    
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
            log.info(f"Starting trading cycle at {datetime.now()}")
            
            if not self.is_trading_hours():
                log.info("Outside trading hours, skipping cycle")
                return
            
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
            
            log.info("Step 4: Updating portfolio")
            self.portfolio.update_positions(current_prices)
            self.risk_manager.update_capital(self.portfolio.get_total_value(current_prices))
            
            metrics = self.portfolio.calculate_metrics()
            log.info(f"Portfolio Status:")
            log.info(f"  Equity: ${metrics.get('current_equity', 0):,.2f}")
            log.info(f"  Cash: ${metrics.get('cash', 0):,.2f}")
            log.info(f"  Open Positions: {metrics.get('num_open_positions', 0)}")
            log.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            log.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            log.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            log.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            executed_trades = decision_result.get('execution', {}).get('num_executed', 0)
            log.info(f"Executed {executed_trades} trades this cycle")
            
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
