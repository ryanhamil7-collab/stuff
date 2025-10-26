import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from src.agents import DataAgent, AnalysisAgent, DecisionAgent
from src.strategies.portfolio import Portfolio
from src.strategies import RiskManager
from src.utils import log, config
from src.backtesting.execution_delays import (
    ExecutionDelaySimulator,
    DelayConfig,
    DelayAwareBacktester
)

class BacktestEngine:
    
    def __init__(
        self,
        initial_capital: float = None,
        enable_execution_delays: bool = None,
        delay_config: Optional[DelayConfig] = None,
        use_prompt_agent: bool = False
    ):
        if initial_capital is None:
            initial_capital = config.get('trading.initial_capital', 100000.0)
        
        if enable_execution_delays is None:
            enable_execution_delays = config.get('backtesting.execution_delays.enabled', True)
        
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.risk_manager = RiskManager(initial_capital)
        self.use_prompt_agent = use_prompt_agent
        
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent(use_prompt_agent=use_prompt_agent)
        self.decision_agent = DecisionAgent()
        
        self.enable_execution_delays = enable_execution_delays
        if self.enable_execution_delays:
            if delay_config is None:
                delay_config = DelayConfig(
                    fixed_delay_ms=config.get('backtesting.execution_delays.fixed_delay_ms', 200.0),
                    use_variable_delay=config.get('backtesting.execution_delays.use_variable_delay', True),
                    delay_mean_ms=config.get('backtesting.execution_delays.delay_mean_ms', 200.0),
                    delay_std_ms=config.get('backtesting.execution_delays.delay_std_ms', 50.0),
                    integrate_slippage=config.get('backtesting.execution_delays.integrate_slippage', True),
                    slippage_bps=config.get('backtesting.execution_delays.slippage_bps', 10.0),
                    model_partial_fills=config.get('backtesting.execution_delays.model_partial_fills', True),
                    regime_based_delays=config.get('backtesting.execution_delays.regime_based_delays', True)
                )
            self.delay_simulator = ExecutionDelaySimulator(delay_config)
            log.info(f"BacktestEngine initialized with execution delay modeling (mean={delay_config.delay_mean_ms}ms)")
        else:
            self.delay_simulator = None
            log.info(f"BacktestEngine initialized without execution delay modeling")
        
        log.info(f"Initial capital: ${initial_capital:,.2f}")
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        log.info("Starting backtest")
        
        if start_date:
            for symbol in data:
                data[symbol] = data[symbol][data[symbol]['Date'] >= start_date]
        
        if end_date:
            for symbol in data:
                data[symbol] = data[symbol][data[symbol]['Date'] <= end_date]
        
        min_length = min(len(df) for df in data.values())
        log.info(f"Backtesting over {min_length} time periods")
        
        for step in range(50, min_length):
            current_data = {}
            for symbol, df in data.items():
                current_data[symbol] = df.iloc[:step+1]
            
            analysis_result = self.analysis_agent.run(
                current_data,
                {}
            )
            
            signals = analysis_result['signals']
            
            current_prices = {}
            for symbol, df in current_data.items():
                current_prices[symbol] = df.iloc[-1]['Close']
            
            decision_result = self.decision_agent.run(
                signals,
                self.portfolio,
                current_prices
            )
            
            self.portfolio.update_positions(current_prices)
            
            if step % 20 == 0:
                metrics = self.portfolio.calculate_metrics()
                log.info(f"Step {step}/{min_length}: Equity=${metrics.get('current_equity', 0):,.2f}, Positions={metrics.get('num_open_positions', 0)}, Sharpe={metrics.get('sharpe_ratio', 0):.2f}")
        
        for symbol in list(self.portfolio.positions.keys()):
            final_price = data[symbol].iloc[-1]['Close']
            self.portfolio.close_position(symbol, final_price, reason='backtest_end')
        
        final_metrics = self.portfolio.calculate_metrics()
        
        result = {
            'metrics': final_metrics,
            'equity_curve': self.portfolio.get_equity_curve(),
            'trade_history': self.portfolio.get_trade_history(),
            'closed_positions': self.portfolio.get_closed_positions(),
            'initial_capital': self.initial_capital,
            'final_capital': final_metrics.get('current_equity', self.initial_capital),
            'total_return': final_metrics.get('total_return', 0),
            'sharpe_ratio': final_metrics.get('sharpe_ratio', 0),
            'max_drawdown': final_metrics.get('max_drawdown', 0),
            'num_trades': final_metrics.get('total_trades', 0),
            'win_rate': final_metrics.get('win_rate', 0)
        }
        
        log.info("Backtest completed")
        log.info(f"Final Equity: ${result['final_capital']:,.2f}")
        log.info(f"Total Return: {result['total_return']:.2%}")
        log.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        log.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
        log.info(f"Win Rate: {result['win_rate']:.2%}")
        
        return result
    
    def run_monte_carlo(
        self,
        data: Dict[str, pd.DataFrame],
        num_simulations: int = 100
    ) -> List[Dict]:
        log.info(f"Running {num_simulations} Monte Carlo simulations")
        
        results = []
        
        for i in range(num_simulations):
            log.info(f"Running simulation {i+1}/{num_simulations}")
            
            self.portfolio.reset()
            self.risk_manager = RiskManager(self.initial_capital)
            
            result = self.run_backtest(data)
            results.append(result)
        
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        returns = [r['total_return'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        
        summary = {
            'num_simulations': num_simulations,
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_drawdown': np.mean(drawdowns),
            'worst_drawdown': min(drawdowns),
            'best_return': max(returns),
            'worst_return': min(returns)
        }
        
        log.info("Monte Carlo simulation completed")
        log.info(f"Average Sharpe: {summary['avg_sharpe']:.2f} ± {summary['std_sharpe']:.2f}")
        log.info(f"Average Return: {summary['avg_return']:.2%} ± {summary['std_return']:.2%}")
        
        return results, summary
    
    def calculate_performance_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        equity_series = equity_curve['Equity']
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        
        num_years = len(equity_series) / 252
        annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        rolling_max = equity_series.expanding().max()
        daily_drawdown = (equity_series - rolling_max) / rolling_max
        recovery_time = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(daily_drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                recovery_time.append(i - drawdown_start)
                in_drawdown = False
        
        avg_recovery_time = np.mean(recovery_time) if recovery_time else 0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': returns.std() * np.sqrt(252),
            'avg_recovery_time': avg_recovery_time,
            'positive_periods': len(positive_returns),
            'negative_periods': len(negative_returns),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }
        
        return metrics
