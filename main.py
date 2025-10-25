#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.utils import log, config
from src.utils.validation import validate_no_lookahead
from src.backtesting import BacktestEngine
from src.agents import DataAgent
from src.autopilot import AutopilotDaemon
from src.strategies.benchmarks import BenchmarkStrategies

def run_backtest(symbols=None, start_date=None, end_date=None, no_lookahead_check=False, run_benchmarks=True):
    log.info("Starting backtest mode")
    
    if symbols is None:
        symbols = config.get('symbols.custom_symbols', ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'])
    
    if start_date is None:
        start_date = config.get('backtest.start_date', '2020-01-01')
    
    if end_date is None:
        end_date = config.get('backtest.end_date', '2025-01-01')
    
    log.info(f"Backtesting symbols: {symbols}")
    log.info(f"Date range: {start_date} to {end_date}")
    
    data_agent = DataAgent()
    log.info("Collecting market data...")
    data = data_agent.collect_market_data(symbols)
    
    log.info("Processing data...")
    processed_data = data_agent.process_data(data)
    
    if not processed_data:
        log.error("No data available for backtesting")
        return
    
    if not no_lookahead_check:
        log.info("Validating data for lookahead bias...")
        if not validate_no_lookahead(processed_data):
            log.error("CRITICAL: Lookahead bias detected! Fix before proceeding.")
            log.error("Use --no-lookahead-check to skip this validation (not recommended)")
            return
    
    if run_benchmarks:
        log.info("Running benchmark strategies for comparison...")
        benchmarks = BenchmarkStrategies.run_all_benchmarks(processed_data, config.get('trading.initial_capital', 100000.0))
    
    log.info(f"Running AI trading system backtest on {len(processed_data)} symbols...")
    backtest_engine = BacktestEngine()
    result = backtest_engine.run_backtest(processed_data, start_date, end_date)
    
    log.info("=" * 80)
    log.info("BACKTEST RESULTS - AI TRADING SYSTEM")
    log.info("=" * 80)
    log.info(f"Initial Capital: ${result['initial_capital']:,.2f}")
    log.info(f"Final Capital: ${result['final_capital']:,.2f}")
    log.info(f"Total Return: {result['total_return']:.2%}")
    log.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    log.info(f"Sortino Ratio: {result['metrics'].get('sortino_ratio', 0):.2f}")
    log.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
    log.info(f"Win Rate: {result['win_rate']:.2%}")
    log.info(f"Total Trades: {result['num_trades']}")
    log.info(f"Profit Factor: {result['metrics'].get('profit_factor', 0):.2f}")
    log.info("=" * 80)
    
    if run_benchmarks and benchmarks:
        log.info("")
        log.info("=" * 80)
        log.info("BENCHMARK COMPARISON")
        log.info("=" * 80)
        log.info(f"{'Strategy':<40} {'Return':<12} {'Sharpe':<10}")
        log.info("-" * 80)
        log.info(f"{'AI Trading System':<40} {result['total_return']:>10.2%}  {result['sharpe_ratio']:>8.2f}")
        for benchmark in benchmarks:
            log.info(f"{benchmark['strategy']:<40} {benchmark['total_return']:>10.2%}  {benchmark['sharpe_ratio']:>8.2f}")
        log.info("=" * 80)
        
        ai_sharpe = result['sharpe_ratio']
        best_benchmark_sharpe = max(b['sharpe_ratio'] for b in benchmarks)
        
        if ai_sharpe > best_benchmark_sharpe:
            log.info(f"✓ AI system outperforms all benchmarks (Sharpe: {ai_sharpe:.2f} vs {best_benchmark_sharpe:.2f})")
        else:
            log.warning(f"⚠ AI system underperforms best benchmark (Sharpe: {ai_sharpe:.2f} vs {best_benchmark_sharpe:.2f})")
            log.warning("Consider tuning parameters or checking for overfitting")
        log.info("=" * 80)
    
    equity_curve = result['equity_curve']
    equity_curve.to_csv('data/backtest_equity_curve.csv', index=False)
    log.info("Equity curve saved to data/backtest_equity_curve.csv")
    
    trade_history = result['trade_history']
    if not trade_history.empty:
        trade_history.to_csv('data/backtest_trade_history.csv', index=False)
        log.info("Trade history saved to data/backtest_trade_history.csv")

def run_autopilot():
    log.info("Starting autopilot mode")
    
    mode = config.get('autopilot.mode', 'paper')
    if mode != 'paper':
        log.error("SAFETY CHECK FAILED: Only paper trading mode is allowed!")
        log.error("Please set autopilot.mode to 'paper' in config/config.yaml")
        sys.exit(1)
    
    log.info("=" * 80)
    log.info("AUTONOMOUS TRADING SYSTEM - AUTOPILOT MODE")
    log.info("=" * 80)
    log.info("⚠️  PAPER TRADING ONLY - No real money at risk")
    log.info("=" * 80)
    
    autopilot = AutopilotDaemon()
    autopilot.run_continuous()

def run_dashboard():
    log.info("Starting dashboard")
    
    import subprocess
    import os
    
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    
    port = config.get('dashboard.port', 8501)
    host = config.get('dashboard.host', '0.0.0.0')
    
    log.info(f"Launching Streamlit dashboard at http://{host}:{port}")
    
    subprocess.run([
        "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host
    ])

def run_single_cycle():
    log.info("Running single trading cycle")
    
    data_agent = DataAgent()
    log.info("Step 1: Collecting and processing market data...")
    data_result = data_agent.run()
    
    if not data_result or not data_result.get('processed_data'):
        log.error("No data collected")
        return
    
    from src.agents import AnalysisAgent, DecisionAgent
    from src.strategies.portfolio import Portfolio
    
    analysis_agent = AnalysisAgent()
    decision_agent = DecisionAgent()
    portfolio = Portfolio()
    
    log.info("Step 2: Analyzing market data...")
    analysis_result = analysis_agent.run(
        data_result['processed_data'],
        data_result.get('news', {})
    )
    
    log.info("Step 3: Making trading decisions...")
    current_prices = {}
    for symbol, df in data_result['processed_data'].items():
        if len(df) > 0:
            current_prices[symbol] = df.iloc[-1]['Close']
    
    decision_result = decision_agent.run(
        analysis_result['signals'],
        portfolio,
        current_prices
    )
    
    log.info("=" * 80)
    log.info("TRADING CYCLE RESULTS")
    log.info("=" * 80)
    log.info(f"Symbols Analyzed: {len(data_result['processed_data'])}")
    log.info(f"Signals Generated: {len(analysis_result['signals'])}")
    log.info(f"Decisions Made: {len(decision_result['decisions'])}")
    log.info(f"Trades Executed: {decision_result['execution']['num_executed']}")
    log.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='Autonomous Trading System - LLM-Powered Algorithmic Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py backtest                    # Run backtest with default settings
  python main.py backtest --symbols AAPL MSFT GOOGL
  python main.py autopilot                   # Run in continuous autopilot mode
  python main.py dashboard                   # Launch web dashboard
  python main.py single                      # Run single trading cycle
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['backtest', 'autopilot', 'dashboard', 'single'],
        help='Operating mode'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='List of symbols to trade (for backtest mode)'
    )
    
    parser.add_argument(
        '--start-date',
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--no-lookahead-check',
        action='store_true',
        help='Skip lookahead bias validation (not recommended)'
    )
    
    parser.add_argument(
        '--no-benchmarks',
        action='store_true',
        help='Skip benchmark strategy comparison'
    )
    
    args = parser.parse_args()
    
    log.info("=" * 80)
    log.info("AUTONOMOUS TRADING SYSTEM")
    log.info("=" * 80)
    log.info(f"Mode: {args.mode}")
    log.info("=" * 80)
    
    try:
        if args.mode == 'backtest':
            run_backtest(
                args.symbols, 
                args.start_date, 
                args.end_date,
                args.no_lookahead_check,
                not args.no_benchmarks
            )
        
        elif args.mode == 'autopilot':
            run_autopilot()
        
        elif args.mode == 'dashboard':
            run_dashboard()
        
        elif args.mode == 'single':
            run_single_cycle()
    
    except KeyboardInterrupt:
        log.info("\nReceived interrupt signal, shutting down...")
    
    except Exception as e:
        log.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
