#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import log
from src.agents import DataAgent
from src.backtesting import BacktestEngine

def main():
    log.info("=" * 80)
    log.info("QUICK BACKTEST EXAMPLE")
    log.info("=" * 80)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    start_date = '2023-01-01'
    end_date = '2024-10-24'
    
    log.info(f"Fetching data for: {', '.join(symbols)}")
    log.info(f"Date range: {start_date} to {end_date}")
    
    data_agent = DataAgent()
    data = data_agent.collect_market_data(symbols, start_date, end_date)
    
    log.info("Processing data with technical indicators...")
    processed_data = data_agent.process_data(data)
    
    if not processed_data:
        log.error("No data available")
        return
    
    log.info(f"Running backtest on {len(processed_data)} symbols...")
    
    backtest_engine = BacktestEngine(initial_capital=100000.0)
    result = backtest_engine.run_backtest(
        processed_data,
        start_date=start_date,
        end_date=end_date
    )
    
    log.info("=" * 80)
    log.info("RESULTS")
    log.info("=" * 80)
    log.info(f"Initial Capital:  ${result['initial_capital']:>12,.2f}")
    log.info(f"Final Capital:    ${result['final_capital']:>12,.2f}")
    log.info(f"Total Return:     {result['total_return']:>12.2%}")
    log.info(f"Sharpe Ratio:     {result['sharpe_ratio']:>12.2f}")
    log.info(f"Max Drawdown:     {result['max_drawdown']:>12.2%}")
    log.info(f"Win Rate:         {result['win_rate']:>12.2%}")
    log.info(f"Total Trades:     {result['num_trades']:>12}")
    log.info("=" * 80)
    
    equity_curve = result['equity_curve']
    output_file = Path(__file__).parent.parent / 'data' / 'example_equity_curve.csv'
    equity_curve.to_csv(output_file, index=False)
    log.info(f"Equity curve saved to: {output_file}")
    
    trade_history = result['trade_history']
    if not trade_history.empty:
        output_file = Path(__file__).parent.parent / 'data' / 'example_trades.csv'
        trade_history.to_csv(output_file, index=False)
        log.info(f"Trade history saved to: {output_file}")
    
    log.info("\n✅ Backtest completed successfully!")

if __name__ == "__main__":
    main()
