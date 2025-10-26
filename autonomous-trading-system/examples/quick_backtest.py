#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import log
from src.agents import DataAgent
from src.backtesting import BacktestEngine

def main():
    parser = argparse.ArgumentParser(description='Run quick backtest')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
                        help='Stock symbols to backtest')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-10-24', help='End date (YYYY-MM-DD)')
    parser.add_argument('--use-prompt-agent', action='store_true',
                        help='Enable PromptAgent for context-aware LLM decisions (slower but more sophisticated)')
    args = parser.parse_args()
    
    log.info("=" * 80)
    log.info("QUICK BACKTEST EXAMPLE")
    log.info("=" * 80)
    
    symbols = args.symbols
    start_date = args.start
    end_date = args.end
    
    if args.use_prompt_agent:
        log.info("⚠️  PromptAgent ENABLED - Slower but more context-aware decisions")
    else:
        log.info("⚡ Fast mode - PromptAgent disabled (use --use-prompt-agent to enable)")
    
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
    
    backtest_engine = BacktestEngine(
        initial_capital=100000.0,
        use_prompt_agent=args.use_prompt_agent
    )
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
