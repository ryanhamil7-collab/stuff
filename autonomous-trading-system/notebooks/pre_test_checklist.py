#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from src.utils import log, config
from src.agents import DataAgent
from src.utils.validation import DataValidator
from src.strategies.benchmarks import BenchmarkStrategies
from src.backtesting import BacktestEngine

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_1_unit_tests():
    print_section("CHECK 1: Unit Tests - Module Imports")
    
    try:
        from src.agents import DataAgent, AnalysisAgent, DecisionAgent
        from src.models import LLMTrader, SentimentAnalyzer
        from src.strategies import AlphaMining, RiskManagement, Portfolio
        from src.backtesting import BacktestEngine
        from src.utils import log, config
        print("✓ All core modules import successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False

def check_2_data_quality():
    print_section("CHECK 2: Data Quality & Lookahead Bias")
    
    try:
        data_agent = DataAgent()
        symbols = ['AAPL']
        
        print(f"Fetching data for {symbols}...")
        data = data_agent.collect_market_data(symbols)
        processed_data = data_agent.process_data(data)
        
        if not processed_data:
            print("✗ No data collected")
            return False
        
        validator = DataValidator()
        
        quality_report = validator.check_data_quality(processed_data)
        print(f"✓ Data quality check completed for {len(quality_report)} symbols")
        
        for symbol, report in quality_report.items():
            print(f"  {symbol}: {report['total_rows']} rows, {report['duplicate_dates']} duplicates")
        
        lookahead_ok = validator.check_lookahead_bias(processed_data)
        if lookahead_ok:
            print("✓ No lookahead bias detected")
        else:
            print("✗ Lookahead bias detected - FIX BEFORE PROCEEDING")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data quality check failed: {str(e)}")
        return False

def check_3_single_symbol_backtest():
    print_section("CHECK 3: Single Symbol Backtest (AAPL 2023-2024)")
    
    try:
        data_agent = DataAgent()
        symbols = ['AAPL']
        
        print("Collecting data...")
        data = data_agent.collect_market_data(symbols)
        processed_data = data_agent.process_data(data)
        
        if not processed_data:
            print("✗ No data for backtest")
            return False
        
        print("Running backtest...")
        backtest_engine = BacktestEngine()
        result = backtest_engine.run_backtest(processed_data, '2023-01-01', '2024-12-31')
        
        print(f"✓ Backtest completed")
        print(f"  Total Return: {result['total_return']:.2%}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"  Win Rate: {result['win_rate']:.2%}")
        print(f"  Total Trades: {result['num_trades']}")
        
        if result['sharpe_ratio'] < 0:
            print("⚠ Warning: Negative Sharpe ratio")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_4_benchmark_comparison():
    print_section("CHECK 4: Benchmark Strategy Comparison")
    
    try:
        data_agent = DataAgent()
        symbols = ['AAPL', 'MSFT']
        
        print("Collecting data...")
        data = data_agent.collect_market_data(symbols)
        processed_data = data_agent.process_data(data)
        
        if not processed_data:
            print("✗ No data for benchmarks")
            return False
        
        print("Running benchmark strategies...")
        benchmarks = BenchmarkStrategies.run_all_benchmarks(processed_data, 100000.0)
        
        print(f"✓ Benchmarks completed: {len(benchmarks)} strategies")
        print(f"\n{'Strategy':<40} {'Return':<12} {'Sharpe':<10}")
        print("-" * 80)
        for benchmark in benchmarks:
            print(f"{benchmark['strategy']:<40} {benchmark['total_return']:>10.2%}  {benchmark['sharpe_ratio']:>8.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_5_transaction_costs():
    print_section("CHECK 5: Transaction Costs & Slippage")
    
    try:
        commission = config.get('backtest.commission', 0.0005)
        slippage = config.get('backtest.slippage', 0.001)
        
        print(f"✓ Commission configured: {commission*100:.2f}% ({commission*10000:.0f} bps)")
        print(f"✓ Slippage configured: {slippage*100:.2f}% ({slippage*10000:.0f} bps)")
        
        if commission == 0 or slippage == 0:
            print("⚠ Warning: Zero transaction costs - unrealistic!")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Transaction cost check failed: {str(e)}")
        return False

def check_6_model_size():
    print_section("CHECK 6: Model Size Check")
    
    try:
        model_name = config.get('llm.model_name', '')
        testing_models = config.get('llm.testing_models', [])
        
        print(f"Production model: {model_name}")
        print(f"Testing models available: {len(testing_models)}")
        for model in testing_models:
            print(f"  - {model}")
        
        if '7B' in model_name or '8B' in model_name:
            print("⚠ Warning: Using large model (7B/8B) - consider testing with smaller model first")
            print(f"  Recommended: Use one of the testing models for initial validation")
        else:
            print("✓ Model size appropriate for testing")
        
        return True
        
    except Exception as e:
        print(f"✗ Model size check failed: {str(e)}")
        return False

def main():
    print("\n" + "=" * 80)
    print("  AUTONOMOUS TRADING SYSTEM - PRE-TEST VALIDATION CHECKLIST")
    print("=" * 80)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    checks = [
        ("Unit Tests", check_1_unit_tests),
        ("Data Quality", check_2_data_quality),
        ("Single Symbol Backtest", check_3_single_symbol_backtest),
        ("Benchmark Comparison", check_4_benchmark_comparison),
        ("Transaction Costs", check_5_transaction_costs),
        ("Model Size", check_6_model_size)
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {str(e)}")
            results[name] = False
    
    print_section("FINAL RESULTS")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<10} {name}")
    
    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
    print("-" * 80)
    
    if passed == total:
        print("\n✓ ALL CHECKS PASSED - System ready for full testing")
        print("\nNext steps:")
        print("  1. Run full backtest: python main.py backtest")
        print("  2. Run walk-forward validation (if enabled in config)")
        print("  3. Compare against benchmarks")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Fix issues before proceeding")
        print("\nRecommended actions:")
        if not results.get("Data Quality"):
            print("  - Fix lookahead bias in data processing")
        if not results.get("Transaction Costs"):
            print("  - Configure realistic commission and slippage")
        if not results.get("Single Symbol Backtest"):
            print("  - Debug backtest engine errors")
        return 1

if __name__ == "__main__":
    exit(main())
