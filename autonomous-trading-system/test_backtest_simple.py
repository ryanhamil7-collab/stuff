#!/usr/bin/env python3
"""
Simple backtest test that only imports what's needed
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import yfinance as yf
from datetime import datetime

print("Testing data fetching...")
symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
start_date = '2023-01-01'
end_date = '2024-10-24'

data = {}
for symbol in symbols:
    print(f"Fetching {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    if not df.empty:
        df.reset_index(inplace=True)
        df['Symbol'] = symbol
        data[symbol] = df
        print(f"  Got {len(df)} rows for {symbol}")

print(f"\nSuccessfully fetched data for {len(data)} symbols")
print(f"Date range: {start_date} to {end_date}")

print("\nTesting simple buy-and-hold strategy...")
initial_capital = 100000.0
capital_per_symbol = initial_capital / len(data)

results = {}
for symbol, df in data.items():
    if len(df) < 2:
        continue
    
    start_price = df.iloc[0]['Close']
    end_price = df.iloc[-1]['Close']
    shares = capital_per_symbol / start_price
    final_value = shares * end_price
    return_pct = (final_value - capital_per_symbol) / capital_per_symbol
    
    results[symbol] = {
        'start_price': start_price,
        'end_price': end_price,
        'shares': shares,
        'final_value': final_value,
        'return': return_pct
    }
    
    print(f"{symbol}: {return_pct:>8.2%} return (${capital_per_symbol:,.2f} -> ${final_value:,.2f})")

total_final_value = sum(r['final_value'] for r in results.values())
total_return = (total_final_value - initial_capital) / initial_capital

print(f"\nPortfolio Results:")
print(f"Initial Capital:  ${initial_capital:>12,.2f}")
print(f"Final Capital:    ${total_final_value:>12,.2f}")
print(f"Total Return:     {total_return:>12.2%}")
print(f"Number of Trades: {len(results) * 2:>12}")  # Buy + Sell for each symbol

print("\n✅ Simple backtest completed successfully!")
