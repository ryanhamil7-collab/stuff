#!/usr/bin/env python3
"""
Live Paper Trading with Alpaca

This script runs the trading system in live paper trading mode using Alpaca's API.
All trades are executed in Alpaca's paper trading environment (no real money).

Requirements:
- Alpaca paper trading account (free at alpaca.markets)
- API keys set in .env file or environment variables
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

from src.utils import log
from src.agents import DataAgent, AnalysisAgent, DecisionAgent
from src.strategies.portfolio import Portfolio
import time
from datetime import datetime

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    log.error("alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")
    sys.exit(1)

class AlpacaPaperTrader:
    """
    Live paper trading using Alpaca API
    """
    
    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            log.error("Alpaca API keys not found!")
            log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
            sys.exit(1)
        
        log.info("Connecting to Alpaca Paper Trading API...")
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        try:
            account = self.api.get_account()
            log.info(f"✓ Connected to Alpaca Paper Trading")
            log.info(f"  Account Status: {account.status}")
            log.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
            log.info(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
            log.info(f"  Cash: ${float(account.cash):,.2f}")
        except Exception as e:
            log.error(f"Failed to connect to Alpaca: {str(e)}")
            sys.exit(1)
        
        self.data_agent = DataAgent()
        self.analysis_agent = AnalysisAgent(use_prompt_agent=False)
        self.decision_agent = DecisionAgent()
        
        initial_capital = float(account.portfolio_value)
        self.portfolio = Portfolio(initial_capital)
        
        log.info("AlpacaPaperTrader initialized")
    
    def get_account_info(self):
        """Get current account information"""
        account = self.api.get_account()
        return {
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'equity': float(account.equity),
            'status': account.status
        }
    
    def get_positions(self):
        """Get current positions from Alpaca"""
        positions = self.api.list_positions()
        return [{
            'symbol': p.symbol,
            'qty': int(p.qty),
            'avg_entry_price': float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc)
        } for p in positions]
    
    def place_order(self, symbol: str, qty: int, side: str):
        """
        Place an order on Alpaca
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
        """
        try:
            log.info(f"Placing {side.upper()} order: {qty} shares of {symbol}")
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            log.info(f"✓ Order placed successfully: {order.id}")
            log.info(f"  Symbol: {order.symbol}")
            log.info(f"  Qty: {order.qty}")
            log.info(f"  Side: {order.side}")
            log.info(f"  Status: {order.status}")
            
            return order
            
        except Exception as e:
            log.error(f"Failed to place order: {str(e)}")
            return None
    
    def run_single_cycle(self):
        """Run a single trading cycle"""
        log.info("=" * 80)
        log.info(f"Starting trading cycle at {datetime.now()}")
        log.info("=" * 80)
        
        log.info("Step 1: Fetching account info...")
        account_info = self.get_account_info()
        log.info(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        log.info(f"  Cash: ${account_info['cash']:,.2f}")
        
        log.info("Step 2: Fetching current positions...")
        positions = self.get_positions()
        log.info(f"  Open Positions: {len(positions)}")
        for pos in positions:
            log.info(f"    {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f} "
                    f"(P&L: ${pos['unrealized_pl']:,.2f} / {pos['unrealized_plpc']:.2%})")
        
        log.info("Step 3: Collecting market data...")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        data = self.data_agent.collect_market_data(symbols)
        processed_data = self.data_agent.process_data(data)
        
        if not processed_data:
            log.warning("No data collected, skipping cycle")
            return
        
        log.info(f"Step 4: Analyzing {len(processed_data)} symbols...")
        analysis_result = self.analysis_agent.run(processed_data, {})
        signals = analysis_result.get('signals', {})
        
        log.info("Step 5: Generating trading signals...")
        for symbol, signal in signals.items():
            log.info(f"  {symbol}: {signal['action']} (score: {signal['combined_score']:.3f})")
        
        log.info("Step 6: Making trading decisions...")
        current_prices = {}
        for symbol, df in processed_data.items():
            if len(df) > 0:
                current_prices[symbol] = df.iloc[-1]['Close']
        
        decision_result = self.decision_agent.run(
            signals,
            self.portfolio,
            current_prices
        )
        
        decisions = decision_result.get('decisions', [])
        log.info(f"  Generated {len(decisions)} trading decisions")
        
        log.info("Step 7: Executing trades on Alpaca...")
        executed_count = 0
        for decision in decisions:
            symbol = decision['symbol']
            action = decision['action']
            shares = decision.get('shares', 0)
            
            if shares > 0:
                side = 'buy' if action == 'BUY' else 'sell'
                order = self.place_order(symbol, shares, side)
                if order:
                    executed_count += 1
                    time.sleep(1)
        
        log.info(f"✓ Executed {executed_count} trades")
        
        log.info("Step 8: Final account status...")
        final_account = self.get_account_info()
        log.info(f"  Portfolio Value: ${final_account['portfolio_value']:,.2f}")
        log.info(f"  Cash: ${final_account['cash']:,.2f}")
        
        log.info("=" * 80)
        log.info("Trading cycle completed")
        log.info("=" * 80)
    
    def run_continuous(self, interval_seconds=300):
        """
        Run continuous trading
        
        Args:
            interval_seconds: Time between trading cycles (default: 300 = 5 minutes)
        """
        log.info("=" * 80)
        log.info("STARTING CONTINUOUS PAPER TRADING")
        log.info("=" * 80)
        log.info(f"⚠️  PAPER TRADING ONLY - No real money at risk")
        log.info(f"Check interval: {interval_seconds} seconds ({interval_seconds/60:.1f} minutes)")
        log.info("Press Ctrl+C to stop")
        log.info("=" * 80)
        
        try:
            while True:
                self.run_single_cycle()
                
                log.info(f"\nWaiting {interval_seconds} seconds until next cycle...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            log.info("\n\nReceived interrupt signal, shutting down...")
            log.info("Final account status:")
            account = self.get_account_info()
            log.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
            log.info(f"  Cash: ${account['cash']:,.2f}")
            log.info("\n✓ Shutdown complete")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Paper Trading with Alpaca')
    parser.add_argument('--single', action='store_true',
                        help='Run a single trading cycle and exit')
    parser.add_argument('--interval', type=int, default=300,
                        help='Interval between cycles in seconds (default: 300 = 5 min)')
    args = parser.parse_args()
    
    trader = AlpacaPaperTrader()
    
    if args.single:
        trader.run_single_cycle()
    else:
        trader.run_continuous(interval_seconds=args.interval)

if __name__ == "__main__":
    main()
