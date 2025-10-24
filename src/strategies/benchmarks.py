import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils import log

class BenchmarkStrategies:
    
    @staticmethod
    def buy_and_hold(data: Dict[str, pd.DataFrame], initial_capital: float = 100000.0) -> Dict:
        log.info("Running Buy and Hold benchmark...")
        
        equity_curve = [initial_capital]
        trades = []
        
        symbols = list(data.keys())
        capital_per_symbol = initial_capital / len(symbols)
        
        positions = {}
        for symbol, df in data.items():
            if len(df) < 2:
                continue
            
            first_price = df.iloc[0]['Close']
            shares = int(capital_per_symbol / first_price)
            
            positions[symbol] = {
                'shares': shares,
                'entry_price': first_price,
                'entry_date': df.iloc[0]['Date'] if 'Date' in df.columns else 0
            }
            
            trades.append({
                'symbol': symbol,
                'action': 'BUY',
                'shares': shares,
                'price': first_price,
                'date': positions[symbol]['entry_date']
            })
        
        for i in range(1, min(len(df) for df in data.values())):
            total_value = 0
            for symbol, position in positions.items():
                current_price = data[symbol].iloc[i]['Close']
                total_value += position['shares'] * current_price
            
            equity_curve.append(total_value)
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        return {
            'strategy': 'Buy and Hold',
            'final_equity': equity_curve[-1],
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    @staticmethod
    def sma_crossover(data: Dict[str, pd.DataFrame], initial_capital: float = 100000.0, 
                      fast_period: int = 50, slow_period: int = 200) -> Dict:
        log.info(f"Running SMA Crossover ({fast_period}/{slow_period}) benchmark...")
        
        from src.strategies.portfolio import Portfolio
        portfolio = Portfolio(initial_capital)
        
        max_len = min(len(df) for df in data.values())
        
        for i in range(max(fast_period, slow_period), max_len):
            current_prices = {}
            signals = {}
            
            for symbol, df in data.items():
                if i >= len(df):
                    continue
                
                current_prices[symbol] = df.iloc[i]['Close']
                
                sma_fast = df.iloc[i-fast_period:i]['Close'].mean()
                sma_slow = df.iloc[i-slow_period:i]['Close'].mean()
                
                if sma_fast > sma_slow:
                    signals[symbol] = 'BUY'
                elif sma_fast < sma_slow:
                    signals[symbol] = 'SELL'
                else:
                    signals[symbol] = 'HOLD'
            
            for symbol, signal in signals.items():
                price = current_prices[symbol]
                
                if signal == 'BUY' and not portfolio.has_position(symbol):
                    shares = int((portfolio.cash / len(data)) / price)
                    if shares > 0:
                        portfolio.open_position(symbol, shares, price)
                
                elif signal == 'SELL' and portfolio.has_position(symbol):
                    portfolio.close_position(symbol, price, reason='sma_crossover')
            
            portfolio.update_positions(current_prices)
        
        for symbol in list(portfolio.positions.keys()):
            final_price = data[symbol].iloc[-1]['Close']
            portfolio.close_position(symbol, final_price, reason='end')
        
        metrics = portfolio.calculate_metrics()
        
        return {
            'strategy': f'SMA Crossover ({fast_period}/{slow_period})',
            'final_equity': metrics.get('current_equity', initial_capital),
            'total_return': metrics.get('total_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'equity_curve': portfolio.get_equity_curve()['Equity'].tolist()
        }
    
    @staticmethod
    def rsi_oversold(data: Dict[str, pd.DataFrame], initial_capital: float = 100000.0,
                     rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> Dict:
        log.info(f"Running RSI Oversold/Overbought benchmark...")
        
        from src.strategies.portfolio import Portfolio
        portfolio = Portfolio(initial_capital)
        
        max_len = min(len(df) for df in data.values())
        
        for i in range(rsi_period + 1, max_len):
            current_prices = {}
            signals = {}
            
            for symbol, df in data.items():
                if i >= len(df) or 'RSI' not in df.columns:
                    continue
                
                current_prices[symbol] = df.iloc[i]['Close']
                rsi = df.iloc[i]['RSI']
                
                if rsi < oversold:
                    signals[symbol] = 'BUY'
                elif rsi > overbought:
                    signals[symbol] = 'SELL'
                else:
                    signals[symbol] = 'HOLD'
            
            for symbol, signal in signals.items():
                price = current_prices[symbol]
                
                if signal == 'BUY' and not portfolio.has_position(symbol):
                    shares = int((portfolio.cash / len(data)) / price)
                    if shares > 0:
                        portfolio.open_position(symbol, shares, price)
                
                elif signal == 'SELL' and portfolio.has_position(symbol):
                    portfolio.close_position(symbol, price, reason='rsi_signal')
            
            portfolio.update_positions(current_prices)
        
        for symbol in list(portfolio.positions.keys()):
            final_price = data[symbol].iloc[-1]['Close']
            portfolio.close_position(symbol, final_price, reason='end')
        
        metrics = portfolio.calculate_metrics()
        
        return {
            'strategy': f'RSI Oversold/Overbought ({oversold}/{overbought})',
            'final_equity': metrics.get('current_equity', initial_capital),
            'total_return': metrics.get('total_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'equity_curve': portfolio.get_equity_curve()['Equity'].tolist()
        }
    
    @staticmethod
    def run_all_benchmarks(data: Dict[str, pd.DataFrame], initial_capital: float = 100000.0) -> List[Dict]:
        log.info("Running all benchmark strategies...")
        
        benchmarks = []
        
        try:
            benchmarks.append(BenchmarkStrategies.buy_and_hold(data, initial_capital))
        except Exception as e:
            log.error(f"Buy and Hold failed: {str(e)}")
        
        try:
            benchmarks.append(BenchmarkStrategies.sma_crossover(data, initial_capital, 50, 200))
        except Exception as e:
            log.error(f"SMA Crossover failed: {str(e)}")
        
        try:
            benchmarks.append(BenchmarkStrategies.rsi_oversold(data, initial_capital))
        except Exception as e:
            log.error(f"RSI strategy failed: {str(e)}")
        
        log.info(f"Completed {len(benchmarks)} benchmark strategies")
        
        for benchmark in benchmarks:
            log.info(f"{benchmark['strategy']}: Return={benchmark['total_return']:.2%}, Sharpe={benchmark['sharpe_ratio']:.2f}")
        
        return benchmarks
