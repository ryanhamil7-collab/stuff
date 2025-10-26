"""
Test PromptAgent to verify it generates balanced BUY/SELL/HOLD signals
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agents.prompt_agent import PromptAgent

def create_test_data(scenario: str) -> pd.DataFrame:
    """Create test market data for different scenarios"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    if scenario == 'bullish':
        prices = np.linspace(100, 150, 100) + np.random.randn(100) * 2
        rsi = 65
        macd = 2.5
        macd_signal = 1.5
    elif scenario == 'bearish':
        prices = np.linspace(150, 100, 100) + np.random.randn(100) * 2
        rsi = 35
        macd = -2.5
        macd_signal = -1.5
    elif scenario == 'overbought':
        prices = np.linspace(100, 180, 100) + np.random.randn(100) * 2
        rsi = 75
        macd = 3.0
        macd_signal = 2.0
    elif scenario == 'oversold':
        prices = np.linspace(150, 80, 100) + np.random.randn(100) * 2
        rsi = 25
        macd = -3.0
        macd_signal = -2.0
    else:
        prices = 100 + np.random.randn(100) * 5
        rsi = 50
        macd = 0.0
        macd_signal = 0.0
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'RSI': rsi + np.random.randn(100) * 2,
        'MACD': macd + np.random.randn(100) * 0.5,
        'MACD_Signal': macd_signal + np.random.randn(100) * 0.5,
        'SMA_20': prices * 0.98,
        'SMA_50': prices * 0.97,
        'BB_Position': 0.5 + np.random.randn(100) * 0.2,
        'ATR': 2.5 + np.random.randn(100) * 0.5
    }, index=dates)
    
    return df

def test_prompt_generation():
    """Test that PromptAgent generates appropriate prompts for different scenarios"""
    
    print("=" * 80)
    print("Testing PromptAgent - Balanced Signal Generation")
    print("=" * 80)
    print()
    
    agent = PromptAgent()
    
    scenarios = {
        'bullish': 'Should suggest BUY or HOLD',
        'bearish': 'Should suggest SELL or HOLD',
        'overbought': 'Should suggest SELL (take profits)',
        'oversold': 'Should suggest BUY (value opportunity)',
        'neutral': 'Should suggest HOLD (wait for clarity)'
    }
    
    for scenario, expected in scenarios.items():
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario.upper()}")
        print(f"Expected: {expected}")
        print(f"{'=' * 80}\n")
        
        market_data = create_test_data(scenario)
        
        technical_signals = {
            'RSI': market_data['RSI'].iloc[-1],
            'MACD': market_data['MACD'].iloc[-1],
            'MACD_Signal': market_data['MACD_Signal'].iloc[-1],
            'SMA_20': market_data['SMA_20'].iloc[-1],
            'SMA_50': market_data['SMA_50'].iloc[-1],
            'BB_Position': market_data['BB_Position'].iloc[-1],
            'ATR': market_data['ATR'].iloc[-1],
            'Signal': 1 if scenario in ['bullish', 'oversold'] else -1 if scenario in ['bearish', 'overbought'] else 0
        }
        
        sentiment_score = 0.5 if scenario in ['bullish', 'oversold'] else -0.5 if scenario in ['bearish', 'overbought'] else 0.0
        
        portfolio_state_no_position = {
            'cash': 100000,
            'total_value': 100000,
            'positions': {}
        }
        
        portfolio_state_with_position = {
            'cash': 50000,
            'total_value': 120000,
            'positions': {
                'AAPL': {
                    'shares': 100,
                    'entry_price': market_data['Close'].iloc[-50],
                    'current_price': market_data['Close'].iloc[-1],
                    'days_held': 50
                }
            }
        }
        
        print("Test 1: No existing position")
        print("-" * 80)
        prompt = agent.build_trading_prompt(
            symbol='AAPL',
            market_data=market_data,
            technical_signals=technical_signals,
            sentiment_score=sentiment_score,
            portfolio_state=portfolio_state_no_position
        )
        
        if 'BUY' in prompt and 'SELL' in prompt and 'HOLD' in prompt:
            print("✓ Prompt includes all three actions (BUY/SELL/HOLD)")
        else:
            print("✗ Prompt missing some actions")
        
        if 'You currently have NO position' in prompt:
            print("✓ Prompt correctly identifies no position")
        else:
            print("✗ Prompt doesn't identify position state")
        
        if scenario in ['bearish', 'overbought']:
            if 'DO NOT SELL' in prompt or 'no position to sell' in prompt:
                print("✓ Prompt correctly warns against selling without position")
            else:
                print("✗ Prompt should warn against selling without position")
        
        print()
        print("Test 2: With existing position")
        print("-" * 80)
        prompt = agent.build_trading_prompt(
            symbol='AAPL',
            market_data=market_data,
            technical_signals=technical_signals,
            sentiment_score=sentiment_score,
            portfolio_state=portfolio_state_with_position
        )
        
        if 'You currently HOLD this position' in prompt:
            print("✓ Prompt correctly identifies existing position")
        else:
            print("✗ Prompt doesn't identify position state")
        
        if scenario in ['bullish', 'oversold']:
            if 'DO NOT BUY MORE' in prompt or 'already have a position' in prompt:
                print("✓ Prompt correctly warns against buying more")
            else:
                print("✗ Prompt should warn against buying more")
        
        if 'Exit signals' in prompt or 'SELL' in prompt:
            print("✓ Prompt includes exit strategy guidance")
        else:
            print("✗ Prompt missing exit strategy")
        
        pnl_pct = ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[-50]) / 
                   market_data['Close'].iloc[-50] * 100)
        print(f"  Position P&L: {pnl_pct:+.2f}%")
        
        if scenario in ['overbought', 'bearish'] and pnl_pct > 10:
            if 'profit target reached' in prompt.lower() or 'take profit' in prompt.lower():
                print("✓ Prompt suggests taking profits")
            else:
                print("  (Could suggest taking profits)")
    
    print("\n" + "=" * 80)
    print("Testing Decision Tracking")
    print("=" * 80)
    
    for i, action in enumerate(['BUY', 'SELL', 'HOLD', 'BUY', 'HOLD', 'SELL', 'HOLD', 'HOLD']):
        agent.track_decision('TEST', {
            'action': action,
            'confidence': 0.7,
            'reasoning': f'Test decision {i+1}'
        })
    
    stats = agent.get_decision_stats()
    print(f"\nDecision Statistics:")
    print(f"  Total Decisions: {stats['total_decisions']}")
    print(f"  BUY:  {stats['buy_count']} ({stats['buy_pct']:.1f}%)")
    print(f"  SELL: {stats['sell_count']} ({stats['sell_pct']:.1f}%)")
    print(f"  HOLD: {stats['hold_count']} ({stats['hold_pct']:.1f}%)")
    print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
    
    if stats['buy_pct'] < 80 and stats['sell_pct'] > 0:
        print("\n✓ Decision distribution looks balanced (not stuck on BUY)")
    else:
        print("\n✗ Decision distribution may be biased")
    
    print("\n" + "=" * 80)
    print("PromptAgent Test Complete!")
    print("=" * 80)

if __name__ == '__main__':
    test_prompt_generation()
