#!/usr/bin/env python3
"""
Comprehensive system test script.
Tests all major components of the trading system.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import log
from src.models.llm_trader import LLMTrader
from src.agents import AnalysisAgent, DecisionAgent
import pandas as pd
import torch

def test_llm_loading():
    """Test if LLM loads correctly"""
    print("=" * 70)
    print("TEST 1: LLM Loading")
    print("=" * 70)
    
    try:
        llm = LLMTrader()
        
        if llm.model is None:
            print("❌ LLM model failed to load")
            print("   Check:")
            print("   1. HuggingFace token is set in .env")
            print("   2. Model name is correct")
            print("   3. GPU/CUDA is available")
            return False
        else:
            print(f"✅ LLM model loaded: {llm.model_name}")
            print(f"   Device: {llm.device}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            return True
    except Exception as e:
        print(f"❌ Error loading LLM: {e}")
        return False

def test_trading_decision():
    """Test if LLM can make a trading decision"""
    print("\n" + "=" * 70)
    print("TEST 2: LLM Trading Decision")
    print("=" * 70)
    
    try:
        llm = LLMTrader()
        
        test_data = {
            'RSI': 35.0,
            'MACD': 1.5,
            'MACD_Signal': 1.0,
            'SMA_20': 150.0,
            'SMA_50': 145.0,
            'Close': 152.0,
            'current_price': 152.0
        }
        
        decision = llm.generate_trading_decision(
            symbol='AAPL',
            technical_data=test_data,
            sentiment_data={'overall_sentiment': 0.5, 'signal': 'positive'},
            alpha_signals={'combined_score': 0.3},
            market_regime='bullish'
        )
        
        print(f"Decision: {decision['action']}")
        print(f"Confidence: {decision['confidence']:.2f}")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        
        if 'Fallback' in decision.get('reasoning', ''):
            print("⚠️  Using fallback logic - LLM may not be working")
            return False
        else:
            print("✅ LLM generated decision successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error generating decision: {e}")
        return False

def test_analysis_agent():
    """Test if AnalysisAgent works"""
    print("\n" + "=" * 70)
    print("TEST 3: Analysis Agent")
    print("=" * 70)
    
    try:
        agent = AnalysisAgent()
        
        dates = pd.date_range('2024-01-01', periods=100)
        test_df = pd.DataFrame({
            'Date': dates,
            'Close': 150 + pd.Series(range(100)) * 0.5,
            'Volume': 1000000,
            'RSI': 45.0,
            'MACD': 1.0,
            'MACD_Signal': 0.8,
            'SMA_20': 150.0,
            'SMA_50': 148.0,
            'Signal': 1
        })
        
        data = {'AAPL': test_df}
        result = agent.run(data, {})
        
        signals = result['signals']
        
        if 'AAPL' in signals:
            signal = signals['AAPL']
            print(f"Signal for AAPL: {signal['action']}")
            print(f"Combined score: {signal['combined_score']:.3f}")
            print(f"LLM action: {signal['llm_action']}")
            print(f"LLM confidence: {signal['llm_confidence']:.2f}")
            print("✅ Analysis Agent working")
            return True
        else:
            print("❌ No signals generated")
            return False
            
    except Exception as e:
        print(f"❌ Error in Analysis Agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 70)
    print("AUTONOMOUS TRADING SYSTEM - COMPREHENSIVE TEST")
    print("=" * 70)
    
    results = []
    
    results.append(("LLM Loading", test_llm_loading()))
    results.append(("LLM Decision", test_trading_decision()))
    results.append(("Analysis Agent", test_analysis_agent()))
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! System is ready for backtesting.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running backtest.")

if __name__ == "__main__":
    main()
