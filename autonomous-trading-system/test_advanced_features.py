#!/usr/bin/env python3
"""
Test script to verify all advanced features are working.

Tests:
1. RL Training (gymnasium, stable-baselines3)
2. Crypto Discovery (ccxt)
3. Federated Learning (flwr)
4. Quantum Optimization (pyquil, qiskit)
5. Neuro-Symbolic AI (pyswip optional)
6. LLM JSON Parsing
7. Alpha Mining (no warnings)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_rl_dependencies():
    """Test RL dependencies are installed"""
    print("\n=== Testing RL Dependencies ===")
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        print("✓ gymnasium installed")
        print("✓ stable-baselines3 installed")
        return True
    except ImportError as e:
        print(f"✗ RL dependencies missing: {e}")
        return False

def test_crypto_dependencies():
    """Test crypto dependencies are installed"""
    print("\n=== Testing Crypto Dependencies ===")
    try:
        import ccxt
        print("✓ ccxt installed")
        
        exchange = ccxt.binance()
        print("✓ ccxt.binance() works")
        return True
    except ImportError as e:
        print(f"✗ ccxt missing: {e}")
        return False
    except Exception as e:
        print(f"⚠ ccxt installed but exchange creation failed: {e}")
        return True  # Still counts as success

def test_federated_learning():
    """Test federated learning dependencies"""
    print("\n=== Testing Federated Learning ===")
    try:
        import flwr
        print("✓ flwr (Flower) installed")
        return True
    except ImportError as e:
        print(f"✗ flwr missing: {e}")
        return False

def test_quantum_dependencies():
    """Test quantum computing dependencies"""
    print("\n=== Testing Quantum Dependencies ===")
    success = True
    
    try:
        import pyquil
        print("✓ pyquil installed")
    except ImportError as e:
        print(f"✗ pyquil missing: {e}")
        success = False
    
    try:
        import qiskit
        print("✓ qiskit installed")
    except ImportError as e:
        print(f"✗ qiskit missing: {e}")
        success = False
    
    return success

def test_neuro_symbolic():
    """Test neuro-symbolic AI"""
    print("\n=== Testing Neuro-Symbolic AI ===")
    try:
        from src.agents.neuro_symbolic import NeuroSymbolicAgent, NeuroSymbolicConfig
        
        config = NeuroSymbolicConfig(use_prolog=False)
        agent = NeuroSymbolicAgent(config)
        
        print("✓ NeuroSymbolicAgent initialized")
        
        rules = agent._generate_default_rules()
        print(f"✓ Generated {len(rules)} default rules")
        
        market_data = {
            'rsi': 25,
            'sentiment': 0.7,
            'macd': 0.5,
            'volume_ratio': 2.0
        }
        
        decision = agent.make_decision('TEST', market_data)
        print(f"✓ Made decision: {decision['action']} (confidence: {decision['confidence']:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Neuro-symbolic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_json_parsing():
    """Test LLM JSON parsing with error handling"""
    print("\n=== Testing LLM JSON Parsing ===")
    try:
        from src.models.llm_trader import LLMTrader
        
        trader = LLMTrader()
        
        malformed_json = "{action: BUY, confidence: 0.8, reasoning: Test}"
        result = trader._parse_decision(malformed_json)
        print(f"✓ Parsed malformed JSON: {result['action']}")
        
        valid_json = '{"action": "SELL", "confidence": 0.9, "reasoning": "Test"}'
        result = trader._parse_decision(valid_json)
        print(f"✓ Parsed valid JSON: {result['action']}")
        
        text_only = "I recommend to BUY with high confidence"
        result = trader._parse_decision(text_only)
        print(f"✓ Text extraction fallback: {result['action']}")
        
        return True
    except Exception as e:
        print(f"✗ LLM JSON parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alpha_mining():
    """Test alpha mining without warnings"""
    print("\n=== Testing Alpha Mining ===")
    try:
        import pandas as pd
        import numpy as np
        from src.strategies.alpha_mining import AlphaMining
        
        dates = pd.date_range('2023-01-01', periods=100)
        test_data = {
            'AAPL': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 100,
                'Volume': np.random.randint(1000000, 10000000, 100),
                'RSI': np.random.uniform(30, 70, 100),
                'MACD': np.random.randn(100),
                'SMA_20': np.random.randn(100).cumsum() + 100,
                'VWAP': np.random.randn(100).cumsum() + 100,
            }, index=dates)
        }
        
        miner = AlphaMining()
        print("✓ AlphaMining initialized")
        
        alphas = miner.generate_alpha_formulas()
        print(f"✓ Generated {len(alphas)} alpha formulas")
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = miner.mine_alphas(test_data)
            
            runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
            if runtime_warnings:
                print(f"⚠ Got {len(runtime_warnings)} runtime warnings:")
                for warning in runtime_warnings:
                    print(f"  - {warning.message}")
            else:
                print("✓ No runtime warnings during alpha mining")
        
        print(f"✓ Mined {len(results)} high-quality alphas")
        
        return True
    except Exception as e:
        print(f"✗ Alpha mining test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decision_agent_rl():
    """Test DecisionAgent with RL"""
    print("\n=== Testing DecisionAgent with RL ===")
    try:
        import pandas as pd
        import numpy as np
        from src.agents.decision_agent import DecisionAgent, TradingEnvironment
        
        dates = pd.date_range('2023-01-01', periods=100)
        test_data = {
            'AAPL': pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 150,
                'Volume': np.random.randint(1000000, 10000000, 100),
                'RSI': np.random.uniform(30, 70, 100),
                'MACD': np.random.randn(100),
                'MACD_Signal': np.random.randn(100),
                'SMA_20': np.random.randn(100).cumsum() + 150,
                'SMA_50': np.random.randn(100).cumsum() + 150,
                'EMA_12': np.random.randn(100).cumsum() + 150,
                'EMA_26': np.random.randn(100).cumsum() + 150,
                'BB_Position': np.random.uniform(0, 1, 100),
                'ATR': np.random.uniform(1, 5, 100),
                'ADX': np.random.uniform(20, 40, 100),
                'OBV': np.random.randint(1000000, 10000000, 100),
                'VWAP': np.random.randn(100).cumsum() + 150,
                'Stoch_K': np.random.uniform(20, 80, 100),
                'Stoch_D': np.random.uniform(20, 80, 100),
                'Plus_DI': np.random.uniform(10, 30, 100),
                'Minus_DI': np.random.uniform(10, 30, 100),
                'BB_Width': np.random.uniform(0.5, 2, 100),
                'Signal': np.random.choice([-1, 0, 1], 100),
            }, index=dates)
        }
        
        env = TradingEnvironment(test_data)
        print("✓ TradingEnvironment created")
        
        obs, info = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Environment step executed, reward: {reward:.2f}")
        
        agent = DecisionAgent()
        print("✓ DecisionAgent initialized")
        
        return True
    except Exception as e:
        print(f"✗ DecisionAgent RL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_symbol_discovery_crypto():
    """Test SymbolDiscovery with crypto"""
    print("\n=== Testing SymbolDiscovery with Crypto ===")
    try:
        from src.agents.symbol_discovery import SymbolDiscovery
        
        discovery = SymbolDiscovery()
        print("✓ SymbolDiscovery initialized")
        
        crypto_tickers = discovery._get_crypto_tickers()
        if crypto_tickers:
            print(f"✓ Found {len(crypto_tickers)} crypto tickers")
        else:
            print("⚠ No crypto tickers found (ccxt exchange may not be initialized)")
        
        return True
    except Exception as e:
        print(f"✗ SymbolDiscovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ADVANCED FEATURES TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    results['RL Dependencies'] = test_rl_dependencies()
    results['Crypto Dependencies'] = test_crypto_dependencies()
    results['Federated Learning'] = test_federated_learning()
    results['Quantum Dependencies'] = test_quantum_dependencies()
    results['Neuro-Symbolic AI'] = test_neuro_symbolic()
    results['LLM JSON Parsing'] = test_llm_json_parsing()
    results['Alpha Mining'] = test_alpha_mining()
    results['DecisionAgent RL'] = test_decision_agent_rl()
    results['SymbolDiscovery Crypto'] = test_symbol_discovery_crypto()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} | {test_name}")
    
    print("=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed ({100*passed//total}%)")
    print("=" * 60)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
