"""
Hybrid Trading Modes - Feature 7

Implements three trading modes:
1. Intraday: 5-min signals (RSI, MACD, sentiment spikes)
2. Interday: Daily signals (VWAP, SMA trends, alphas)
3. Hybrid: 60% intraday, 40% interday allocation

This enables the system to trade across multiple timeframes autonomously.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from enum import Enum
from src.utils import log, config
from src.strategies.portfolio import Portfolio
from src.strategies.risk_manager import RiskManager

class TradingMode(Enum):
    """Trading mode enumeration"""
    INTRADAY = "intraday"
    INTERDAY = "interday"
    HYBRID = "hybrid"

class HybridTrader:
    """
    Hybrid Trading System supporting multiple timeframes
    
    Features:
    - Intraday: 5-min bars, fast signals (RSI, MACD, sentiment)
    - Interday: Daily bars, slow signals (VWAP, SMA, alphas)
    - Hybrid: Automatic allocation between modes
    - Market-aware scheduling (NYSE hours)
    - Risk-adjusted position sizing per mode
    """
    
    def __init__(
        self,
        mode: str = None,
        intraday_allocation: float = None,
        interday_allocation: float = None
    ):
        """
        Initialize hybrid trader
        
        Args:
            mode: Trading mode ('intraday', 'interday', 'hybrid')
            intraday_allocation: % of capital for intraday (0-1)
            interday_allocation: % of capital for interday (0-1)
        """
        hybrid_config = config.get('trading.hybrid_mode', {})
        
        self.mode = TradingMode(mode or hybrid_config.get('mode', 'hybrid'))
        
        if self.mode == TradingMode.INTRADAY:
            self.intraday_allocation = 1.0
            self.interday_allocation = 0.0
        elif self.mode == TradingMode.INTERDAY:
            self.intraday_allocation = 0.0
            self.interday_allocation = 1.0
        else:  # HYBRID
            self.intraday_allocation = intraday_allocation or hybrid_config.get('intraday_allocation', 0.6)
            self.interday_allocation = interday_allocation or hybrid_config.get('interday_allocation', 0.4)
        
        total = self.intraday_allocation + self.interday_allocation
        if abs(total - 1.0) > 0.01:
            log.warning(f"Allocations sum to {total:.2f}, normalizing to 1.0")
            self.intraday_allocation /= total
            self.interday_allocation /= total
        
        self.intraday_config = {
            'timeframe': '5min',
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sentiment_threshold': 0.5,
            'max_positions': 10,
            'position_size': 0.1  # 10% per position
        }
        
        self.interday_config = {
            'timeframe': '1d',
            'sma_short': 20,
            'sma_long': 50,
            'vwap_period': 20,
            'alpha_threshold': 0.3,
            'max_positions': 5,
            'position_size': 0.2  # 20% per position
        }
        
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        log.info(f"HybridTrader initialized in {self.mode.value} mode")
        log.info(f"Allocations: Intraday={self.intraday_allocation:.1%}, Interday={self.interday_allocation:.1%}")
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        current_time: datetime = None
    ) -> Dict[str, Dict]:
        """
        Generate trading signals based on current mode
        
        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            current_time: Current timestamp (for market hours check)
            
        Returns:
            Dict of symbol -> signal dict with action, confidence, mode
        """
        if current_time is None:
            current_time = datetime.now()
        
        signals = {}
        
        is_market_open = self._is_market_open(current_time)
        
        if self.intraday_allocation > 0 and is_market_open:
            intraday_signals = self._generate_intraday_signals(data)
            for symbol, signal in intraday_signals.items():
                signal['mode'] = 'intraday'
                signal['allocation'] = self.intraday_allocation
                signals[symbol] = signal
        
        if self.interday_allocation > 0:
            interday_signals = self._generate_interday_signals(data)
            for symbol, signal in interday_signals.items():
                if symbol in signals and self.mode == TradingMode.HYBRID:
                    signals[symbol] = self._combine_signals(
                        signals[symbol],
                        signal
                    )
                else:
                    signal['mode'] = 'interday'
                    signal['allocation'] = self.interday_allocation
                    signals[symbol] = signal
        
        return signals
    
    def _generate_intraday_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate intraday signals using 5-min data
        
        Uses: RSI, MACD, sentiment spikes
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < 50:
                continue
            
            try:
                rsi = self._calculate_rsi(df['Close'], self.intraday_config['rsi_period'])
                macd, macd_signal, macd_hist = self._calculate_macd(
                    df['Close'],
                    self.intraday_config['macd_fast'],
                    self.intraday_config['macd_slow'],
                    self.intraday_config['macd_signal']
                )
                
                current_rsi = rsi.iloc[-1]
                current_macd = macd.iloc[-1]
                current_macd_signal = macd_signal.iloc[-1]
                current_macd_hist = macd_hist.iloc[-1]
                
                sentiment = df.get('Sentiment', pd.Series([0.5] * len(df))).iloc[-1]
                
                action = 'HOLD'
                confidence = 0.5
                reasoning = []
                
                if current_rsi < self.intraday_config['rsi_oversold']:
                    action = 'BUY'
                    confidence += 0.2
                    reasoning.append(f"RSI oversold ({current_rsi:.1f})")
                elif current_rsi > self.intraday_config['rsi_overbought']:
                    action = 'SELL'
                    confidence += 0.2
                    reasoning.append(f"RSI overbought ({current_rsi:.1f})")
                
                if current_macd > current_macd_signal and current_macd_hist > 0:
                    if action == 'BUY':
                        confidence += 0.2
                    else:
                        action = 'BUY'
                        confidence += 0.1
                    reasoning.append("MACD bullish crossover")
                elif current_macd < current_macd_signal and current_macd_hist < 0:
                    if action == 'SELL':
                        confidence += 0.2
                    else:
                        action = 'SELL'
                        confidence += 0.1
                    reasoning.append("MACD bearish crossover")
                
                if sentiment > self.intraday_config['sentiment_threshold']:
                    if action == 'BUY':
                        confidence += 0.1
                    reasoning.append(f"Positive sentiment ({sentiment:.2f})")
                elif sentiment < -self.intraday_config['sentiment_threshold']:
                    if action == 'SELL':
                        confidence += 0.1
                    reasoning.append(f"Negative sentiment ({sentiment:.2f})")
                
                if confidence > 0.6 and action != 'HOLD':
                    signals[symbol] = {
                        'action': action,
                        'confidence': min(confidence, 1.0),
                        'reasoning': ', '.join(reasoning),
                        'indicators': {
                            'rsi': current_rsi,
                            'macd': current_macd,
                            'macd_signal': current_macd_signal,
                            'sentiment': sentiment
                        }
                    }
            
            except Exception as e:
                log.debug(f"Error generating intraday signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _generate_interday_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate interday signals using daily data
        
        Uses: VWAP, SMA trends, alpha signals
        """
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < 50:
                continue
            
            try:
                sma_short = df['Close'].rolling(self.interday_config['sma_short']).mean()
                sma_long = df['Close'].rolling(self.interday_config['sma_long']).mean()
                vwap = self._calculate_vwap(df, self.interday_config['vwap_period'])
                
                current_price = df['Close'].iloc[-1]
                current_sma_short = sma_short.iloc[-1]
                current_sma_long = sma_long.iloc[-1]
                current_vwap = vwap.iloc[-1]
                
                alpha = df.get('Alpha', pd.Series([0] * len(df))).iloc[-1]
                
                action = 'HOLD'
                confidence = 0.5
                reasoning = []
                
                if current_sma_short > current_sma_long:
                    action = 'BUY'
                    confidence += 0.2
                    reasoning.append(f"SMA{self.interday_config['sma_short']} > SMA{self.interday_config['sma_long']}")
                elif current_sma_short < current_sma_long:
                    action = 'SELL'
                    confidence += 0.2
                    reasoning.append(f"SMA{self.interday_config['sma_short']} < SMA{self.interday_config['sma_long']}")
                
                if current_price > current_vwap:
                    if action == 'BUY':
                        confidence += 0.15
                    reasoning.append(f"Price above VWAP")
                elif current_price < current_vwap:
                    if action == 'SELL':
                        confidence += 0.15
                    reasoning.append(f"Price below VWAP")
                
                if alpha > self.interday_config['alpha_threshold']:
                    if action == 'BUY':
                        confidence += 0.15
                    else:
                        action = 'BUY'
                        confidence += 0.1
                    reasoning.append(f"Strong alpha ({alpha:.2f})")
                elif alpha < -self.interday_config['alpha_threshold']:
                    if action == 'SELL':
                        confidence += 0.15
                    else:
                        action = 'SELL'
                        confidence += 0.1
                    reasoning.append(f"Weak alpha ({alpha:.2f})")
                
                if confidence > 0.6 and action != 'HOLD':
                    signals[symbol] = {
                        'action': action,
                        'confidence': min(confidence, 1.0),
                        'reasoning': ', '.join(reasoning),
                        'indicators': {
                            'sma_short': current_sma_short,
                            'sma_long': current_sma_long,
                            'vwap': current_vwap,
                            'alpha': alpha
                        }
                    }
            
            except Exception as e:
                log.debug(f"Error generating interday signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _combine_signals(self, intraday_signal: Dict, interday_signal: Dict) -> Dict:
        """
        Combine intraday and interday signals for hybrid mode
        
        Uses weighted average based on allocations
        """
        if intraday_signal['action'] == interday_signal['action']:
            combined_confidence = (
                intraday_signal['confidence'] * self.intraday_allocation +
                interday_signal['confidence'] * self.interday_allocation
            )
            
            return {
                'action': intraday_signal['action'],
                'confidence': min(combined_confidence * 1.2, 1.0),  # 20% boost for agreement
                'reasoning': f"Intraday: {intraday_signal['reasoning']}; Interday: {interday_signal['reasoning']}",
                'mode': 'hybrid',
                'allocation': 1.0,
                'indicators': {
                    **intraday_signal.get('indicators', {}),
                    **interday_signal.get('indicators', {})
                }
            }
        
        else:
            intraday_weight = intraday_signal['confidence'] * self.intraday_allocation
            interday_weight = interday_signal['confidence'] * self.interday_allocation
            
            if intraday_weight > interday_weight:
                return {
                    **intraday_signal,
                    'confidence': intraday_weight,
                    'mode': 'hybrid',
                    'allocation': self.intraday_allocation,
                    'reasoning': f"Intraday dominant: {intraday_signal['reasoning']}"
                }
            else:
                return {
                    **interday_signal,
                    'confidence': interday_weight,
                    'mode': 'hybrid',
                    'allocation': self.interday_allocation,
                    'reasoning': f"Interday dominant: {interday_signal['reasoning']}"
                }
    
    def _is_market_open(self, current_time: datetime) -> bool:
        """Check if market is currently open"""
        if current_time.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        current_time_only = current_time.time()
        return self.market_open <= current_time_only <= self.market_close
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate VWAP indicator"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
        return vwap
    
    def get_mode_info(self) -> Dict:
        """Get current mode information"""
        return {
            'mode': self.mode.value,
            'intraday_allocation': self.intraday_allocation,
            'interday_allocation': self.interday_allocation,
            'intraday_config': self.intraday_config,
            'interday_config': self.interday_config
        }
