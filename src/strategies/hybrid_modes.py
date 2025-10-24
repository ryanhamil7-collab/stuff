"""
Hybrid Trading Modes: Intraday, Interday, and Hybrid strategies.

Implements multi-horizon trading with different signals and metrics:
- Intraday: 5-min RSI/MACD crossovers, sentiment spikes
- Interday: Multi-day VWAP deviations, fundamentals
- Hybrid: Portfolio split (e.g., 60% intra, 40% inter)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Literal, Tuple
from dataclasses import dataclass
from src.utils import log

@dataclass
class HybridConfig:
    """Configuration for hybrid trading modes."""
    mode: Literal["intraday", "interday", "hybrid"] = "hybrid"
    intraday_allocation: float = 0.6  # 60% for intraday
    interday_allocation: float = 0.4  # 40% for interday
    
    intraday_timeframe: str = "5min"
    intraday_indicators: List[str] = None
    intraday_symbols: List[str] = None  # Volatile symbols like TSLA
    
    interday_timeframe: str = "1day"
    interday_indicators: List[str] = None
    interday_symbols: List[str] = None  # Stable symbols like MSFT
    
    def __post_init__(self):
        if self.intraday_indicators is None:
            self.intraday_indicators = ["rsi", "macd", "sentiment_spike", "volume_surge"]
        if self.interday_indicators is None:
            self.interday_indicators = ["vwap_deviation", "sma_trend", "fundamentals", "alpha_factors"]
        if self.intraday_symbols is None:
            self.intraday_symbols = ["TSLA", "NVDA", "AMD", "COIN"]
        if self.interday_symbols is None:
            self.interday_symbols = ["MSFT", "AAPL", "GOOGL", "JPM"]


class IntradaySignalGenerator:
    """
    Generate intraday trading signals.
    
    Focus: Quick wins with 5-min timeframes
    Signals: RSI/MACD crossovers, sentiment spikes, volume surges
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        log.info("IntradaySignalGenerator initialized")
    
    def detect_rsi_crossover(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Detect RSI crossovers (oversold/overbought).
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period
        
        Returns:
            Series with signals: 1 (buy), -1 (sell), 0 (hold)
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        signals[rsi < 30] = 1   # Oversold - buy
        signals[rsi > 70] = -1  # Overbought - sell
        
        return signals
    
    def detect_macd_crossover(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect MACD crossovers.
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            Series with signals: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        signals = pd.Series(0, index=df.index)
        
        bullish = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        signals[bullish] = 1
        
        bearish = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        signals[bearish] = -1
        
        return signals
    
    def detect_sentiment_spike(self, sentiment_data: pd.DataFrame, threshold: float = 0.3) -> pd.Series:
        """
        Detect sentiment spikes.
        
        Args:
            sentiment_data: DataFrame with 'sentiment' column
            threshold: Spike threshold
        
        Returns:
            Series with signals: 1 (positive spike), -1 (negative spike), 0 (no spike)
        """
        if sentiment_data is None or len(sentiment_data) == 0:
            return pd.Series(0, index=sentiment_data.index if sentiment_data is not None else [])
        
        sentiment_change = sentiment_data['sentiment'].diff()
        
        signals = pd.Series(0, index=sentiment_data.index)
        signals[sentiment_change > threshold] = 1   # Positive spike
        signals[sentiment_change < -threshold] = -1  # Negative spike
        
        return signals
    
    def detect_volume_surge(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """
        Detect volume surges.
        
        Args:
            df: DataFrame with 'volume' column
            threshold: Volume surge threshold (multiple of average)
        
        Returns:
            Series with signals: 1 (surge detected), 0 (no surge)
        """
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        avg_volume = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / avg_volume
        
        signals = pd.Series(0, index=df.index)
        signals[volume_ratio > threshold] = 1
        
        return signals
    
    def generate_intraday_signals(
        self,
        df: pd.DataFrame,
        sentiment_data: pd.DataFrame = None
    ) -> Dict[str, pd.Series]:
        """
        Generate all intraday signals.
        
        Args:
            df: Market data DataFrame
            sentiment_data: Sentiment data DataFrame
        
        Returns:
            Dictionary of signal series
        """
        signals = {}
        
        if "rsi" in self.config.intraday_indicators:
            signals['rsi'] = self.detect_rsi_crossover(df)
        
        if "macd" in self.config.intraday_indicators:
            signals['macd'] = self.detect_macd_crossover(df)
        
        if "sentiment_spike" in self.config.intraday_indicators and sentiment_data is not None:
            signals['sentiment_spike'] = self.detect_sentiment_spike(sentiment_data)
        
        if "volume_surge" in self.config.intraday_indicators:
            signals['volume_surge'] = self.detect_volume_surge(df)
        
        return signals
    
    def compute_intraday_score(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """
        Compute aggregate intraday score.
        
        Args:
            signals: Dictionary of signal series
        
        Returns:
            Aggregate score series
        """
        if not signals:
            return pd.Series(0)
        
        combined = pd.DataFrame(signals)
        score = combined.mean(axis=1)
        
        return score


class InterdaySignalGenerator:
    """
    Generate interday trading signals.
    
    Focus: Compounded returns with daily timeframes
    Signals: VWAP deviations, SMA trends, fundamentals, alpha factors
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        log.info("InterdaySignalGenerator initialized")
    
    def detect_vwap_deviation(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Detect VWAP deviations.
        
        Args:
            df: DataFrame with 'close', 'volume' columns
            window: Rolling window
        
        Returns:
            Series with signals: 1 (below VWAP), -1 (above VWAP), 0 (neutral)
        """
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3 if 'high' in df.columns else df['close']
        vwap = (typical_price * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        
        deviation = (df['close'] - vwap) / vwap
        
        signals = pd.Series(0, index=df.index)
        signals[deviation < -0.02] = 1   # Price below VWAP - buy
        signals[deviation > 0.02] = -1   # Price above VWAP - sell
        
        return signals
    
    def detect_sma_trend(self, df: pd.DataFrame, short: int = 50, long: int = 200) -> pd.Series:
        """
        Detect SMA trend.
        
        Args:
            df: DataFrame with 'close' column
            short: Short SMA period
            long: Long SMA period
        
        Returns:
            Series with signals: 1 (uptrend), -1 (downtrend), 0 (neutral)
        """
        sma_short = df['close'].rolling(window=short).mean()
        sma_long = df['close'].rolling(window=long).mean()
        
        signals = pd.Series(0, index=df.index)
        
        golden_cross = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
        signals[golden_cross] = 1
        
        death_cross = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
        signals[death_cross] = -1
        
        signals[sma_short > sma_long] = signals[sma_short > sma_long].replace(0, 0.5)
        signals[sma_short < sma_long] = signals[sma_short < sma_long].replace(0, -0.5)
        
        return signals
    
    def compute_alpha_factors(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute alpha factors for interday trading.
        
        Args:
            df: DataFrame with market data
        
        Returns:
            Series with alpha scores
        """
        returns = df['close'].pct_change()
        momentum = returns.rolling(window=20).mean()
        
        z_score = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        mean_reversion = -z_score  # Negative z-score = buy signal
        
        alpha_score = (momentum + mean_reversion) / 2
        
        return alpha_score
    
    def generate_interday_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate all interday signals.
        
        Args:
            df: Market data DataFrame
        
        Returns:
            Dictionary of signal series
        """
        signals = {}
        
        if "vwap_deviation" in self.config.interday_indicators:
            signals['vwap_deviation'] = self.detect_vwap_deviation(df)
        
        if "sma_trend" in self.config.interday_indicators:
            signals['sma_trend'] = self.detect_sma_trend(df)
        
        if "alpha_factors" in self.config.interday_indicators:
            signals['alpha_factors'] = self.compute_alpha_factors(df)
        
        return signals
    
    def compute_interday_score(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """
        Compute aggregate interday score.
        
        Args:
            signals: Dictionary of signal series
        
        Returns:
            Aggregate score series
        """
        if not signals:
            return pd.Series(0)
        
        combined = pd.DataFrame(signals)
        score = combined.mean(axis=1)
        
        return score


class HybridStrategyManager:
    """
    Manage hybrid trading strategies.
    
    Combines intraday and interday signals with portfolio allocation.
    """
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.intraday_generator = IntradaySignalGenerator(config)
        self.interday_generator = InterdaySignalGenerator(config)
        log.info(f"HybridStrategyManager initialized: mode={config.mode}")
    
    def classify_symbol(self, symbol: str) -> Literal["intraday", "interday"]:
        """
        Classify symbol as intraday or interday.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Classification
        """
        if symbol in self.config.intraday_symbols:
            return "intraday"
        elif symbol in self.config.interday_symbols:
            return "interday"
        else:
            return "interday"
    
    def generate_hybrid_signals(
        self,
        symbol: str,
        intraday_data: pd.DataFrame = None,
        interday_data: pd.DataFrame = None,
        sentiment_data: pd.DataFrame = None
    ) -> Dict:
        """
        Generate hybrid signals for a symbol.
        
        Args:
            symbol: Stock symbol
            intraday_data: Intraday market data
            interday_data: Interday market data
            sentiment_data: Sentiment data
        
        Returns:
            Dictionary with signals and scores
        """
        result = {
            'symbol': symbol,
            'mode': self.config.mode,
            'intraday_signals': {},
            'interday_signals': {},
            'intraday_score': 0,
            'interday_score': 0,
            'final_score': 0,
            'allocation': {}
        }
        
        classification = self.classify_symbol(symbol)
        
        if self.config.mode in ["intraday", "hybrid"] and intraday_data is not None:
            result['intraday_signals'] = self.intraday_generator.generate_intraday_signals(
                intraday_data, sentiment_data
            )
            result['intraday_score'] = self.intraday_generator.compute_intraday_score(
                result['intraday_signals']
            ).iloc[-1] if len(result['intraday_signals']) > 0 else 0
        
        if self.config.mode in ["interday", "hybrid"] and interday_data is not None:
            result['interday_signals'] = self.interday_generator.generate_interday_signals(
                interday_data
            )
            result['interday_score'] = self.interday_generator.compute_interday_score(
                result['interday_signals']
            ).iloc[-1] if len(result['interday_signals']) > 0 else 0
        
        if self.config.mode == "intraday":
            result['final_score'] = result['intraday_score']
            result['allocation'] = {'intraday': 1.0, 'interday': 0.0}
        elif self.config.mode == "interday":
            result['final_score'] = result['interday_score']
            result['allocation'] = {'intraday': 0.0, 'interday': 1.0}
        else:  # hybrid
            if classification == "intraday":
                intra_weight = 0.7
                inter_weight = 0.3
            else:
                intra_weight = 0.3
                inter_weight = 0.7
            
            result['final_score'] = (
                result['intraday_score'] * intra_weight +
                result['interday_score'] * inter_weight
            )
            result['allocation'] = {'intraday': intra_weight, 'interday': inter_weight}
        
        return result
    
    def compute_portfolio_allocation(
        self,
        signals: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Compute portfolio allocation across symbols.
        
        Args:
            signals: Dictionary of signals by symbol
        
        Returns:
            Dictionary of allocations by symbol
        """
        allocations = {}
        
        intraday_symbols = []
        interday_symbols = []
        
        for symbol, signal in signals.items():
            classification = self.classify_symbol(symbol)
            if classification == "intraday":
                intraday_symbols.append(symbol)
            else:
                interday_symbols.append(symbol)
        
        if self.config.mode == "intraday":
            for symbol in intraday_symbols:
                allocations[symbol] = 1.0 / len(intraday_symbols) if intraday_symbols else 0
        elif self.config.mode == "interday":
            for symbol in interday_symbols:
                allocations[symbol] = 1.0 / len(interday_symbols) if interday_symbols else 0
        else:  # hybrid
            intra_total = self.config.intraday_allocation
            inter_total = self.config.interday_allocation
            
            for symbol in intraday_symbols:
                allocations[symbol] = intra_total / len(intraday_symbols) if intraday_symbols else 0
            
            for symbol in interday_symbols:
                allocations[symbol] = inter_total / len(interday_symbols) if interday_symbols else 0
        
        return allocations
