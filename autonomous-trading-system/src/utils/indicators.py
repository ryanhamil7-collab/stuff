import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

try:
    import pandas_ta as ta
except ImportError as e:
    import warnings
    warnings.warn(
        f"pandas_ta import failed: {e}. "
        "Technical indicators will still work using manual implementations. "
        "For full pandas_ta features, install: pip install pandas-ta==0.3.14b0 (Python 3.10) "
        "or pip install pandas-ta-openbb==0.4.22 (fallback)"
    )
    ta = None

class TechnicalIndicators:
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        df = data.copy()
        for period in periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        df = data.copy()
        for period in periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = data.copy()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        df = data.copy()
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        df = data.copy()
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        bb_std = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        return df
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = data.copy()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return df
    
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        return df
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        df = data.copy()
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_K'] = df['Stoch_K'].rolling(window=smooth_k).mean()
        df['Stoch_D'] = df['Stoch_K'].rolling(window=smooth_d).mean()
        return df
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = data.copy()
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.calculate_atr(df, period)['ATR']
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.ewm(alpha=1/period).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        return df
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        df = data.copy()
        
        df = TechnicalIndicators.calculate_sma(df, config.get('sma_periods', [20, 50, 200]))
        df = TechnicalIndicators.calculate_ema(df, config.get('ema_periods', [12, 26]))
        df = TechnicalIndicators.calculate_rsi(df, config.get('rsi_period', 14))
        
        macd_config = config.get('macd', {})
        df = TechnicalIndicators.calculate_macd(
            df, 
            macd_config.get('fast', 12),
            macd_config.get('slow', 26),
            macd_config.get('signal', 9)
        )
        
        bb_config = config.get('bollinger', {})
        df = TechnicalIndicators.calculate_bollinger_bands(
            df,
            bb_config.get('period', 20),
            bb_config.get('std', 2)
        )
        
        df = TechnicalIndicators.calculate_atr(df, config.get('atr_period', 14))
        df = TechnicalIndicators.calculate_obv(df)
        df = TechnicalIndicators.calculate_vwap(df)
        df = TechnicalIndicators.calculate_stochastic(df)
        df = TechnicalIndicators.calculate_adx(df)
        
        return df
    
    @staticmethod
    def detect_market_regime(data: pd.DataFrame) -> str:
        if len(data) < 50:
            return "unknown"
        
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        volatility = data['Close'].pct_change().std()
        
        if current_price > sma_20 > sma_50:
            return "bull"
        elif current_price < sma_20 < sma_50:
            return "bear"
        else:
            return "sideways"
    
    @staticmethod
    def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['Signal'] = 0
        
        if 'RSI' in df.columns:
            df.loc[df['RSI'] < 30, 'Signal'] += 1
            df.loc[df['RSI'] > 70, 'Signal'] -= 1
        
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] += 1
            df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] -= 1
        
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] += 1
            df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] -= 1
        
        if 'BB_Position' in df.columns:
            df.loc[df['BB_Position'] < 0.2, 'Signal'] += 1
            df.loc[df['BB_Position'] > 0.8, 'Signal'] -= 1
        
        df['Signal'] = df['Signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        return df
