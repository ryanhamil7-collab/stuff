"""
Technical Indicators Implementation
All 19 technical indicators using pandas-ta and custom implementations.
"""

import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("Warning: pandas_ta not installed. Some indicators may not be available.")


class TechnicalIndicators:
    """Comprehensive technical indicators calculator."""
    
    def __init__(self):
        self.indicators = {}
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, periods: list = [20, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        df = data.copy()
        for period in periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, periods: list = [12, 26]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        df = data.copy()
        for period in periods:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        df = data.copy()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        df = data.copy()
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = data.copy()
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (std * std_dev)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        return df
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        df = data.copy()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        df = data.copy()
        
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=period).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        return df
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        df = data.copy()
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        return df
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        df = data.copy()
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        return df
    
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price."""
        df = data.copy()
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        return df
    
    @staticmethod
    def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        df = data.copy()
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        return df
    
    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        df = data.copy()
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)
        return df
    
    @staticmethod
    def calculate_momentum(data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Momentum."""
        df = data.copy()
        df['Momentum'] = df['Close'] - df['Close'].shift(period)
        return df
    
    @staticmethod
    def calculate_roc(data: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Calculate Rate of Change."""
        df = data.copy()
        df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        return df
    
    @staticmethod
    def calculate_ichimoku(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud."""
        df = data.copy()
        
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichimoku_Base'] = (high_26 + low_26) / 2
        
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
        
        df['Ichimoku_Lagging'] = df['Close'].shift(-26)
        
        return df
    
    @staticmethod
    def calculate_parabolic_sar(data: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """Calculate Parabolic SAR."""
        df = data.copy()
        sar = [df['Low'].iloc[0]]
        ep = df['High'].iloc[0]
        af = af_start
        trend = 1
        
        for i in range(1, len(df)):
            sar.append(sar[-1] + af * (ep - sar[-1]))
            
            if trend == 1:
                if df['Low'].iloc[i] < sar[-1]:
                    trend = -1
                    sar[-1] = ep
                    ep = df['Low'].iloc[i]
                    af = af_start
                else:
                    if df['High'].iloc[i] > ep:
                        ep = df['High'].iloc[i]
                        af = min(af + af_start, af_max)
            else:
                if df['High'].iloc[i] > sar[-1]:
                    trend = 1
                    sar[-1] = ep
                    ep = df['High'].iloc[i]
                    af = af_start
                else:
                    if df['Low'].iloc[i] < ep:
                        ep = df['Low'].iloc[i]
                        af = min(af + af_start, af_max)
        
        df['PSAR'] = sar
        return df
    
    @staticmethod
    def calculate_supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend."""
        df = data.copy()
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        
        hl_avg = (df['High'] + df['Low']) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = [0] * len(df)
        direction = [1] * len(df)
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > upper_band.iloc[i-1]:
                direction[i] = 1
            elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band.iloc[i]
            else:
                supertrend[i] = upper_band.iloc[i]
        
        df['Supertrend'] = supertrend
        df['Supertrend_Direction'] = direction
        
        return df
    
    @staticmethod
    def calculate_keltner_channels(data: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Calculate Keltner Channels."""
        df = data.copy()
        df['KC_Middle'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        
        df['KC_Upper'] = df['KC_Middle'] + (multiplier * atr)
        df['KC_Lower'] = df['KC_Middle'] - (multiplier * atr)
        
        return df
    
    @staticmethod
    def calculate_donchian_channels(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Donchian Channels."""
        df = data.copy()
        df['DC_Upper'] = df['High'].rolling(window=period).max()
        df['DC_Lower'] = df['Low'].rolling(window=period).min()
        df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
        return df
    
    def calculate_all(self, data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Calculate all configured indicators."""
        df = data.copy()
        
        if config is None:
            config = {
                'sma': {'periods': [20, 50, 200]},
                'ema': {'periods': [12, 26]},
                'rsi': {'period': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger_bands': {'period': 20, 'std_dev': 2},
                'atr': {'period': 14},
                'adx': {'period': 14},
                'stochastic': {'k_period': 14, 'd_period': 3},
            }
        
        try:
            if 'sma' in config:
                df = self.calculate_sma(df, **config['sma'])
            if 'ema' in config:
                df = self.calculate_ema(df, **config['ema'])
            if 'rsi' in config:
                df = self.calculate_rsi(df, **config['rsi'])
            if 'macd' in config:
                df = self.calculate_macd(df, **config['macd'])
            if 'bollinger_bands' in config:
                df = self.calculate_bollinger_bands(df, **config['bollinger_bands'])
            if 'atr' in config:
                df = self.calculate_atr(df, **config['atr'])
            if 'adx' in config:
                df = self.calculate_adx(df, **config['adx'])
            if 'stochastic' in config:
                df = self.calculate_stochastic(df, **config['stochastic'])
            if 'obv' in config:
                df = self.calculate_obv(df)
            if 'vwap' in config:
                df = self.calculate_vwap(df)
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return df
