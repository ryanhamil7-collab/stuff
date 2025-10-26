import pandas as pd
import numpy as np
from typing import List, Dict, Callable
from src.utils import log, config

class AlphaMining:
    
    def __init__(self):
        self.alphas = []
        self.alpha_performance = {}
        log.info("AlphaMining initialized")
    
    def generate_alpha_formulas(self) -> List[Dict]:
        alphas = []
        
        alphas.append({
            'name': 'alpha_001_vwap_deviation',
            'formula': lambda df: (df['Close'] - df['VWAP']) / df['VWAP'],
            'description': 'Price deviation from VWAP'
        })
        
        alphas.append({
            'name': 'alpha_002_momentum',
            'formula': lambda df: df['Close'].pct_change(20),
            'description': '20-day momentum'
        })
        
        alphas.append({
            'name': 'alpha_003_mean_reversion',
            'formula': lambda df: (df['Close'] - df['SMA_20']) / df['SMA_20'] if 'SMA_20' in df.columns else 0,
            'description': 'Mean reversion from 20-day SMA'
        })
        
        alphas.append({
            'name': 'alpha_004_volume_price',
            'formula': lambda df: df['Volume'].pct_change() * df['Close'].pct_change(),
            'description': 'Volume-price correlation'
        })
        
        alphas.append({
            'name': 'alpha_005_rsi_divergence',
            'formula': lambda df: df['RSI'] - 50 if 'RSI' in df.columns else 0,
            'description': 'RSI divergence from neutral'
        })
        
        alphas.append({
            'name': 'alpha_006_bollinger_position',
            'formula': lambda df: df['BB_Position'] if 'BB_Position' in df.columns else 0.5,
            'description': 'Position within Bollinger Bands'
        })
        
        alphas.append({
            'name': 'alpha_007_macd_strength',
            'formula': lambda df: df['MACD_Hist'] / df['ATR'] if 'MACD_Hist' in df.columns and 'ATR' in df.columns else 0,
            'description': 'MACD histogram normalized by ATR'
        })
        
        alphas.append({
            'name': 'alpha_008_trend_strength',
            'formula': lambda df: (df['SMA_20'] - df['SMA_50']) / df['SMA_50'] if 'SMA_20' in df.columns and 'SMA_50' in df.columns else 0,
            'description': 'Trend strength from SMA crossover'
        })
        
        alphas.append({
            'name': 'alpha_009_volatility_adjusted_return',
            'formula': lambda df: df['Close'].pct_change(5) / df['ATR'] if 'ATR' in df.columns else 0,
            'description': '5-day return adjusted by volatility'
        })
        
        alphas.append({
            'name': 'alpha_010_volume_momentum',
            'formula': lambda df: (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std(),
            'description': 'Volume momentum z-score'
        })
        
        alphas.append({
            'name': 'alpha_011_price_acceleration',
            'formula': lambda df: df['Close'].pct_change().diff(),
            'description': 'Price acceleration (second derivative)'
        })
        
        alphas.append({
            'name': 'alpha_012_high_low_range',
            'formula': lambda df: (df['High'] - df['Low']) / df['Close'],
            'description': 'Intraday range relative to close'
        })
        
        alphas.append({
            'name': 'alpha_013_gap',
            'formula': lambda df: (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1),
            'description': 'Overnight gap'
        })
        
        alphas.append({
            'name': 'alpha_014_obv_trend',
            'formula': lambda df: df['OBV'].pct_change(10) if 'OBV' in df.columns else 0,
            'description': 'On-Balance Volume trend'
        })
        
        alphas.append({
            'name': 'alpha_015_adx_strength',
            'formula': lambda df: df['ADX'] / 100 if 'ADX' in df.columns else 0,
            'description': 'ADX trend strength normalized'
        })
        
        log.info(f"Generated {len(alphas)} alpha formulas")
        return alphas
    
    def calculate_alpha(self, df: pd.DataFrame, alpha: Dict) -> pd.Series:
        try:
            result = alpha['formula'](df)
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(df), index=df.index)
            return result.fillna(0)
        except Exception as e:
            log.error(f"Error calculating alpha {alpha['name']}: {str(e)}")
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_rank_ic(self, alpha_values: pd.Series, returns: pd.Series) -> float:
        try:
            if len(alpha_values) != len(returns):
                return 0.0
            
            valid_mask = ~(alpha_values.isna() | returns.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            alpha_valid = alpha_values[valid_mask]
            returns_valid = returns[valid_mask]
            
            if alpha_valid.std() == 0 or returns_valid.std() == 0:
                return 0.0
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                correlation = alpha_valid.corr(returns_valid, method='spearman')
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            log.error(f"Error calculating RankIC: {str(e)}")
            return 0.0
    
    def calculate_turnover(self, alpha_values: pd.Series) -> float:
        try:
            changes = alpha_values.diff().abs()
            alpha_mean = alpha_values.abs().mean()
            
            if alpha_mean == 0 or np.isnan(alpha_mean):
                return 0.0
            
            turnover = changes.mean() / alpha_mean
            return turnover if not np.isnan(turnover) else 0.0
        except Exception as e:
            log.error(f"Error calculating turnover: {str(e)}")
            return 0.0
    
    def evaluate_alpha(self, df: pd.DataFrame, alpha: Dict) -> Dict:
        alpha_values = self.calculate_alpha(df, alpha)
        
        returns = df['Close'].pct_change().shift(-1)
        
        rank_ic = self.calculate_rank_ic(alpha_values, returns)
        
        turnover = self.calculate_turnover(alpha_values)
        
        ic_series = []
        window = 20
        for i in range(window, len(alpha_values)):
            window_ic = self.calculate_rank_ic(
                alpha_values.iloc[i-window:i],
                returns.iloc[i-window:i]
            )
            ic_series.append(window_ic)
        
        ic_mean = np.mean(ic_series) if ic_series else 0.0
        ic_std = np.std(ic_series) if ic_series else 1.0
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        
        evaluation = {
            'name': alpha['name'],
            'rank_ic': rank_ic,
            'icir': icir,
            'turnover': turnover,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'description': alpha['description']
        }
        
        return evaluation
    
    def mine_alphas(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        log.info(f"Mining alphas for {len(data)} symbols")
        
        alphas = self.generate_alpha_formulas()
        
        alpha_evaluations = []
        
        for alpha in alphas:
            evaluations = []
            
            for symbol, df in data.items():
                if len(df) < 50:
                    continue
                
                eval_result = self.evaluate_alpha(df, alpha)
                evaluations.append(eval_result)
            
            if evaluations:
                avg_evaluation = {
                    'name': alpha['name'],
                    'description': alpha['description'],
                    'rank_ic': np.mean([e['rank_ic'] for e in evaluations]),
                    'icir': np.mean([e['icir'] for e in evaluations]),
                    'turnover': np.mean([e['turnover'] for e in evaluations]),
                    'ic_std': np.mean([e['ic_std'] for e in evaluations]),
                    'num_symbols': len(evaluations)
                }
                
                alpha_evaluations.append(avg_evaluation)
        
        alpha_evaluations.sort(key=lambda x: x['icir'], reverse=True)
        
        min_rankic = config.get('alpha.min_rankic', 0.02)
        max_turnover = config.get('alpha.max_turnover', 0.3)
        
        filtered_alphas = [
            alpha for alpha in alpha_evaluations
            if abs(alpha['rank_ic']) >= min_rankic and alpha['turnover'] <= max_turnover
        ]
        
        log.info(f"Mined {len(filtered_alphas)} high-quality alphas from {len(alphas)} candidates")
        
        for i, alpha in enumerate(filtered_alphas[:10]):
            log.info(f"Top Alpha #{i+1}: {alpha['name']} - RankIC: {alpha['rank_ic']:.4f}, ICIR: {alpha['icir']:.4f}, Turnover: {alpha['turnover']:.4f}")
        
        return filtered_alphas
    
    def get_alpha_signals(self, df: pd.DataFrame, top_alphas: List[Dict]) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        alphas = self.generate_alpha_formulas()
        alpha_dict = {a['name']: a for a in alphas}
        
        for alpha_eval in top_alphas[:5]:
            alpha = alpha_dict.get(alpha_eval['name'])
            if alpha:
                alpha_values = self.calculate_alpha(df, alpha)
                normalized = (alpha_values - alpha_values.mean()) / (alpha_values.std() + 1e-8)
                signals += normalized
        
        signals = signals / len(top_alphas[:5])
        
        return signals
