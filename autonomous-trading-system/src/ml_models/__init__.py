"""
ML Models Package

Comprehensive machine learning models for autonomous trading:
- Time-series forecasting (LSTM, GRU, ARIMA, O-LGT hybrid)
- Reinforcement learning (DQN, PPO)
- Ensemble methods (XGBoost, Random Forest, Voting, Stacking)
- Multimodal fusion (BERT + LSTM)
- Explainability (SHAP, LIME)

All models optimized for GPU acceleration (A100/L4).
"""

from .time_series_models import LSTMModel, GRUModel, TimeSeriesForecaster, ARIMAForecaster
from .olgt_model import OLGTModel, OLGTForecaster
from .rl_trading import TradingEnvironment, RLTrader
from .ensemble_methods import XGBoostTrader, RandomForestTrader, EnsembleTrader, create_default_ensemble
from .multimodal_fusion import BERTSentimentAnalyzer, MultimodalFusionModel, MultimodalTrader
from .explainability import SHAPExplainer, LIMEExplainer, ExplainabilityManager

__all__ = [
    'LSTMModel',
    'GRUModel',
    'TimeSeriesForecaster',
    'ARIMAForecaster',
    
    'OLGTModel',
    'OLGTForecaster',
    
    'TradingEnvironment',
    'RLTrader',
    
    'XGBoostTrader',
    'RandomForestTrader',
    'EnsembleTrader',
    'create_default_ensemble',
    
    'BERTSentimentAnalyzer',
    'MultimodalFusionModel',
    'MultimodalTrader',
    
    'SHAPExplainer',
    'LIMEExplainer',
    'ExplainabilityManager',
]
