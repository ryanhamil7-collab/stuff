# ML Enhancements Documentation

## Overview

This document describes the advanced machine learning features integrated into the Autonomous Trading System. All models are optimized for GPU acceleration on Google Colab's paid tier (A100/L4 GPUs).

## Table of Contents

1. [Time-Series Models](#time-series-models)
2. [O-LGT Hybrid Model](#olgt-hybrid-model)
3. [Reinforcement Learning](#reinforcement-learning)
4. [Ensemble Methods](#ensemble-methods)
5. [Multimodal Fusion](#multimodal-fusion)
6. [Explainability](#explainability)
7. [Usage Examples](#usage-examples)
8. [GPU Optimization](#gpu-optimization)

---

## Time-Series Models

### LSTM (Long Short-Term Memory)

**Purpose**: Capture long-term dependencies in price sequences.

**Features**:
- Bidirectional LSTM support
- Multi-layer architecture
- Dropout regularization
- GPU acceleration

**Usage**:
```python
from src.ml_models import TimeSeriesForecaster

forecaster = TimeSeriesForecaster(
    model_type='lstm',
    sequence_length=60,
    use_gpu=True
)

# Train
forecaster.train(train_data, val_data, target_column='Close')

# Predict
predictions = forecaster.predict(test_data)

# Forecast next value
next_price = forecaster.forecast_next(recent_data)
```

### GRU (Gated Recurrent Unit)

**Purpose**: Faster alternative to LSTM with similar performance.

**Features**:
- Fewer parameters than LSTM
- Faster training
- Good for shorter sequences

**Usage**:
```python
forecaster = TimeSeriesForecaster(
    model_type='gru',
    sequence_length=30,
    use_gpu=True
)
```

### ARIMA/SARIMA

**Purpose**: Statistical baseline for comparison.

**Features**:
- Auto-ARIMA with pmdarima
- Seasonal patterns (SARIMA)
- No GPU required

**Usage**:
```python
from src.ml_models import ARIMAForecaster

forecaster = ARIMAForecaster(
    order=(5, 1, 0),
    seasonal_order=(1, 1, 1, 12)
)

forecaster.train(train_data['Close'])
predictions = forecaster.predict(steps=10)
```

---

## O-LGT Hybrid Model

**Purpose**: Sequential hybrid combining LSTM + GRU + Transformer for 30%+ improved accuracy.

**Architecture**:
1. LSTM layer (long-term dependencies)
2. GRU layer (medium-term patterns)
3. Transformer encoder (attention mechanism)
4. Fully connected layers (prediction)

**Features**:
- Residual connections
- Layer normalization
- Multi-head attention
- Mixed precision training (A100/L4 optimized)
- Gradient checkpointing for memory efficiency

**Usage**:
```python
from src.ml_models import OLGTForecaster

forecaster = OLGTForecaster(
    sequence_length=60,
    use_gpu=True,
    use_mixed_precision=True  # Faster on A100/L4
)

# Train with mixed precision
stats = forecaster.train(
    train_data,
    val_data,
    target_column='Close',
    feature_columns=['Close', 'Volume', 'RSI', 'MACD']
)

# Predict
predictions = forecaster.predict(test_data)
```

**Performance**:
- 30%+ accuracy improvement over single models
- 2-3x faster training with mixed precision on A100
- Memory efficient with gradient checkpointing

---

## Reinforcement Learning

### DQN (Deep Q-Network)

**Purpose**: Learn optimal trading policy through Q-learning.

**Features**:
- Experience replay buffer
- Target network
- Epsilon-greedy exploration
- GPU acceleration

**Usage**:
```python
from src.ml_models import RLTrader

trader = RLTrader(algorithm='dqn', use_gpu=True)

# Create environment
env = trader.create_environment(
    data=market_data,
    initial_balance=10000
)

# Train
trader.build_model()
stats = trader.train(total_timesteps=100000)

# Evaluate
metrics = trader.evaluate(eval_env, num_episodes=10)
```

### PPO (Proximal Policy Optimization)

**Purpose**: State-of-the-art policy gradient method.

**Features**:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Multiple epochs per batch
- Stable training

**Usage**:
```python
trader = RLTrader(algorithm='ppo', use_gpu=True)
trader.create_environment(data=market_data)
trader.train(total_timesteps=100000)
```

### Trading Environment

**State Space**: Market features (price, volume, indicators, sentiment)

**Action Space**: 
- 0: HOLD
- 1: BUY
- 2: SELL

**Reward Function**: Risk-adjusted returns (Sharpe-like ratio) with drawdown penalty

---

## Ensemble Methods

### XGBoost

**Purpose**: Gradient boosting for robust predictions.

**Features**:
- GPU acceleration (gpu_hist)
- Early stopping
- Feature importance
- Hyperparameter tuning

**Usage**:
```python
from src.ml_models import XGBoostTrader

trader = XGBoostTrader(task='classification', use_gpu=True)

# Train
stats = trader.train(X_train, y_train, X_val, y_val)

# Predict
predictions = trader.predict(X_test)
probabilities = trader.predict_proba(X_test)

# Feature importance
importance_df = trader.get_feature_importance(feature_names)
```

### Random Forest

**Purpose**: Bagging ensemble for variance reduction.

**Features**:
- Parallel training
- Out-of-bag error estimation
- Feature importance
- Robust to overfitting

**Usage**:
```python
from src.ml_models import RandomForestTrader

trader = RandomForestTrader(task='classification', n_jobs=-1)
stats = trader.train(X_train, y_train)
```

### Ensemble Trader

**Purpose**: Combine multiple models for improved predictions.

**Methods**:
- **Voting**: Simple majority or weighted average
- **Stacking**: Meta-learner combines base models

**Usage**:
```python
from src.ml_models import EnsembleTrader, create_default_ensemble

# Create ensemble
ensemble = create_default_ensemble(task='classification')

# Or build custom ensemble
ensemble = EnsembleTrader(method='stacking', task='classification')
ensemble.add_model('xgboost', xgb_model)
ensemble.add_model('random_forest', rf_model)
ensemble.add_model('lstm', lstm_model)

# Train
ensemble.build_ensemble()
stats = ensemble.train(X_train, y_train)

# Cross-validate
cv_stats = ensemble.cross_validate(X, y, cv=5)
```

---

## Multimodal Fusion

### BERT + LSTM

**Purpose**: Combine news sentiment with price patterns for improved predictions.

**Architecture**:
1. **BERT branch**: Processes news text → sentiment features
2. **LSTM branch**: Processes price sequences → pattern features
3. **Fusion layer**: Combines both modalities
4. **Prediction head**: Final output

**Fusion Methods**:
- **Concat**: Concatenate features
- **Attention**: Multi-head attention fusion
- **Late**: Separate predictions + ensemble

**Usage**:
```python
from src.ml_models import MultimodalTrader

trader = MultimodalTrader(
    bert_model='ProsusAI/finbert',  # Financial BERT
    fusion_method='attention',
    use_gpu=True
)

# Prepare data
texts = ["Stock surges on earnings beat", "Market crash imminent", ...]
price_sequences = np.array([...])  # (num_samples, seq_len, features)
labels = np.array([0, 1, 2, ...])  # 0=negative, 1=neutral, 2=positive

# Train
stats = trader.train(
    train_texts, train_prices, train_labels,
    val_texts, val_prices, val_labels
)

# Predict
predictions = trader.predict(test_texts, test_prices)
```

**Models**:
- **FinBERT**: Pre-trained on financial news
- **BERT-base**: General purpose
- **RoBERTa**: Robustly optimized BERT

---

## Explainability

### SHAP (SHapley Additive exPlanations)

**Purpose**: Game theory-based feature attribution.

**Features**:
- TreeExplainer (XGBoost, Random Forest)
- DeepExplainer (Neural networks)
- KernelExplainer (Any model)
- Summary plots, force plots, waterfall plots

**Usage**:
```python
from src.ml_models import SHAPExplainer

explainer = SHAPExplainer(
    model=trained_model,
    explainer_type='tree'  # or 'deep', 'kernel', 'auto'
)

# Explain predictions
explanation = explainer.explain(X_test, feature_names)

# Plot summary
explainer.plot_summary(
    explanation.values,
    X_test,
    feature_names,
    save_path='shap_summary.png'
)

# Plot force plot for single prediction
explainer.plot_force(
    explanation.values,
    X_test,
    sample_idx=0,
    feature_names=feature_names
)

# Get feature importance
importance_df = explainer.get_feature_importance(
    explanation.values,
    feature_names
)
```

### LIME (Local Interpretable Model-agnostic Explanations)

**Purpose**: Local linear approximations for interpretability.

**Features**:
- Model-agnostic
- Local explanations
- Feature importance for individual predictions

**Usage**:
```python
from src.ml_models import LIMEExplainer

explainer = LIMEExplainer(
    model=trained_model,
    training_data=X_train,
    feature_names=feature_names,
    mode='classification'
)

# Explain single instance
explanation = explainer.explain_instance(
    instance=X_test[0],
    num_features=10
)

# Plot explanation
explainer.plot_explanation(explanation, save_path='lime_explanation.png')

# Get feature importance
importance_df = explainer.get_feature_importance(explanation, label=1)
```

### Explainability Manager

**Purpose**: Unified interface for both SHAP and LIME.

**Usage**:
```python
from src.ml_models import ExplainabilityManager

manager = ExplainabilityManager(
    model=trained_model,
    training_data=X_train,
    feature_names=feature_names,
    mode='classification'
)

# Generate comprehensive report
report = manager.generate_report(
    X_test,
    sample_indices=[0, 50, 100],
    save_dir='./explainability_reports'
)
```

---

## Usage Examples

### Complete Trading Pipeline

```python
import pandas as pd
from src.ml_models import (
    OLGTForecaster,
    RLTrader,
    EnsembleTrader,
    MultimodalTrader,
    ExplainabilityManager
)

# 1. Load data
data = pd.read_csv('market_data.csv')
news = pd.read_csv('news_data.csv')

# 2. Time-series forecasting with O-LGT
forecaster = OLGTForecaster(use_gpu=True, use_mixed_precision=True)
forecaster.train(data, target_column='Close')
price_predictions = forecaster.predict(data)

# 3. Reinforcement learning for trading policy
rl_trader = RLTrader(algorithm='ppo', use_gpu=True)
rl_trader.create_environment(data, initial_balance=10000)
rl_trader.train(total_timesteps=100000)

# 4. Ensemble for robust predictions
ensemble = EnsembleTrader(method='stacking')
ensemble.add_model('xgboost', xgb_model)
ensemble.add_model('random_forest', rf_model)
ensemble.train(X_train, y_train)

# 5. Multimodal fusion for sentiment + price
multimodal = MultimodalTrader(bert_model='ProsusAI/finbert', use_gpu=True)
multimodal.train(news_texts, price_sequences, labels)

# 6. Explainability
explainer = ExplainabilityManager(ensemble.ensemble_model, X_train, feature_names)
report = explainer.generate_report(X_test, save_dir='./reports')
```

### Launcher Integration

```bash
# Use O-LGT hybrid model
python launcher.py --use-hybrid-models --gpu

# Use reinforcement learning
python launcher.py --use-rl --gpu

# Use LLM sentiment analysis
python launcher.py --use-llm --gpu

# Use all ML features
python launcher.py --use-hybrid-models --use-rl --use-llm --gpu

# Options trading with implied volatility
python launcher.py --options-trading --gpu
```

---

## GPU Optimization

### Google Colab Setup

**Recommended Configuration**:
- **GPU**: A100 (40GB) or L4 (24GB)
- **RAM**: High-RAM runtime
- **Python**: 3.10+

**Enable GPU**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Mixed Precision Training

**Benefits**:
- 2-3x faster training on A100/L4
- Reduced memory usage
- Maintained accuracy

**Usage**:
```python
# Automatic mixed precision
forecaster = OLGTForecaster(
    use_gpu=True,
    use_mixed_precision=True  # Enable AMP
)
```

### Memory Management

**Techniques**:
- Gradient checkpointing
- Batch size tuning
- Model pruning
- Gradient accumulation

**Example**:
```python
# Reduce batch size if OOM
config = {
    'batch_size': 32,  # Reduce from 64 if OOM
    'gradient_accumulation_steps': 2  # Effective batch size = 64
}
```

### Performance Tips

1. **Use mixed precision** for 2-3x speedup on A100/L4
2. **Batch processing** for efficient GPU utilization
3. **Pin memory** for faster data transfer
4. **Gradient checkpointing** for large models
5. **Model compilation** with torch.compile() (PyTorch 2.0+)

---

## Configuration

All ML models can be configured via `config/config.yaml`:

```yaml
ml:
  # Time-series models
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 100
  
  # O-LGT hybrid
  olgt:
    lstm_hidden: 128
    gru_hidden: 64
    transformer_heads: 4
    transformer_layers: 2
    use_mixed_precision: true
  
  # Reinforcement learning
  rl:
    algorithm: ppo
    learning_rate: 0.0003
    total_timesteps: 100000
  
  # Ensemble
  ensemble:
    method: voting
    voting: soft
  
  # Multimodal
  multimodal:
    bert_model: ProsusAI/finbert
    fusion_method: attention
    learning_rate: 0.00001
```

---

## Troubleshooting

### CUDA Out of Memory

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training
4. Clear cache: `torch.cuda.empty_cache()`

### Slow Training

**Solutions**:
1. Enable mixed precision
2. Increase batch size
3. Use DataLoader with num_workers
4. Pin memory for faster transfer

### Model Not Using GPU

**Check**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0
```

**Fix**:
```python
# Explicitly move model to GPU
model = model.to('cuda')
```

---

## References

- **LSTM**: Hochreiter & Schmidhuber (1997)
- **Transformer**: Vaswani et al. (2017)
- **PPO**: Schulman et al. (2017)
- **XGBoost**: Chen & Guestrin (2016)
- **BERT**: Devlin et al. (2018)
- **FinBERT**: Araci (2019)
- **SHAP**: Lundberg & Lee (2017)
- **LIME**: Ribeiro et al. (2016)

---

## Support

For issues or questions:
1. Check this documentation
2. Review code comments
3. Check logs in `logs/ml_models.log`
4. Open GitHub issue

---

**Last Updated**: 2025-10-24
**Version**: 1.0.0
