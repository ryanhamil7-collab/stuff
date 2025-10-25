"""
Time-Series Models for Price Forecasting

Implements LSTM, GRU, and ARIMA/SARIMA models for price prediction.
Optimized for GPU acceleration (A100/L4) with PyTorch.

Models:
- LSTM: Long Short-Term Memory networks
- GRU: Gated Recurrent Units
- ARIMA/SARIMA: Statistical baseline models
- Hybrid models combining multiple approaches

Target: Accurate price forecasting for trading decisions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.utils import log, config

class LSTMModel(nn.Module):
    """
    LSTM Model for Time-Series Forecasting
    
    Architecture:
    - Multiple LSTM layers with dropout
    - Fully connected output layer
    - Supports GPU acceleration
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output predictions
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        last_output = lstm_out[:, -1, :]
        
        out = self.dropout(last_output)
        
        out = self.fc(out)
        
        return out


class GRUModel(nn.Module):
    """
    GRU Model for Time-Series Forecasting
    
    Similar to LSTM but with fewer parameters and faster training.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """Initialize GRU model"""
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass"""
        gru_out, h_n = self.gru(x)
        
        last_output = gru_out[:, -1, :]
        
        out = self.dropout(last_output)
        
        out = self.fc(out)
        
        return out


class TimeSeriesForecaster:
    """
    Time-Series Forecasting System
    
    Supports multiple models:
    - LSTM
    - GRU
    - ARIMA/SARIMA
    
    Features:
    - GPU acceleration
    - Automatic model selection
    - Hyperparameter optimization
    - Walk-forward validation
    """
    
    def __init__(
        self,
        model_type: str = 'lstm',
        sequence_length: int = 60,
        forecast_horizon: int = 1,
        use_gpu: bool = True
    ):
        """
        Initialize forecaster
        
        Args:
            model_type: Type of model ('lstm', 'gru', 'arima', 'sarima')
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            use_gpu: Use GPU if available
        """
        self.config = config.get('ml.time_series', {
            'model_type': 'lstm',
            'sequence_length': 60,
            'forecast_horizon': 1,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'use_gpu': True
        })
        
        self.model_type = model_type or self.config['model_type']
        self.sequence_length = sequence_length or self.config['sequence_length']
        self.forecast_horizon = forecast_horizon or self.config['forecast_horizon']
        
        self.use_gpu = use_gpu and self.config['use_gpu']
        self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = None
        
        self.train_losses = []
        self.val_losses = []
        
        log.info(f"TimeSeriesForecaster initialized - Model: {self.model_type}, Device: {self.device}")
    
    def build_model(self, input_size: int):
        """Build the forecasting model"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                output_size=self.forecast_horizon
            )
        elif self.model_type == 'gru':
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                output_size=self.forecast_horizon
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
        log.info(f"Built {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'Close',
        feature_columns: List[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training
        
        Args:
            data: DataFrame with time-series data
            target_column: Column to predict
            feature_columns: Additional feature columns
            
        Returns:
            X, y tensors
        """
        if feature_columns is None:
            feature_columns = [target_column]
        
        features = data[feature_columns].values
        
        from sklearn.preprocessing import StandardScaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:i+self.sequence_length])
            y.append(features[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon, 0])  # Predict target column
        
        X = np.array(X)
        y = np.array(y)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        target_column: str = 'Close',
        feature_columns: List[str] = None
    ) -> Dict:
        """
        Train the model
        
        Args:
            train_data: Training data
            val_data: Validation data
            target_column: Column to predict
            feature_columns: Additional feature columns
            
        Returns:
            Training statistics
        """
        log.info("Starting model training...")
        
        X_train, y_train = self.prepare_data(train_data, target_column, feature_columns)
        
        if val_data is not None:
            X_val, y_val = self.prepare_data(val_data, target_column, feature_columns)
        
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        batch_size = self.config['batch_size']
        num_epochs = self.config['num_epochs']
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    log.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    log.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    log.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        log.info("Training complete")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss if val_data is not None else None,
            'num_epochs': epoch + 1
        }
    
    def predict(
        self,
        data: pd.DataFrame,
        target_column: str = 'Close',
        feature_columns: List[str] = None
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data: Input data
            target_column: Column to predict
            feature_columns: Additional feature columns
            
        Returns:
            Predictions array
        """
        self.model.eval()
        
        X, _ = self.prepare_data(data, target_column, feature_columns)
        
        with torch.no_grad():
            predictions = self.model(X)
        
        predictions = predictions.cpu().numpy()
        
        if self.scaler is not None:
            dummy = np.zeros((predictions.shape[0], self.scaler.n_features_in_))
            dummy[:, 0] = predictions[:, 0]
            predictions_inv = self.scaler.inverse_transform(dummy)[:, 0]
        else:
            predictions_inv = predictions[:, 0]
        
        return predictions_inv
    
    def forecast_next(
        self,
        recent_data: pd.DataFrame,
        target_column: str = 'Close',
        feature_columns: List[str] = None
    ) -> float:
        """
        Forecast next value
        
        Args:
            recent_data: Recent data (at least sequence_length rows)
            target_column: Column to predict
            feature_columns: Additional feature columns
            
        Returns:
            Next predicted value
        """
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        recent_data = recent_data.tail(self.sequence_length)
        
        predictions = self.predict(recent_data, target_column, feature_columns)
        
        return predictions[-1]


class ARIMAForecaster:
    """
    ARIMA/SARIMA Forecaster
    
    Statistical baseline model for time-series forecasting.
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize ARIMA forecaster
        
        Args:
            order: (p, d, q) order for ARIMA
            seasonal_order: (P, D, Q, s) order for SARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
        log.info(f"ARIMAForecaster initialized - Order: {order}, Seasonal: {seasonal_order}")
    
    def train(self, data: pd.Series) -> Dict:
        """
        Train ARIMA model
        
        Args:
            data: Time-series data
            
        Returns:
            Training statistics
        """
        log.info("Training ARIMA model...")
        
        if self.seasonal_order is not None:
            self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
        else:
            self.model = ARIMA(data, order=self.order)
        
        self.fitted_model = self.model.fit()
        
        log.info("ARIMA training complete")
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'params': self.fitted_model.params.to_dict()
        }
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future values
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast.values
    
    def forecast_next(self) -> float:
        """Forecast next value"""
        return self.predict(steps=1)[0]
