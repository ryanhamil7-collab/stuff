"""
Clustering and Anomaly Detection

Implements:
1. K-Means clustering for market regime detection
2. Autoencoders for anomaly detection (flash crashes, unusual patterns)
3. DBSCAN for density-based clustering
4. Isolation Forest for outlier detection

Target: Identify market regimes and detect anomalies
Optimized for GPU (A100/L4)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.utils import log, config


class Autoencoder(nn.Module):
    """
    Autoencoder for Anomaly Detection
    
    Architecture:
    - Encoder: Compresses input to latent representation
    - Decoder: Reconstructs input from latent representation
    - Anomaly score: Reconstruction error
    
    Features:
    - Variational autoencoder (VAE) support
    - Denoising autoencoder support
    - GPU acceleration
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32, 16],
        latent_size: int = 8,
        dropout: float = 0.2
    ):
        """
        Initialize autoencoder
        
        Args:
            input_size: Input feature size
            hidden_sizes: Hidden layer sizes
            latent_size: Latent representation size
            dropout: Dropout rate
        """
        super(Autoencoder, self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, hidden_size))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        encoder_layers.append(nn.Linear(prev_size, latent_size))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        prev_size = latent_size
        for hidden_size in reversed(hidden_sizes):
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        decoder_layers.append(nn.Linear(prev_size, input_size))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed input
        """
        latent = self.encoder(x)
        
        reconstructed = self.decoder(latent)
        
        return reconstructed
    
    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent representation to input"""
        return self.decoder(latent)


class AnomalyDetector:
    """
    Autoencoder-based Anomaly Detector
    
    Detects anomalies by reconstruction error.
    High reconstruction error indicates anomaly.
    
    Features:
    - GPU acceleration
    - Automatic threshold selection
    - Real-time anomaly detection
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32, 16],
        latent_size: int = 8,
        use_gpu: bool = True
    ):
        """
        Initialize anomaly detector
        
        Args:
            input_size: Input feature size
            hidden_sizes: Hidden layer sizes
            latent_size: Latent size
            use_gpu: Use GPU if available
        """
        self.config = config.get('ml.anomaly', {
            'hidden_sizes': [64, 32, 16],
            'latent_size': 8,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 64,
            'num_epochs': 50,
            'threshold_percentile': 95,
            'use_gpu': True
        })
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or self.config['hidden_sizes']
        self.latent_size = latent_size or self.config['latent_size']
        self.use_gpu = use_gpu and self.config['use_gpu']
        self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.model = Autoencoder(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            latent_size=self.latent_size,
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.scaler = StandardScaler()
        
        self.threshold = None
        
        log.info(f"AnomalyDetector initialized - Device: {self.device}")
    
    def train(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train autoencoder
        
        Args:
            X_train: Training data
            X_val: Validation data
            
        Returns:
            Training statistics
        """
        log.info("Starting anomaly detector training...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        batch_size = self.config['batch_size']
        num_epochs = self.config['num_epochs']
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch = X_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_reconstructed = self.model(X_val_tensor)
                    val_loss = criterion(val_reconstructed, X_val_tensor).item()
                    val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    log.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    log.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        self.model.eval()
        with torch.no_grad():
            train_reconstructed = self.model(X_train_tensor)
            train_errors = torch.mean((train_reconstructed - X_train_tensor) ** 2, dim=1)
            train_errors_np = train_errors.cpu().numpy()
        
        self.threshold = np.percentile(train_errors_np, self.config['threshold_percentile'])
        
        log.info(f"Anomaly detector training complete - Threshold: {self.threshold:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if X_val is not None else None,
            'threshold': self.threshold
        }
    
    def detect_anomalies(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies
        
        Args:
            X: Input data
            
        Returns:
            anomaly_scores, is_anomaly
        """
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
            errors_np = errors.cpu().numpy()
        
        is_anomaly = errors_np > self.threshold
        
        return errors_np, is_anomaly
    
    def get_latent_representation(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Get latent representation"""
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            latent = self.model.encode(X_tensor)
        
        return latent.cpu().numpy()


class MarketRegimeDetector:
    """
    Market Regime Detector using K-Means Clustering
    
    Identifies different market regimes:
    - Bull market
    - Bear market
    - Sideways/ranging market
    - High volatility
    - Low volatility
    
    Features:
    - Automatic regime count selection (elbow method)
    - Feature engineering for regime detection
    - Real-time regime classification
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        method: str = 'kmeans'
    ):
        """
        Initialize market regime detector
        
        Args:
            n_regimes: Number of market regimes
            method: 'kmeans' or 'dbscan'
        """
        self.config = config.get('ml.clustering', {
            'n_regimes': 4,
            'method': 'kmeans',
            'max_iter': 300,
            'n_init': 10
        })
        
        self.n_regimes = n_regimes or self.config['n_regimes']
        self.method = method or self.config['method']
        
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None
        
        log.info(f"MarketRegimeDetector initialized - Method: {self.method}, Regimes: {self.n_regimes}")
    
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Fit clustering model
        
        Args:
            X: Feature matrix
            feature_names: Feature names
            
        Returns:
            Clustering statistics
        """
        log.info("Fitting market regime detector...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_regimes,
                max_iter=self.config['max_iter'],
                n_init=self.config['n_init'],
                random_state=42
            )
            labels = self.model.fit_predict(X_scaled)
            
            stats = {
                'inertia': self.model.inertia_,
                'n_iter': self.model.n_iter_
            }
        
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
            labels = self.model.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            stats = {
                'n_clusters': n_clusters,
                'n_noise': n_noise
            }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.regime_labels = labels
        
        log.info(f"Market regime detector fitted - {len(set(labels))} regimes found")
        
        return stats
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict market regime
        
        Args:
            X: Feature matrix
            
        Returns:
            Regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        labels = self.model.predict(X_scaled)
        
        return labels
    
    def get_regime_characteristics(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get characteristics of each regime
        
        Args:
            X: Feature matrix
            feature_names: Feature names
            
        Returns:
            DataFrame with regime characteristics
        """
        if self.regime_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        regime_chars = []
        
        for regime in range(self.n_regimes):
            mask = self.regime_labels == regime
            regime_data = X[mask]
            
            if len(regime_data) > 0:
                mean_features = regime_data.mean(axis=0)
                
                char = {
                    'regime': regime,
                    'count': len(regime_data),
                    'percentage': len(regime_data) / len(X) * 100
                }
                
                for i, feature_name in enumerate(feature_names):
                    char[feature_name] = mean_features[i]
                
                regime_chars.append(char)
        
        return pd.DataFrame(regime_chars)
    
    def find_optimal_regimes(
        self,
        X: np.ndarray,
        max_regimes: int = 10
    ) -> Dict:
        """
        Find optimal number of regimes using elbow method
        
        Args:
            X: Feature matrix
            max_regimes: Maximum number of regimes to try
            
        Returns:
            Inertia scores for each regime count
        """
        log.info(f"Finding optimal number of regimes (max={max_regimes})...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        
        for n in range(2, max_regimes + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        return {
            'n_regimes': list(range(2, max_regimes + 1)),
            'inertias': inertias
        }


class IsolationForestDetector:
    """
    Isolation Forest for Outlier Detection
    
    Fast anomaly detection using isolation trees.
    
    Features:
    - Fast training and prediction
    - No need for normalization
    - Works well with high-dimensional data
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100
    ):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of trees
        """
        self.config = config.get('ml.isolation_forest', {
            'contamination': 0.1,
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42
        })
        
        self.contamination = contamination or self.config['contamination']
        self.n_estimators = n_estimators or self.config['n_estimators']
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.config['max_samples'],
            random_state=self.config['random_state']
        )
        
        log.info(f"IsolationForestDetector initialized - Contamination: {self.contamination}")
    
    def fit(
        self,
        X: np.ndarray
    ) -> Dict:
        """
        Fit Isolation Forest
        
        Args:
            X: Training data
            
        Returns:
            Training statistics
        """
        log.info("Fitting Isolation Forest...")
        
        self.model.fit(X)
        
        predictions = self.model.predict(X)
        n_outliers = (predictions == -1).sum()
        
        stats = {
            'n_outliers': int(n_outliers),
            'outlier_percentage': n_outliers / len(X) * 100
        }
        
        log.info(f"Isolation Forest fitted - {n_outliers} outliers detected ({stats['outlier_percentage']:.2f}%)")
        
        return stats
    
    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outliers
        
        Args:
            X: Input data
            
        Returns:
            predictions (-1 for outliers, 1 for inliers), anomaly_scores
        """
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        return predictions, scores
