"""
Ensemble Methods for Trading

Implements:
1. XGBoost for gradient boosting
2. Random Forests for bagging
3. Stacking ensemble combining multiple models
4. Voting ensemble for consensus predictions

Target: Robust predictions through model diversity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from src.utils import log, config


class XGBoostTrader:
    """
    XGBoost Trading Model
    
    Uses gradient boosting for:
    - Price direction prediction (classification)
    - Price level prediction (regression)
    
    Features:
    - GPU acceleration
    - Hyperparameter tuning
    - Feature importance analysis
    - Early stopping
    """
    
    def __init__(
        self,
        task: str = 'classification',
        use_gpu: bool = True
    ):
        """
        Initialize XGBoost trader
        
        Args:
            task: 'classification' or 'regression'
            use_gpu: Use GPU if available
        """
        self.config = config.get('ml.xgboost', {
            'task': 'classification',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'use_gpu': True
        })
        
        self.task = task or self.config['task']
        self.use_gpu = use_gpu and self.config['use_gpu']
        
        self.model = None
        self.feature_importance = None
        
        log.info(f"XGBoostTrader initialized - Task: {self.task}, GPU: {self.use_gpu}")
    
    def build_model(self):
        """Build XGBoost model"""
        params = {
            'n_estimators': self.config['n_estimators'],
            'max_depth': self.config['max_depth'],
            'learning_rate': self.config['learning_rate'],
            'subsample': self.config['subsample'],
            'colsample_bytree': self.config['colsample_bytree'],
            'gamma': self.config['gamma'],
            'reg_alpha': self.config['reg_alpha'],
            'reg_lambda': self.config['reg_lambda'],
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'random_state': 42
        }
        
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)
        
        log.info("Built XGBoost model")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training statistics
        """
        if self.model is None:
            self.build_model()
        
        log.info("Starting XGBoost training...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.feature_importance = self.model.feature_importances_
        
        train_score = self.model.score(X_train, y_train)
        
        stats = {
            'train_score': train_score,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        }
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            stats['val_score'] = val_score
        
        log.info(f"XGBoost training complete - Train Score: {train_score:.4f}")
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class RandomForestTrader:
    """
    Random Forest Trading Model
    
    Uses bagging ensemble for:
    - Price direction prediction (classification)
    - Price level prediction (regression)
    
    Features:
    - Parallel training
    - Feature importance
    - Out-of-bag error estimation
    """
    
    def __init__(
        self,
        task: str = 'classification',
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest trader
        
        Args:
            task: 'classification' or 'regression'
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.config = config.get('ml.random_forest', {
            'task': 'classification',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1
        })
        
        self.task = task or self.config['task']
        self.n_jobs = n_jobs or self.config['n_jobs']
        
        self.model = None
        self.feature_importance = None
        
        log.info(f"RandomForestTrader initialized - Task: {self.task}")
    
    def build_model(self):
        """Build Random Forest model"""
        params = {
            'n_estimators': self.config['n_estimators'],
            'max_depth': self.config['max_depth'],
            'min_samples_split': self.config['min_samples_split'],
            'min_samples_leaf': self.config['min_samples_leaf'],
            'max_features': self.config['max_features'],
            'n_jobs': self.n_jobs,
            'random_state': 42,
            'oob_score': True
        }
        
        if self.task == 'classification':
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)
        
        log.info("Built Random Forest model")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training statistics
        """
        if self.model is None:
            self.build_model()
        
        log.info("Starting Random Forest training...")
        
        self.model.fit(X_train, y_train)
        
        self.feature_importance = self.model.feature_importances_
        
        train_score = self.model.score(X_train, y_train)
        oob_score = self.model.oob_score_
        
        stats = {
            'train_score': train_score,
            'oob_score': oob_score
        }
        
        log.info(f"Random Forest training complete - Train Score: {train_score:.4f}, OOB Score: {oob_score:.4f}")
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class EnsembleTrader:
    """
    Ensemble Trading System
    
    Combines multiple models:
    - XGBoost
    - Random Forest
    - LSTM (from time_series_models)
    - O-LGT (from olgt_model)
    
    Methods:
    - Voting: Simple majority or weighted average
    - Stacking: Meta-learner combines base models
    """
    
    def __init__(
        self,
        method: str = 'voting',
        task: str = 'classification'
    ):
        """
        Initialize ensemble trader
        
        Args:
            method: 'voting' or 'stacking'
            task: 'classification' or 'regression'
        """
        self.config = config.get('ml.ensemble', {
            'method': 'voting',
            'task': 'classification',
            'voting': 'soft',
            'weights': None
        })
        
        self.method = method or self.config['method']
        self.task = task or self.config['task']
        
        self.base_models = []
        self.ensemble_model = None
        
        log.info(f"EnsembleTrader initialized - Method: {self.method}, Task: {self.task}")
    
    def add_model(self, name: str, model):
        """Add base model to ensemble"""
        self.base_models.append((name, model))
        log.info(f"Added model to ensemble: {name}")
    
    def build_ensemble(self):
        """Build ensemble model"""
        if len(self.base_models) == 0:
            raise ValueError("No base models added. Call add_model() first.")
        
        if self.method == 'voting':
            if self.task == 'classification':
                self.ensemble_model = VotingClassifier(
                    estimators=self.base_models,
                    voting=self.config['voting'],
                    weights=self.config['weights']
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=self.base_models,
                    weights=self.config['weights']
                )
        
        elif self.method == 'stacking':
            if self.task == 'classification':
                meta_learner = LogisticRegression()
                self.ensemble_model = StackingClassifier(
                    estimators=self.base_models,
                    final_estimator=meta_learner,
                    cv=5
                )
            else:
                meta_learner = Ridge()
                self.ensemble_model = StackingRegressor(
                    estimators=self.base_models,
                    final_estimator=meta_learner,
                    cv=5
                )
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        log.info(f"Built {self.method} ensemble with {len(self.base_models)} models")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """
        Train ensemble model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training statistics
        """
        if self.ensemble_model is None:
            self.build_ensemble()
        
        log.info("Starting ensemble training...")
        
        self.ensemble_model.fit(X_train, y_train)
        
        train_score = self.ensemble_model.score(X_train, y_train)
        
        stats = {
            'train_score': train_score,
            'num_models': len(self.base_models),
            'method': self.method
        }
        
        log.info(f"Ensemble training complete - Train Score: {train_score:.4f}")
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if self.ensemble_model is None:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        return self.ensemble_model.predict_proba(X)
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict:
        """
        Cross-validate ensemble
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        if self.ensemble_model is None:
            self.build_ensemble()
        
        log.info(f"Cross-validating ensemble with {cv} folds...")
        
        scores = cross_val_score(self.ensemble_model, X, y, cv=cv)
        
        stats = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores.tolist()
        }
        
        log.info(f"Cross-validation complete - Mean Score: {stats['mean_score']:.4f} (+/- {stats['std_score']:.4f})")
        
        return stats


def create_default_ensemble(task: str = 'classification') -> EnsembleTrader:
    """
    Create default ensemble with XGBoost and Random Forest
    
    Args:
        task: 'classification' or 'regression'
        
    Returns:
        EnsembleTrader instance
    """
    ensemble = EnsembleTrader(method='voting', task=task)
    
    xgb_trader = XGBoostTrader(task=task)
    xgb_trader.build_model()
    ensemble.add_model('xgboost', xgb_trader.model)
    
    rf_trader = RandomForestTrader(task=task)
    rf_trader.build_model()
    ensemble.add_model('random_forest', rf_trader.model)
    
    return ensemble
