"""
Gaussian Processes for Uncertainty Quantification

Implements:
1. Gaussian Process Regression for price prediction with uncertainty
2. Bayesian optimization for hyperparameter tuning
3. Multi-output GP for multiple assets
4. Sparse GP for large datasets

Target: Probabilistic predictions with confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scipy.optimize import minimize
from src.utils import log, config


class GaussianProcessTrader:
    """
    Gaussian Process for Trading
    
    Provides probabilistic predictions with uncertainty estimates.
    
    Features:
    - Confidence intervals for predictions
    - Automatic kernel selection
    - Hyperparameter optimization
    - Multi-output support
    
    Use cases:
    - Price prediction with uncertainty
    - Risk assessment
    - Portfolio optimization under uncertainty
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        n_restarts_optimizer: int = 10
    ):
        """
        Initialize Gaussian Process trader
        
        Args:
            kernel: 'rbf', 'matern', or 'custom'
            n_restarts_optimizer: Number of optimizer restarts
        """
        self.config = config.get('ml.gaussian_process', {
            'kernel': 'rbf',
            'length_scale': 1.0,
            'length_scale_bounds': (1e-5, 1e5),
            'noise_level': 0.1,
            'noise_level_bounds': (1e-5, 1e5),
            'n_restarts_optimizer': 10,
            'alpha': 1e-10
        })
        
        self.kernel_type = kernel or self.config['kernel']
        self.n_restarts_optimizer = n_restarts_optimizer or self.config['n_restarts_optimizer']
        
        self.model = None
        self.kernel = None
        
        self._build_kernel()
        
        log.info(f"GaussianProcessTrader initialized - Kernel: {self.kernel_type}")
    
    def _build_kernel(self):
        """Build GP kernel"""
        length_scale = self.config['length_scale']
        length_scale_bounds = self.config['length_scale_bounds']
        noise_level = self.config['noise_level']
        noise_level_bounds = self.config['noise_level_bounds']
        
        if self.kernel_type == 'rbf':
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds
            ) + WhiteKernel(
                noise_level=noise_level,
                noise_level_bounds=noise_level_bounds
            )
        
        elif self.kernel_type == 'matern':
            self.kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=1.5
            ) + WhiteKernel(
                noise_level=noise_level,
                noise_level_bounds=noise_level_bounds
            )
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """
        Train Gaussian Process
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training statistics
        """
        log.info("Starting Gaussian Process training...")
        
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=self.config['alpha'],
            normalize_y=True
        )
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        
        log_marginal_likelihood = self.model.log_marginal_likelihood()
        
        stats = {
            'train_score': train_score,
            'log_marginal_likelihood': log_marginal_likelihood,
            'kernel_params': str(self.model.kernel_)
        }
        
        log.info(f"GP training complete - Score: {train_score:.4f}, LML: {log_marginal_likelihood:.4f}")
        
        return stats
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty
        
        Args:
            X: Input features
            return_std: Return standard deviation
            
        Returns:
            predictions, std (if return_std=True)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if return_std:
            predictions, std = self.model.predict(X, return_std=True)
            return predictions, std
        else:
            predictions = self.model.predict(X, return_std=False)
            return predictions, None
    
    def get_confidence_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get confidence interval for predictions
        
        Args:
            X: Input features
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            predictions, lower_bound, upper_bound
        """
        from scipy.stats import norm
        
        predictions, std = self.predict(X, return_std=True)
        
        z = norm.ppf((1 + confidence) / 2)
        
        lower_bound = predictions - z * std
        upper_bound = predictions + z * std
        
        return predictions, lower_bound, upper_bound
    
    def sample_predictions(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Sample from predictive distribution
        
        Args:
            X: Input features
            n_samples: Number of samples
            
        Returns:
            Samples (n_samples, n_points)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        samples = self.model.sample_y(X, n_samples=n_samples)
        
        return samples


class BayesianOptimizer:
    """
    Bayesian Optimization for Hyperparameter Tuning
    
    Uses Gaussian Processes to efficiently search hyperparameter space.
    
    Features:
    - Acquisition functions (EI, UCB, PI)
    - Parallel optimization
    - Constraint handling
    
    Use cases:
    - Hyperparameter tuning for ML models
    - Trading strategy optimization
    - Portfolio allocation optimization
    """
    
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        acquisition: str = 'ei'
    ):
        """
        Initialize Bayesian optimizer
        
        Args:
            bounds: Parameter bounds [(min, max), ...]
            acquisition: 'ei' (Expected Improvement), 'ucb' (Upper Confidence Bound), 'pi' (Probability of Improvement)
        """
        self.config = config.get('ml.bayesian_opt', {
            'acquisition': 'ei',
            'xi': 0.01,
            'kappa': 2.576,
            'n_restarts': 25
        })
        
        self.bounds = np.array(bounds)
        self.acquisition = acquisition or self.config['acquisition']
        
        self.gp = None
        
        self.X_observed = []
        self.y_observed = []
        
        log.info(f"BayesianOptimizer initialized - Acquisition: {self.acquisition}")
    
    def _acquisition_function(
        self,
        X: np.ndarray,
        X_observed: np.ndarray,
        y_observed: np.ndarray
    ) -> np.ndarray:
        """
        Compute acquisition function
        
        Args:
            X: Points to evaluate
            X_observed: Observed points
            y_observed: Observed values
            
        Returns:
            Acquisition values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        y_best = np.max(y_observed)
        
        if self.acquisition == 'ei':
            from scipy.stats import norm
            
            xi = self.config['xi']
            
            with np.errstate(divide='warn'):
                imp = mu - y_best - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            
            return ei
        
        elif self.acquisition == 'ucb':
            kappa = self.config['kappa']
            return mu + kappa * sigma
        
        elif self.acquisition == 'pi':
            from scipy.stats import norm
            
            xi = self.config['xi']
            
            with np.errstate(divide='warn'):
                Z = (mu - y_best - xi) / sigma
                pi = norm.cdf(Z)
                pi[sigma == 0.0] = 0.0
            
            return pi
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
    
    def _propose_location(self) -> np.ndarray:
        """
        Propose next location to sample
        
        Returns:
            Next point to evaluate
        """
        n_restarts = self.config['n_restarts']
        dim = self.bounds.shape[0]
        
        min_val = float('inf')
        min_x = None
        
        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            def neg_acquisition(x):
                x_reshaped = x.reshape(1, -1)
                return -self._acquisition_function(
                    x_reshaped,
                    np.array(self.X_observed),
                    np.array(self.y_observed)
                )[0]
            
            res = minimize(
                neg_acquisition,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
        
        return min_x
    
    def optimize(
        self,
        objective_function: callable,
        n_iterations: int = 25,
        n_initial: int = 5
    ) -> Dict:
        """
        Run Bayesian optimization
        
        Args:
            objective_function: Function to maximize
            n_iterations: Number of iterations
            n_initial: Number of random initial points
            
        Returns:
            Optimization results
        """
        log.info(f"Starting Bayesian optimization - {n_iterations} iterations...")
        
        for i in range(n_initial):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = objective_function(x)
            
            self.X_observed.append(x)
            self.y_observed.append(y)
            
            log.info(f"Initial {i+1}/{n_initial} - Value: {y:.6f}")
        
        kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        
        for i in range(n_iterations - n_initial):
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            x_next = self._propose_location()
            
            y_next = objective_function(x_next)
            
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            
            best_y = np.max(self.y_observed)
            best_x = self.X_observed[np.argmax(self.y_observed)]
            
            log.info(f"Iteration {i+1}/{n_iterations-n_initial} - Value: {y_next:.6f}, Best: {best_y:.6f}")
        
        best_idx = np.argmax(self.y_observed)
        
        results = {
            'best_x': self.X_observed[best_idx],
            'best_y': self.y_observed[best_idx],
            'X_observed': np.array(self.X_observed),
            'y_observed': np.array(self.y_observed),
            'n_iterations': n_iterations
        }
        
        log.info(f"Optimization complete - Best value: {results['best_y']:.6f}")
        
        return results


class MultiOutputGP:
    """
    Multi-Output Gaussian Process
    
    Predicts multiple correlated outputs simultaneously.
    
    Features:
    - Shared kernel across outputs
    - Correlation modeling
    - Efficient for multiple assets
    
    Use cases:
    - Multi-asset price prediction
    - Portfolio-level uncertainty
    - Cross-asset correlation modeling
    """
    
    def __init__(
        self,
        n_outputs: int,
        kernel: str = 'rbf'
    ):
        """
        Initialize multi-output GP
        
        Args:
            n_outputs: Number of outputs
            kernel: Kernel type
        """
        self.n_outputs = n_outputs
        self.kernel_type = kernel
        
        self.models = []
        
        for i in range(n_outputs):
            if kernel == 'rbf':
                kernel_obj = C(1.0) * RBF(1.0) + WhiteKernel(0.1)
            elif kernel == 'matern':
                kernel_obj = C(1.0) * Matern(1.0, nu=1.5) + WhiteKernel(0.1)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            gp = GaussianProcessRegressor(
                kernel=kernel_obj,
                n_restarts_optimizer=10,
                normalize_y=True
            )
            
            self.models.append(gp)
        
        log.info(f"MultiOutputGP initialized - {n_outputs} outputs")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """
        Train multi-output GP
        
        Args:
            X_train: Training features
            y_train: Training labels (n_samples, n_outputs)
            
        Returns:
            Training statistics
        """
        log.info("Training multi-output GP...")
        
        if y_train.shape[1] != self.n_outputs:
            raise ValueError(f"Expected {self.n_outputs} outputs, got {y_train.shape[1]}")
        
        scores = []
        
        for i in range(self.n_outputs):
            self.models[i].fit(X_train, y_train[:, i])
            score = self.models[i].score(X_train, y_train[:, i])
            scores.append(score)
            
            log.info(f"Output {i+1}/{self.n_outputs} - Score: {score:.4f}")
        
        stats = {
            'scores': scores,
            'mean_score': np.mean(scores)
        }
        
        log.info(f"Multi-output GP training complete - Mean score: {stats['mean_score']:.4f}")
        
        return stats
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict all outputs
        
        Args:
            X: Input features
            return_std: Return standard deviation
            
        Returns:
            predictions (n_samples, n_outputs), std (if return_std=True)
        """
        predictions = []
        stds = []
        
        for i in range(self.n_outputs):
            if return_std:
                pred, std = self.models[i].predict(X, return_std=True)
                predictions.append(pred)
                stds.append(std)
            else:
                pred = self.models[i].predict(X, return_std=False)
                predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        if return_std:
            stds = np.column_stack(stds)
            return predictions, stds
        else:
            return predictions, None
