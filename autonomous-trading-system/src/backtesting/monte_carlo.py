import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.utils import log

class MonteCarloSimulator:
    """
    Monte Carlo simulation for backtesting with regime shocks.
    
    Implements 1000+ paths with:
    - Random slippage variations
    - Regime shifts (bull/bear/sideways)
    - Transaction cost noise
    """
    
    def __init__(self, num_simulations: int = 1000, slippage_std: float = 0.001):
        self.num_simulations = num_simulations
        self.slippage_std = slippage_std
        log.info(f"MonteCarloSimulator initialized: {num_simulations} simulations")
    
    def simulate_paths(
        self,
        base_returns: np.ndarray,
        base_volatility: float,
        regime_probs: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Generate Monte Carlo paths with regime shocks.
        
        Args:
            base_returns: Historical returns
            base_volatility: Base volatility
            regime_probs: Probability of each regime
        
        Returns:
            Array of simulated return paths (num_simulations x len(base_returns))
        """
        if regime_probs is None:
            regime_probs = {'bull': 0.3, 'bear': 0.2, 'sideways': 0.5}
        
        n_periods = len(base_returns)
        simulated_paths = np.zeros((self.num_simulations, n_periods))
        
        for i in range(self.num_simulations):
            path = np.zeros(n_periods)
            
            for t in range(n_periods):
                regime = np.random.choice(
                    list(regime_probs.keys()),
                    p=list(regime_probs.values())
                )
                
                if regime == 'bull':
                    regime_drift = 0.0005
                    regime_vol_mult = 0.8
                elif regime == 'bear':
                    regime_drift = -0.001
                    regime_vol_mult = 1.5
                else:
                    regime_drift = 0.0
                    regime_vol_mult = 1.0
                
                base_return = base_returns[t] if t < len(base_returns) else 0
                
                noise = np.random.normal(0, base_volatility * regime_vol_mult)
                
                slippage_shock = np.random.normal(0, self.slippage_std)
                
                path[t] = base_return + regime_drift + noise + slippage_shock
            
            simulated_paths[i] = path
        
        return simulated_paths
    
    def compute_statistics(self, simulated_paths: np.ndarray) -> Dict:
        """
        Compute statistics from simulated paths.
        
        Args:
            simulated_paths: Array of simulated paths
        
        Returns:
            Dictionary of statistics
        """
        cumulative_returns = np.cumprod(1 + simulated_paths, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        sharpe_ratios = []
        max_drawdowns = []
        
        for path in simulated_paths:
            returns_series = pd.Series(path)
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
            sharpe_ratios.append(sharpe)
            
            cumulative = (1 + returns_series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns.append(drawdown.min())
        
        sharpe_ratios = np.array(sharpe_ratios)
        max_drawdowns = np.array(max_drawdowns)
        
        stats = {
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_95': np.percentile(final_returns, 95),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.min(max_drawdowns),
            'failure_rate': np.sum(final_returns < -0.1) / len(final_returns)
        }
        
        return stats
    
    def run_monte_carlo(
        self,
        historical_returns: pd.Series,
        regime_probs: Dict[str, float] = None
    ) -> Dict:
        """
        Run full Monte Carlo simulation.
        
        Args:
            historical_returns: Historical return series
            regime_probs: Regime probabilities
        
        Returns:
            Simulation results
        """
        log.info(f"Running Monte Carlo simulation with {self.num_simulations} paths")
        
        base_returns = historical_returns.values
        base_volatility = historical_returns.std()
        
        simulated_paths = self.simulate_paths(base_returns, base_volatility, regime_probs)
        
        stats = self.compute_statistics(simulated_paths)
        
        log.info(f"Monte Carlo results:")
        log.info(f"  Mean return: {stats['mean_return']:.2%}")
        log.info(f"  5th percentile: {stats['percentile_5']:.2%}")
        log.info(f"  95th percentile: {stats['percentile_95']:.2%}")
        log.info(f"  Mean Sharpe: {stats['mean_sharpe']:.2f}")
        log.info(f"  Failure rate: {stats['failure_rate']:.2%}")
        
        return {
            'statistics': stats,
            'simulated_paths': simulated_paths,
            'num_simulations': self.num_simulations
        }

class RegimeDetector:
    """
    Hidden Markov Model for regime detection.
    
    Detects bull/bear/sideways regimes using HMM.
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        log.info(f"RegimeDetector initialized with {n_regimes} regimes")
    
    def fit(self, returns: pd.Series) -> 'RegimeDetector':
        """
        Fit HMM to return data.
        
        Args:
            returns: Return series
        
        Returns:
            Self
        """
        try:
            from hmmlearn import hmm
            
            returns_array = returns.values.reshape(-1, 1)
            
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            self.model.fit(returns_array)
            
            log.info("HMM regime detection model fitted")
            
            return self
            
        except ImportError:
            log.error("hmmlearn not installed. Install with: pip install hmmlearn")
            return self
        except Exception as e:
            log.error(f"HMM fitting failed: {str(e)}")
            return self
    
    def predict_regimes(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regimes for return series.
        
        Args:
            returns: Return series
        
        Returns:
            Array of regime labels
        """
        if self.model is None:
            log.warning("Model not fitted, returning default regimes")
            return np.zeros(len(returns), dtype=int)
        
        try:
            returns_array = returns.values.reshape(-1, 1)
            regimes = self.model.predict(returns_array)
            
            means = [self.model.means_[i][0] for i in range(self.n_regimes)]
            regime_mapping = np.argsort(means)
            
            regime_labels = np.array(['bear', 'sideways', 'bull'])[regime_mapping[regimes]]
            
            return regime_labels
            
        except Exception as e:
            log.error(f"Regime prediction failed: {str(e)}")
            return np.array(['sideways'] * len(returns))
    
    def get_regime_statistics(self, returns: pd.Series, regimes: np.ndarray) -> Dict:
        """
        Compute statistics for each regime.
        
        Args:
            returns: Return series
            regimes: Regime labels
        
        Returns:
            Dictionary of regime statistics
        """
        regime_stats = {}
        
        for regime in ['bull', 'bear', 'sideways']:
            mask = regimes == regime
            if mask.sum() > 0:
                regime_returns = returns[mask]
                regime_stats[regime] = {
                    'count': mask.sum(),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
                }
            else:
                regime_stats[regime] = {
                    'count': 0,
                    'mean_return': 0,
                    'volatility': 0,
                    'sharpe': 0
                }
        
        return regime_stats
