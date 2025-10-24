"""
Adversarial Robustness Training - Feature 13

Trains models to be robust against adversarial attacks and market manipulation:
1. Generate adversarial examples (FGSM, PGD)
2. Train on perturbed data
3. Detect market manipulation patterns
4. Robust decision-making under uncertainty

Target: +15-20% robustness against adversarial scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from src.utils import log, config

class AdversarialTrainer:
    """
    Adversarial Robustness Training System
    
    Implements adversarial training to make models robust against:
    - Market manipulation (pump and dump, spoofing)
    - Data poisoning attacks
    - Adversarial perturbations in features
    - Black swan events
    
    Methods:
    - FGSM (Fast Gradient Sign Method)
    - PGD (Projected Gradient Descent)
    - Adversarial data augmentation
    - Robust loss functions
    """
    
    def __init__(self, model: nn.Module = None):
        """
        Initialize adversarial trainer
        
        Args:
            model: PyTorch model to train
        """
        self.config = config.get('training.adversarial', {
            'enabled': True,
            'attack_types': ['fgsm', 'pgd', 'manipulation'],
            'epsilon': 0.1,  # Perturbation magnitude
            'alpha': 0.01,   # Step size for PGD
            'num_steps': 10, # PGD iterations
            'augmentation_ratio': 0.3,  # 30% adversarial examples
            'robust_loss_weight': 0.5
        })
        
        self.model = model
        self.epsilon = self.config['epsilon']
        self.alpha = self.config['alpha']
        self.num_steps = self.config['num_steps']
        
        self.attack_stats = {
            'fgsm_generated': 0,
            'pgd_generated': 0,
            'manipulation_detected': 0,
            'total_adversarial_examples': 0
        }
        
        log.info("AdversarialTrainer initialized")
        log.info(f"Attack types: {self.config['attack_types']}")
        log.info(f"Epsilon: {self.epsilon}, Alpha: {self.alpha}, Steps: {self.num_steps}")
    
    def generate_adversarial_examples(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        attack_type: str = 'fgsm'
    ) -> torch.Tensor:
        """
        Generate adversarial examples
        
        Args:
            X: Input features (batch_size, num_features)
            y: True labels (batch_size,)
            attack_type: Type of attack ('fgsm', 'pgd', 'manipulation')
            
        Returns:
            Adversarial examples
        """
        if attack_type == 'fgsm':
            return self._fgsm_attack(X, y)
        elif attack_type == 'pgd':
            return self._pgd_attack(X, y)
        elif attack_type == 'manipulation':
            return self._market_manipulation_attack(X)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
    
    def _fgsm_attack(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Generates adversarial examples by adding perturbations in the direction
        of the gradient that maximizes the loss.
        """
        X_adv = X.clone().detach().requires_grad_(True)
        
        if self.model is None:
            log.warning("No model provided, returning perturbed data")
            return X + self.epsilon * torch.sign(torch.randn_like(X))
        
        outputs = self.model(X_adv)
        
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            loss = F.cross_entropy(outputs, y)
        else:
            loss = F.mse_loss(outputs.squeeze(), y.float())
        
        loss.backward()
        
        grad_sign = X_adv.grad.sign()
        X_adv = X + self.epsilon * grad_sign
        
        X_adv = torch.clamp(X_adv, X.min(), X.max())
        
        self.attack_stats['fgsm_generated'] += len(X)
        
        return X_adv.detach()
    
    def _pgd_attack(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack
        
        Iterative version of FGSM that takes multiple small steps.
        More powerful but slower than FGSM.
        """
        X_adv = X.clone().detach()
        
        if self.model is None:
            log.warning("No model provided, returning perturbed data")
            return X + self.epsilon * torch.sign(torch.randn_like(X))
        
        X_adv = X_adv + torch.empty_like(X_adv).uniform_(-self.epsilon, self.epsilon)
        X_adv = torch.clamp(X_adv, X.min(), X.max())
        
        for step in range(self.num_steps):
            X_adv.requires_grad = True
            
            outputs = self.model(X_adv)
            
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                loss = F.cross_entropy(outputs, y)
            else:
                loss = F.mse_loss(outputs.squeeze(), y.float())
            
            loss.backward()
            
            grad_sign = X_adv.grad.sign()
            X_adv = X_adv + self.alpha * grad_sign
            
            perturbation = torch.clamp(X_adv - X, -self.epsilon, self.epsilon)
            X_adv = X + perturbation
            X_adv = torch.clamp(X_adv, X.min(), X.max())
            X_adv = X_adv.detach()
        
        self.attack_stats['pgd_generated'] += len(X)
        
        return X_adv
    
    def _market_manipulation_attack(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate market manipulation patterns
        
        Generates adversarial examples that mimic:
        - Pump and dump schemes
        - Spoofing (fake orders)
        - Wash trading
        - Flash crashes
        """
        X_adv = X.clone()
        batch_size = X.shape[0]
        
        manipulation_types = ['pump_dump', 'spoofing', 'flash_crash', 'wash_trading']
        
        for i in range(batch_size):
            manip_type = np.random.choice(manipulation_types)
            
            if manip_type == 'pump_dump':
                X_adv[i] = self._simulate_pump_dump(X[i])
            
            elif manip_type == 'spoofing':
                X_adv[i] = self._simulate_spoofing(X[i])
            
            elif manip_type == 'flash_crash':
                X_adv[i] = self._simulate_flash_crash(X[i])
            
            elif manip_type == 'wash_trading':
                X_adv[i] = self._simulate_wash_trading(X[i])
        
        self.attack_stats['manipulation_detected'] += batch_size
        
        return X_adv
    
    def _simulate_pump_dump(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate pump and dump pattern"""
        x_adv = x.clone()
        
        price_features = x_adv[:5]
        
        pump_factor = 1 + torch.rand(1).item() * 0.3 + 0.2
        price_features *= pump_factor
        
        x_adv[:5] = price_features
        
        return x_adv
    
    def _simulate_spoofing(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate spoofing (fake orders)"""
        x_adv = x.clone()
        
        volume_idx = len(x) // 2
        volume_features = x_adv[volume_idx:volume_idx+3]
        
        volume_features *= (5 + torch.rand(1).item() * 5)
        
        x_adv[volume_idx:volume_idx+3] = volume_features
        
        return x_adv
    
    def _simulate_flash_crash(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate flash crash"""
        x_adv = x.clone()
        
        crash_factor = 1 - (0.1 + torch.rand(1).item() * 0.2)
        x_adv[:5] *= crash_factor
        
        return x_adv
    
    def _simulate_wash_trading(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate wash trading"""
        x_adv = x.clone()
        
        volume_idx = len(x) // 2
        x_adv[volume_idx:volume_idx+3] *= (3 + torch.rand(1).item() * 3)
        
        return x_adv
    
    def train_with_adversarial_examples(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10
    ) -> Dict:
        """
        Train model with adversarial examples
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            num_epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        if self.model is None:
            raise ValueError("Model not provided")
        
        self.model.train()
        
        stats = {
            'epoch_losses': [],
            'adversarial_losses': [],
            'clean_losses': [],
            'total_examples': 0
        }
        
        augmentation_ratio = self.config['augmentation_ratio']
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            adv_loss = 0.0
            clean_loss = 0.0
            num_batches = 0
            
            for batch_idx, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                
                outputs_clean = self.model(X)
                if len(outputs_clean.shape) > 1 and outputs_clean.shape[1] > 1:
                    loss_clean = F.cross_entropy(outputs_clean, y)
                else:
                    loss_clean = F.mse_loss(outputs_clean.squeeze(), y.float())
                
                if np.random.random() < augmentation_ratio:
                    attack_type = np.random.choice(self.config['attack_types'])
                    X_adv = self.generate_adversarial_examples(X, y, attack_type)
                    
                    outputs_adv = self.model(X_adv)
                    if len(outputs_adv.shape) > 1 and outputs_adv.shape[1] > 1:
                        loss_adv = F.cross_entropy(outputs_adv, y)
                    else:
                        loss_adv = F.mse_loss(outputs_adv.squeeze(), y.float())
                    
                    robust_weight = self.config['robust_loss_weight']
                    loss = (1 - robust_weight) * loss_clean + robust_weight * loss_adv
                    
                    adv_loss += loss_adv.item()
                else:
                    loss = loss_clean
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                clean_loss += loss_clean.item()
                num_batches += 1
                stats['total_examples'] += len(X)
            
            avg_loss = epoch_loss / num_batches
            avg_clean = clean_loss / num_batches
            avg_adv = adv_loss / num_batches if adv_loss > 0 else 0
            
            stats['epoch_losses'].append(avg_loss)
            stats['clean_losses'].append(avg_clean)
            stats['adversarial_losses'].append(avg_adv)
            
            log.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} (Clean: {avg_clean:.4f}, Adv: {avg_adv:.4f})")
        
        log.info("Adversarial training complete")
        log.info(f"Attack stats: {self.attack_stats}")
        
        return stats
    
    def evaluate_robustness(
        self,
        test_loader: torch.utils.data.DataLoader,
        attack_types: List[str] = None
    ) -> Dict:
        """
        Evaluate model robustness against adversarial attacks
        
        Args:
            test_loader: Test data loader
            attack_types: List of attack types to evaluate
            
        Returns:
            Robustness metrics
        """
        if self.model is None:
            raise ValueError("Model not provided")
        
        if attack_types is None:
            attack_types = self.config['attack_types']
        
        self.model.eval()
        
        results = {
            'clean_accuracy': 0.0,
            'attack_results': {}
        }
        
        total_correct_clean = 0
        total_samples = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                outputs = self.model(X)
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    predictions = outputs.argmax(dim=1)
                    total_correct_clean += (predictions == y).sum().item()
                else:
                    predictions = (outputs.squeeze() > 0.5).long()
                    total_correct_clean += (predictions == y).sum().item()
                
                total_samples += len(y)
        
        results['clean_accuracy'] = total_correct_clean / total_samples if total_samples > 0 else 0
        
        for attack_type in attack_types:
            total_correct_adv = 0
            total_samples_adv = 0
            
            for X, y in test_loader:
                X_adv = self.generate_adversarial_examples(X, y, attack_type)
                
                with torch.no_grad():
                    outputs_adv = self.model(X_adv)
                    if len(outputs_adv.shape) > 1 and outputs_adv.shape[1] > 1:
                        predictions_adv = outputs_adv.argmax(dim=1)
                        total_correct_adv += (predictions_adv == y).sum().item()
                    else:
                        predictions_adv = (outputs_adv.squeeze() > 0.5).long()
                        total_correct_adv += (predictions_adv == y).sum().item()
                
                total_samples_adv += len(y)
            
            adv_accuracy = total_correct_adv / total_samples_adv if total_samples_adv > 0 else 0
            results['attack_results'][attack_type] = {
                'accuracy': adv_accuracy,
                'robustness': adv_accuracy / results['clean_accuracy'] if results['clean_accuracy'] > 0 else 0
            }
        
        log.info("Robustness evaluation complete")
        log.info(f"Clean accuracy: {results['clean_accuracy']:.2%}")
        for attack_type, metrics in results['attack_results'].items():
            log.info(f"{attack_type} - Accuracy: {metrics['accuracy']:.2%}, Robustness: {metrics['robustness']:.2%}")
        
        return results
    
    def detect_manipulation(
        self,
        market_data: pd.DataFrame,
        symbol: str
    ) -> Dict:
        """
        Detect potential market manipulation in real-time data
        
        Args:
            market_data: Recent market data
            symbol: Trading symbol
            
        Returns:
            Dict with manipulation detection results
        """
        detection_results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'manipulation_detected': False,
            'manipulation_type': None,
            'confidence': 0.0,
            'details': {}
        }
        
        if self._detect_pump_dump(market_data):
            detection_results['manipulation_detected'] = True
            detection_results['manipulation_type'] = 'pump_dump'
            detection_results['confidence'] = 0.8
        
        elif self._detect_spoofing(market_data):
            detection_results['manipulation_detected'] = True
            detection_results['manipulation_type'] = 'spoofing'
            detection_results['confidence'] = 0.7
        
        elif self._detect_wash_trading(market_data):
            detection_results['manipulation_detected'] = True
            detection_results['manipulation_type'] = 'wash_trading'
            detection_results['confidence'] = 0.6
        
        if detection_results['manipulation_detected']:
            log.warning(f"Manipulation detected for {symbol}: {detection_results['manipulation_type']} (confidence: {detection_results['confidence']:.1%})")
        
        return detection_results
    
    def _detect_pump_dump(self, data: pd.DataFrame) -> bool:
        """Detect pump and dump pattern"""
        if len(data) < 20:
            return False
        
        recent_returns = data['Close'].pct_change().tail(20)
        
        max_return = recent_returns.max()
        if max_return > 0.2:  # 20% spike
            returns_after_spike = recent_returns[recent_returns.idxmax():]
            if len(returns_after_spike) > 5 and returns_after_spike.mean() < -0.05:
                return True
        
        return False
    
    def _detect_spoofing(self, data: pd.DataFrame) -> bool:
        """Detect spoofing pattern"""
        if len(data) < 10:
            return False
        
        volume_changes = data['Volume'].pct_change().tail(10)
        price_changes = data['Close'].pct_change().tail(10)
        
        if volume_changes.max() > 5.0 and abs(price_changes.mean()) < 0.01:
            return True
        
        return False
    
    def _detect_wash_trading(self, data: pd.DataFrame) -> bool:
        """Detect wash trading pattern"""
        if len(data) < 10:
            return False
        
        recent_data = data.tail(10)
        avg_volume = recent_data['Volume'].mean()
        price_volatility = recent_data['Close'].std() / recent_data['Close'].mean()
        
        if avg_volume > data['Volume'].mean() * 2 and price_volatility < 0.01:
            return True
        
        return False
    
    def get_training_stats(self) -> Dict:
        """Get adversarial training statistics"""
        return {
            'attack_stats': self.attack_stats,
            'config': self.config,
            'total_adversarial_examples': self.attack_stats['total_adversarial_examples']
        }
