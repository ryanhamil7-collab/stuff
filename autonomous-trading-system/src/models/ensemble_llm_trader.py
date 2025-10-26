import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional
import json
from src.utils import log, config
from src.models.llm_trader import LLMTrader

class EnsembleLLMTrader:
    """
    Ensemble of multiple LLM models for more robust trading decisions.
    Combines predictions from multiple models using weighted voting.
    """
    
    def __init__(self):
        self.ensemble_config = config.get('llm.ensemble', {})
        self.enabled = self.ensemble_config.get('enabled', False)
        self.voting_strategy = self.ensemble_config.get('voting_strategy', 'weighted')
        
        self.models = []
        self.model_weights = []
        
        if self.enabled:
            log.info("Initializing Ensemble LLM Trader")
            self._load_ensemble_models()
        else:
            log.info("Ensemble disabled, using single model")
            self.primary_model = LLMTrader()
    
    def _load_ensemble_models(self):
        """Load all models in the ensemble"""
        models_config = self.ensemble_config.get('models', [])
        
        for model_config in models_config:
            model_name = model_config.get('name')
            weight = model_config.get('weight', 1.0)
            
            try:
                log.info(f"Loading ensemble model: {model_name} (weight: {weight})")
                model = LLMTrader(model_name=model_name)
                
                if model.model is not None:
                    self.models.append(model)
                    self.model_weights.append(weight)
                    log.info(f"✓ Loaded {model_name}")
                else:
                    log.warning(f"Failed to load {model_name}, skipping")
            
            except Exception as e:
                log.error(f"Error loading {model_name}: {str(e)}")
        
        if not self.models:
            log.warning("No ensemble models loaded, falling back to single model")
            self.enabled = False
            self.primary_model = LLMTrader()
        else:
            log.info(f"Ensemble initialized with {len(self.models)} models")
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
    
    def generate_trading_decision(
        self,
        symbol: str,
        technical_data: Dict,
        sentiment_data: Dict,
        alpha_signals: Dict,
        market_regime: str,
        **kwargs
    ) -> Dict:
        """Generate trading decision using ensemble or single model"""
        
        if not self.enabled:
            return self.primary_model.generate_trading_decision(
                symbol, technical_data, sentiment_data, alpha_signals, market_regime, **kwargs
            )
        
        decisions = []
        for model in self.models:
            try:
                decision = model.generate_trading_decision(
                    symbol, technical_data, sentiment_data, alpha_signals, market_regime, **kwargs
                )
                decisions.append(decision)
            except Exception as e:
                log.error(f"Error in ensemble model: {str(e)}")
                decisions.append({
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': 'Model error'
                })
        
        return self._combine_decisions(decisions, symbol)
    
    def batch_generate_decisions(
        self,
        symbols_data: Dict[str, Dict],
        batch_size: int = 16
    ) -> Dict[str, Dict]:
        """Generate batch decisions using ensemble or single model"""
        
        if not self.enabled:
            return self.primary_model.batch_generate_decisions(symbols_data, batch_size)
        
        all_decisions = {}
        
        for model_idx, model in enumerate(self.models):
            try:
                log.info(f"Running batch inference on ensemble model {model_idx + 1}/{len(self.models)}")
                model_decisions = model.batch_generate_decisions(symbols_data, batch_size)
                
                for symbol, decision in model_decisions.items():
                    if symbol not in all_decisions:
                        all_decisions[symbol] = []
                    all_decisions[symbol].append(decision)
            
            except Exception as e:
                log.error(f"Error in ensemble model {model_idx}: {str(e)}")
        
        combined_decisions = {}
        for symbol, decisions in all_decisions.items():
            combined_decisions[symbol] = self._combine_decisions(decisions, symbol)
        
        return combined_decisions
    
    def _combine_decisions(self, decisions: List[Dict], symbol: str) -> Dict:
        """Combine multiple model decisions using weighted voting"""
        
        if not decisions:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'No decisions available',
                'risk_level': 'MEDIUM',
                'time_horizon': 'MEDIUM'
            }
        
        if self.voting_strategy == 'weighted':
            return self._weighted_voting(decisions)
        elif self.voting_strategy == 'majority':
            return self._majority_voting(decisions)
        elif self.voting_strategy == 'confidence':
            return self._confidence_voting(decisions)
        else:
            return decisions[0]
    
    def _weighted_voting(self, decisions: List[Dict]) -> Dict:
        """Combine decisions using weighted average"""
        
        action_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_confidence = 0.0
        reasonings = []
        
        for idx, decision in enumerate(decisions):
            weight = self.model_weights[idx] if idx < len(self.model_weights) else 1.0
            action = decision.get('action', 'HOLD')
            confidence = decision.get('confidence', 0.5)
            
            action_scores[action] += weight * confidence
            total_confidence += weight * confidence
            reasonings.append(f"Model {idx+1}: {action} ({confidence:.2f})")
        
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action] / max(sum(self.model_weights), 1.0)
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'reasoning': ' | '.join(reasonings),
            'risk_level': 'MEDIUM',
            'time_horizon': 'MEDIUM',
            'ensemble_scores': action_scores
        }
    
    def _majority_voting(self, decisions: List[Dict]) -> Dict:
        """Combine decisions using majority vote"""
        
        action_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for decision in decisions:
            action = decision.get('action', 'HOLD')
            action_votes[action] += 1
        
        final_action = max(action_votes, key=action_votes.get)
        final_confidence = action_votes[final_action] / len(decisions)
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'reasoning': f'Majority vote: {action_votes}',
            'risk_level': 'MEDIUM',
            'time_horizon': 'MEDIUM'
        }
    
    def _confidence_voting(self, decisions: List[Dict]) -> Dict:
        """Select decision with highest confidence"""
        
        best_decision = max(decisions, key=lambda d: d.get('confidence', 0.0))
        return best_decision
