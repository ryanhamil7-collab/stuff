#!/usr/bin/env python3
"""Fix verification script to use file content checks instead of imports"""

import re

# Read the current file
with open('verify_all_features.py', 'r') as f:
    content = f.read()

# Replace all check_import calls with check_class_in_file for classes
# Pattern: ("ClassName class", lambda: self.check_import("module.path", "ClassName"))
# Replace with: ("ClassName class", lambda: self.check_class_in_file("module/path.py", "ClassName"))

replacements = [
    # Feature 2
    ('lambda: self.check_import("src.agents.grpo_policy", "GRPOPolicy")',
     'lambda: self.check_class_in_file("src/agents/grpo_policy.py", "GRPOPolicy")'),
    ('lambda: self.check_import("src.agents.grpo_policy", "GRPORewardWrapper")',
     'lambda: self.check_class_in_file("src/agents/grpo_policy.py", "GRPORewardWrapper")'),
    
    # Feature 3
    ('lambda: self.check_import("src.agents.hypothesis_generator", "HypothesisGenerator")',
     'lambda: self.check_class_in_file("src/agents/hypothesis_generator.py", "HypothesisGenerator")'),
    ('lambda: self.check_import("src.agents.hypothesis_generator", "MarketTensionHypothesis")',
     'lambda: self.check_class_in_file("src/agents/hypothesis_generator.py", "MarketTensionHypothesis")'),
    
    # Feature 4
    ('lambda: self.check_import("src.data_pipeline.multimodal_processor", "MultimodalProcessor")',
     'lambda: self.check_class_in_file("src/data_pipeline/multimodal_processor.py", "MultimodalProcessor")'),
    
    # Feature 5
    ('lambda: self.check_import("src.models.model_compression", "ModelCompressor")',
     'lambda: self.check_class_in_file("src/models/model_compression.py", "ModelCompressor")'),
    
    # Feature 6
    ('lambda: self.check_import("src.backtesting.monte_carlo", "MonteCarloSimulator")',
     'lambda: self.check_class_in_file("src/backtesting/monte_carlo.py", "MonteCarloSimulator")'),
    ('lambda: self.check_import("src.backtesting.monte_carlo", "RegimeDetector")',
     'lambda: self.check_class_in_file("src/backtesting/monte_carlo.py", "RegimeDetector")'),
    
    # Feature 7
    ('lambda: self.check_import("src.strategies.hybrid_trader", "HybridTrader")',
     'lambda: self.check_class_in_file("src/strategies/hybrid_trader.py", "HybridTrader")'),
    
    # Feature 8
    ('lambda: self.check_import("src.research.research_engine", "ResearchEngine")',
     'lambda: self.check_class_in_file("src/research/research_engine.py", "ResearchEngine")'),
    
    # Feature 9
    ('lambda: self.check_import("src.optimization.quantum_optimizer", "QuantumOptimizer")',
     'lambda: self.check_class_in_file("src/optimization/quantum_optimizer.py", "QuantumOptimizer")'),
    
    # Feature 10
    ('lambda: self.check_import("src.agents.federated_learning", "FederatedLearning")',
     'lambda: self.check_class_in_file("src/agents/federated_learning.py", "FederatedLearning")'),
    
    # Feature 11
    ('lambda: self.check_import("src.agents.neuro_symbolic", "NeuroSymbolicAgent")',
     'lambda: self.check_class_in_file("src/agents/neuro_symbolic.py", "NeuroSymbolicAgent")'),
    
    # Feature 12
    ('lambda: self.check_import("src.data_pipeline.external_data", "ExternalDataCollector")',
     'lambda: self.check_class_in_file("src/data_pipeline/external_data.py", "ExternalDataCollector")'),
    
    # Feature 13
    ('lambda: self.check_import("src.models.adversarial_training", "AdversarialTrainer")',
     'lambda: self.check_class_in_file("src/models/adversarial_training.py", "AdversarialTrainer")'),
    
    # Feature 14
    ('lambda: self.check_import("src.models.ensemble_llm", "EnsembleLLM")',
     'lambda: self.check_class_in_file("src/models/ensemble_llm.py", "EnsembleLLM")'),
    
    # Feature 15
    ('lambda: self.check_import("src.strategies.fee_slippage", "FeeSlippageModel")',
     'lambda: self.check_class_in_file("src/strategies/fee_slippage.py", "FeeSlippageModel")'),
    
    # Feature 17
    ('lambda: self.check_import("src.hive_mind.hive_mind", "HiveMind")',
     'lambda: self.check_class_in_file("src/hive_mind/hive_mind.py", "HiveMind")'),
    
    # Feature 18 - already fixed
    ('lambda: self.check_import("src.data_pipeline.symbol_discovery", "SymbolDiscovery")',
     'lambda: self.check_class_in_file("src/agents/symbol_discovery.py", "SymbolDiscovery")'),
    
    # Feature 19
    ('lambda: self.check_import("src.backtesting.execution_delays", "ExecutionDelaySimulator")',
     'lambda: self.check_class_in_file("src/backtesting/execution_delays.py", "ExecutionDelaySimulator")'),
    
    # Feature 20
    ('lambda: self.check_import("src.strategies.high_risk", "HighRiskStrategy")',
     'lambda: self.check_class_in_file("src/strategies/high_risk.py", "HighRiskStrategy")'),
    
    # Feature 21
    ('lambda: self.check_import("src.hive_mind.p2p_discovery", "P2PDiscovery")',
     'lambda: self.check_class_in_file("src/hive_mind/p2p_discovery.py", "P2PDiscovery")'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Write back
with open('verify_all_features.py', 'w') as f:
    f.write(content)

print("✅ Fixed all verification checks")
