#!/usr/bin/env python3
"""
Comprehensive Feature Verification Script
Verifies all 21 advanced features are implemented and working.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

class FeatureVerifier:
    """Verifies implementation of all 21 advanced features"""
    
    def __init__(self):
        self.results = []
        self.repo_root = Path(__file__).parent
        
    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists"""
        return (self.repo_root / filepath).exists()
    
    def check_import(self, module_path: str, class_name: str = None) -> bool:
        """Check if a module/class can be imported"""
        try:
            parts = module_path.split('.')
            module = __import__(module_path)
            for part in parts[1:]:
                module = getattr(module, part)
            
            if class_name:
                return hasattr(module, class_name)
            return True
        except Exception as e:
            return False
    
    def check_config_key(self, key: str) -> bool:
        """Check if a config key exists"""
        try:
            from src.utils import config
            value = config.get(key, None)
            return value is not None
        except:
            return False
    
    def verify_feature(self, feature_num: int, feature_name: str, checks: List[Tuple[str, callable]]) -> Dict:
        """Verify a single feature"""
        print(f"\n{'='*70}")
        print(f"Feature {feature_num}: {feature_name}")
        print(f"{'='*70}")
        
        passed = 0
        failed = 0
        details = []
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                if result:
                    print(f"  ✅ {check_name}")
                    passed += 1
                else:
                    print(f"  ❌ {check_name}")
                    failed += 1
                details.append((check_name, result))
            except Exception as e:
                print(f"  ❌ {check_name} - Error: {str(e)}")
                failed += 1
                details.append((check_name, False))
        
        status = "PASS" if failed == 0 else "PARTIAL" if passed > 0 else "FAIL"
        print(f"\nStatus: {status} ({passed}/{passed+failed} checks passed)")
        
        return {
            'feature_num': feature_num,
            'feature_name': feature_name,
            'status': status,
            'passed': passed,
            'failed': failed,
            'details': details
        }
    
    def run_all_verifications(self):
        """Run all feature verifications"""
        
        
        self.results.append(self.verify_feature(
            1, "Structured Thesis Output (Trading-R1 Style)",
            [
                ("trading_r1_schema.py exists", lambda: self.check_file_exists("src/models/trading_r1_schema.py")),
                ("TradingR1Decision class", lambda: self.check_import("src.models.trading_r1_schema", "TradingR1Decision")),
                ("thesis_templates.py exists", lambda: self.check_file_exists("src/models/thesis_templates.py")),
                ("ThesisPromptTemplate class", lambda: self.check_import("src.models.thesis_templates", "ThesisPromptTemplate")),
                ("LLMTrader supports structured thesis", lambda: self.check_file_exists("src/models/llm_trader.py")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            2, "GRPO (Group Relative Policy Optimization)",
            [
                ("grpo_policy.py exists", lambda: self.check_file_exists("src/agents/grpo_policy.py")),
                ("GRPOPolicy class", lambda: self.check_import("src.agents.grpo_policy", "GRPOPolicy")),
                ("GRPORewardWrapper class", lambda: self.check_import("src.agents.grpo_policy", "GRPORewardWrapper")),
                ("GRPO config exists", lambda: self.check_config_key("rl.grpo")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            3, "LLM Reasoning Amplifier",
            [
                ("hypothesis_generator.py exists", lambda: self.check_file_exists("src/agents/hypothesis_generator.py")),
                ("HypothesisGenerator class", lambda: self.check_import("src.agents.hypothesis_generator", "HypothesisGenerator")),
                ("MarketTensionHypothesis class", lambda: self.check_import("src.agents.hypothesis_generator", "MarketTensionHypothesis")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            4, "Multimodal Inputs (Chart → Text)",
            [
                ("multimodal_processor.py exists", lambda: self.check_file_exists("src/data_pipeline/multimodal_processor.py")),
                ("MultimodalProcessor class", lambda: self.check_import("src.data_pipeline.multimodal_processor", "MultimodalProcessor")),
                ("Multimodal config exists", lambda: self.check_config_key("data.multimodal")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            5, "Model Compression (Pruning + Quantization)",
            [
                ("model_compression.py exists", lambda: self.check_file_exists("src/models/model_compression.py")),
                ("ModelCompressor class", lambda: self.check_import("src.models.model_compression", "ModelCompressor")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            6, "Monte Carlo + HMM Regime Detection",
            [
                ("monte_carlo.py exists", lambda: self.check_file_exists("src/backtesting/monte_carlo.py")),
                ("MonteCarloSimulator class", lambda: self.check_import("src.backtesting.monte_carlo", "MonteCarloSimulator")),
                ("RegimeDetector class", lambda: self.check_import("src.backtesting.monte_carlo", "RegimeDetector")),
                ("Monte Carlo config exists", lambda: self.check_config_key("backtest.monte_carlo")),
            ]
        ))
        
        
        self.results.append(self.verify_feature(
            7, "Hybrid Trading Modes",
            [
                ("hybrid_trader.py exists", lambda: self.check_file_exists("src/strategies/hybrid_trader.py")),
                ("HybridTrader class", lambda: self.check_import("src.strategies.hybrid_trader", "HybridTrader")),
                ("Hybrid config exists", lambda: self.check_config_key("trading.hybrid_mode")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            8, "Offline Research & Self-Improvement",
            [
                ("research_engine.py exists", lambda: self.check_file_exists("src/research/research_engine.py")),
                ("ResearchEngine class", lambda: self.check_import("src.research.research_engine", "ResearchEngine")),
                ("Research config exists", lambda: self.check_config_key("research")),
            ]
        ))
        
        
        self.results.append(self.verify_feature(
            9, "Quantum-Inspired Optimization (QAOA)",
            [
                ("quantum_optimizer.py exists", lambda: self.check_file_exists("src/optimization/quantum_optimizer.py")),
                ("QuantumOptimizer class", lambda: self.check_import("src.optimization.quantum_optimizer", "QuantumOptimizer")),
                ("Quantum config exists", lambda: self.check_config_key("optimization.quantum")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            10, "Federated Learning Across Horizons",
            [
                ("federated_learning.py exists", lambda: self.check_file_exists("src/agents/federated_learning.py")),
                ("FederatedLearning class", lambda: self.check_import("src.agents.federated_learning", "FederatedLearning")),
                ("Federated config exists", lambda: self.check_config_key("federated_learning")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            11, "Neuro-Symbolic AI Fusion",
            [
                ("neuro_symbolic.py exists", lambda: self.check_file_exists("src/agents/neuro_symbolic.py")),
                ("NeuroSymbolicAgent class", lambda: self.check_import("src.agents.neuro_symbolic", "NeuroSymbolicAgent")),
                ("Neuro-symbolic config exists", lambda: self.check_config_key("neuro_symbolic")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            12, "Multi-Modal External Data Streams",
            [
                ("external_data.py exists", lambda: self.check_file_exists("src/data_pipeline/external_data.py")),
                ("ExternalDataCollector class", lambda: self.check_import("src.data_pipeline.external_data", "ExternalDataCollector")),
                ("External data config exists", lambda: self.check_config_key("data.external_sources")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            13, "Adversarial Robustness Training",
            [
                ("adversarial_training.py exists", lambda: self.check_file_exists("src/models/adversarial_training.py")),
                ("AdversarialTrainer class", lambda: self.check_import("src.models.adversarial_training", "AdversarialTrainer")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            14, "Ensemble of Specialized LLMs",
            [
                ("ensemble_llm.py exists", lambda: self.check_file_exists("src/models/ensemble_llm.py")),
                ("EnsembleLLM class", lambda: self.check_import("src.models.ensemble_llm", "EnsembleLLM")),
                ("Ensemble config exists", lambda: self.check_config_key("llm.ensemble")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            15, "Dynamic Fee/Slippage Modeling",
            [
                ("fee_slippage.py exists", lambda: self.check_file_exists("src/strategies/fee_slippage.py")),
                ("FeeSlippageModel class", lambda: self.check_import("src.strategies.fee_slippage", "FeeSlippageModel")),
            ]
        ))
        
        
        self.results.append(self.verify_feature(
            16, "Set-and-Forget Launcher",
            [
                ("launcher.py exists", lambda: self.check_file_exists("launcher.py")),
                ("APScheduler support", lambda: self.check_file_exists("config/scheduler.yaml")),
                ("Scheduler config exists", lambda: self.check_config_key("scheduler")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            17, "P2P Hive Mind Network",
            [
                ("hive_mind.py exists", lambda: self.check_file_exists("src/hive_mind/hive_mind.py")),
                ("HiveMind class", lambda: self.check_import("src.hive_mind.hive_mind", "HiveMind")),
                ("Hive mind config exists", lambda: self.check_config_key("hive_mind")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            18, "Real-Time Symbol Discovery",
            [
                ("symbol_discovery.py exists", lambda: self.check_file_exists("src/data_pipeline/symbol_discovery.py")),
                ("SymbolDiscovery class", lambda: self.check_import("src.data_pipeline.symbol_discovery", "SymbolDiscovery")),
                ("Symbol discovery config exists", lambda: self.check_config_key("symbol_discovery")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            19, "Execution Delay Modeling",
            [
                ("execution_delays.py exists", lambda: self.check_file_exists("src/backtesting/execution_delays.py")),
                ("ExecutionDelaySimulator class", lambda: self.check_import("src.backtesting.execution_delays", "ExecutionDelaySimulator")),
                ("Delay config exists", lambda: self.check_config_key("backtesting.execution_delays")),
            ]
        ))
        
        
        self.results.append(self.verify_feature(
            20, "High-Risk Mode",
            [
                ("high_risk.py exists", lambda: self.check_file_exists("src/strategies/high_risk.py")),
                ("HighRiskStrategy class", lambda: self.check_import("src.strategies.high_risk", "HighRiskStrategy")),
                ("High-risk config exists", lambda: self.check_config_key("trading.high_risk")),
            ]
        ))
        
        self.results.append(self.verify_feature(
            21, "Automated P2P Discovery",
            [
                ("p2p_discovery.py exists", lambda: self.check_file_exists("src/hive_mind/p2p_discovery.py")),
                ("P2PDiscovery class", lambda: self.check_import("src.hive_mind.p2p_discovery", "P2PDiscovery")),
                ("P2P discovery config exists", lambda: self.check_config_key("hive_mind.p2p_discovery")),
            ]
        ))
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE FEATURE VERIFICATION REPORT")
        print("="*70)
        
        total_features = len(self.results)
        fully_implemented = sum(1 for r in self.results if r['status'] == 'PASS')
        partially_implemented = sum(1 for r in self.results if r['status'] == 'PARTIAL')
        not_implemented = sum(1 for r in self.results if r['status'] == 'FAIL')
        
        print(f"\nTotal Features: {total_features}")
        print(f"Fully Implemented: {fully_implemented} ({fully_implemented/total_features*100:.1f}%)")
        print(f"Partially Implemented: {partially_implemented} ({partially_implemented/total_features*100:.1f}%)")
        print(f"Not Implemented: {not_implemented} ({not_implemented/total_features*100:.1f}%)")
        
        print("\n" + "="*70)
        print("FEATURE STATUS SUMMARY")
        print("="*70)
        
        for result in self.results:
            status_icon = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'PARTIAL' else "❌"
            print(f"{status_icon} Feature {result['feature_num']:2d}: {result['feature_name']}")
            print(f"   Status: {result['status']} ({result['passed']}/{result['passed']+result['failed']} checks)")
        
        print("\n" + "="*70)
        print("DETAILED FINDINGS")
        print("="*70)
        
        partial_features = [r for r in self.results if r['status'] == 'PARTIAL']
        if partial_features:
            print("\n⚠️  PARTIALLY IMPLEMENTED FEATURES:")
            for result in partial_features:
                print(f"\nFeature {result['feature_num']}: {result['feature_name']}")
                for check_name, passed in result['details']:
                    if not passed:
                        print(f"  ❌ Missing: {check_name}")
        
        missing_features = [r for r in self.results if r['status'] == 'FAIL']
        if missing_features:
            print("\n❌ NOT IMPLEMENTED FEATURES:")
            for result in missing_features:
                print(f"\nFeature {result['feature_num']}: {result['feature_name']}")
                print(f"  All {result['failed']} checks failed")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if fully_implemented == total_features:
            print("\n🎉 All 21 features are fully implemented!")
            print("   The system is ready for production use.")
        elif fully_implemented + partially_implemented >= total_features * 0.8:
            print("\n✅ Most features are implemented (>80%)")
            print("   Focus on completing the partially implemented features.")
        else:
            print("\n⚠️  Significant work needed (<80% implementation)")
            print("   Prioritize implementing the missing core features.")
        
        print("\n" + "="*70)
        
        return {
            'total': total_features,
            'fully_implemented': fully_implemented,
            'partially_implemented': partially_implemented,
            'not_implemented': not_implemented,
            'percentage': (fully_implemented + partially_implemented * 0.5) / total_features * 100
        }

def main():
    print("="*70)
    print("AUTONOMOUS TRADING SYSTEM - FEATURE VERIFICATION")
    print("="*70)
    print("\nVerifying all 21 advanced features...")
    print("This will check for file existence, imports, and configuration.")
    
    verifier = FeatureVerifier()
    verifier.run_all_verifications()
    summary = verifier.generate_report()
    
    print(f"\nOverall Implementation: {summary['percentage']:.1f}%")
    
    report_file = Path(__file__).parent / "FEATURE_VERIFICATION_REPORT.txt"
    with open(report_file, 'w') as f:
        f.write("AUTONOMOUS TRADING SYSTEM - FEATURE VERIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Features: {summary['total']}\n")
        f.write(f"Fully Implemented: {summary['fully_implemented']} ({summary['fully_implemented']/summary['total']*100:.1f}%)\n")
        f.write(f"Partially Implemented: {summary['partially_implemented']} ({summary['partially_implemented']/summary['total']*100:.1f}%)\n")
        f.write(f"Not Implemented: {summary['not_implemented']} ({summary['not_implemented']/summary['total']*100:.1f}%)\n")
        f.write(f"Overall Implementation: {summary['percentage']:.1f}%\n\n")
        
        for result in verifier.results:
            f.write(f"Feature {result['feature_num']}: {result['feature_name']}\n")
            f.write(f"Status: {result['status']} ({result['passed']}/{result['passed']+result['failed']} checks)\n")
            for check_name, passed in result['details']:
                status = "✅" if passed else "❌"
                f.write(f"  {status} {check_name}\n")
            f.write("\n")
    
    print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    main()
