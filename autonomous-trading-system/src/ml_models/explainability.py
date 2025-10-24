"""
Model Explainability with SHAP and LIME

Provides interpretability for ML trading models:
1. SHAP (SHapley Additive exPlanations) - Game theory based
2. LIME (Local Interpretable Model-agnostic Explanations) - Local approximations

Target: Understand why models make specific predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable
import shap
from lime import lime_tabular
from src.utils import log, config


class SHAPExplainer:
    """
    SHAP-based Model Explainer
    
    Uses Shapley values to explain model predictions.
    
    Features:
    - TreeExplainer for tree-based models (XGBoost, RF)
    - DeepExplainer for neural networks
    - KernelExplainer for any model
    - Summary plots and force plots
    """
    
    def __init__(
        self,
        model,
        explainer_type: str = 'auto',
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            explainer_type: 'tree', 'deep', 'kernel', or 'auto'
            background_data: Background dataset for KernelExplainer
        """
        self.model = model
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.explainer = None
        
        self._build_explainer()
        
        log.info(f"SHAPExplainer initialized - Type: {self.explainer_type}")
    
    def _build_explainer(self):
        """Build SHAP explainer"""
        if self.explainer_type == 'auto':
            model_type = type(self.model).__name__
            
            if 'XGB' in model_type or 'RandomForest' in model_type:
                self.explainer_type = 'tree'
            elif 'Sequential' in model_type or 'Module' in model_type:
                self.explainer_type = 'deep'
            else:
                self.explainer_type = 'kernel'
        
        if self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        
        elif self.explainer_type == 'deep':
            if self.background_data is None:
                raise ValueError("background_data required for DeepExplainer")
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        
        elif self.explainer_type == 'kernel':
            if self.background_data is None:
                raise ValueError("background_data required for KernelExplainer")
            
            def predict_fn(X):
                return self.model.predict(X)
            
            self.explainer = shap.KernelExplainer(predict_fn, self.background_data)
        
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
    
    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> shap.Explanation:
        """
        Explain predictions
        
        Args:
            X: Input data
            feature_names: Feature names
            
        Returns:
            SHAP explanation object
        """
        log.info(f"Computing SHAP values for {len(X)} samples...")
        
        shap_values = self.explainer.shap_values(X)
        
        if feature_names is not None:
            explanation = shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=X,
                feature_names=feature_names
            )
        else:
            explanation = shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=X
            )
        
        return explanation
    
    def plot_summary(
        self,
        shap_values,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP summary
        
        Args:
            shap_values: SHAP values
            X: Input data
            feature_names: Feature names
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            log.info(f"Summary plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_force(
        self,
        shap_values,
        X: np.ndarray,
        sample_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP force plot for single prediction
        
        Args:
            shap_values: SHAP values
            X: Input data
            sample_idx: Sample index to explain
            feature_names: Feature names
            save_path: Path to save plot
        """
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[sample_idx],
            X[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            log.info(f"Force plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_waterfall(
        self,
        explanation: shap.Explanation,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP waterfall for single prediction
        
        Args:
            explanation: SHAP explanation
            sample_idx: Sample index
            save_path: Path to save plot
        """
        shap.waterfall_plot(
            explanation[sample_idx],
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            log.info(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(
        self,
        shap_values,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from SHAP values
        
        Args:
            shap_values: SHAP values
            feature_names: Feature names
            
        Returns:
            DataFrame with feature importance
        """
        importance = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LIMEExplainer:
    """
    LIME-based Model Explainer
    
    Uses local linear approximations to explain predictions.
    
    Features:
    - Tabular data explainer
    - Classification and regression support
    - Feature importance for individual predictions
    """
    
    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'classification'
    ):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained model
            training_data: Training data for sampling
            feature_names: Feature names
            class_names: Class names (for classification)
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode
        )
        
        log.info(f"LIMEExplainer initialized - Mode: {mode}")
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> lime_tabular.LimeTabularExplainer:
        """
        Explain single instance
        
        Args:
            instance: Single data point
            num_features: Number of features to show
            num_samples: Number of samples for approximation
            
        Returns:
            LIME explanation
        """
        log.info(f"Explaining instance with LIME (num_samples={num_samples})...")
        
        if self.mode == 'classification':
            predict_fn = lambda x: self.model.predict_proba(x)
        else:
            predict_fn = lambda x: self.model.predict(x)
        
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation
    
    def plot_explanation(
        self,
        explanation,
        save_path: Optional[str] = None
    ):
        """
        Plot LIME explanation
        
        Args:
            explanation: LIME explanation
            save_path: Path to save plot
        """
        fig = explanation.as_pyplot_figure()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            log.info(f"LIME plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(
        self,
        explanation,
        label: int = 1
    ) -> pd.DataFrame:
        """
        Get feature importance from LIME explanation
        
        Args:
            explanation: LIME explanation
            label: Class label (for classification)
            
        Returns:
            DataFrame with feature importance
        """
        weights = explanation.as_list(label=label)
        
        features = []
        importances = []
        
        for feature, importance in weights:
            features.append(feature)
            importances.append(importance)
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        })
        
        return importance_df


class ExplainabilityManager:
    """
    Unified Explainability Manager
    
    Manages both SHAP and LIME explainers.
    
    Features:
    - Automatic explainer selection
    - Batch explanations
    - Visualization management
    - Report generation
    """
    
    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'classification'
    ):
        """
        Initialize explainability manager
        
        Args:
            model: Trained model
            training_data: Training data
            feature_names: Feature names
            class_names: Class names
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        
        self.shap_explainer = None
        self.lime_explainer = None
        
        log.info("ExplainabilityManager initialized")
    
    def init_shap(
        self,
        explainer_type: str = 'auto',
        background_data: Optional[np.ndarray] = None
    ):
        """Initialize SHAP explainer"""
        if background_data is None:
            background_data = self.training_data[:100]  # Sample for efficiency
        
        self.shap_explainer = SHAPExplainer(
            model=self.model,
            explainer_type=explainer_type,
            background_data=background_data
        )
        
        log.info("SHAP explainer initialized")
    
    def init_lime(self):
        """Initialize LIME explainer"""
        self.lime_explainer = LIMEExplainer(
            model=self.model,
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode
        )
        
        log.info("LIME explainer initialized")
    
    def explain_with_shap(
        self,
        X: np.ndarray,
        plot_summary: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Explain with SHAP
        
        Args:
            X: Input data
            plot_summary: Whether to plot summary
            save_dir: Directory to save plots
            
        Returns:
            Explanation results
        """
        if self.shap_explainer is None:
            self.init_shap()
        
        explanation = self.shap_explainer.explain(X, self.feature_names)
        
        if plot_summary:
            save_path = f"{save_dir}/shap_summary.png" if save_dir else None
            self.shap_explainer.plot_summary(
                explanation.values,
                X,
                self.feature_names,
                save_path
            )
        
        importance_df = self.shap_explainer.get_feature_importance(
            explanation.values,
            self.feature_names
        )
        
        return {
            'explanation': explanation,
            'feature_importance': importance_df
        }
    
    def explain_with_lime(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        plot: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Explain with LIME
        
        Args:
            instance: Single instance
            num_features: Number of features
            plot: Whether to plot
            save_dir: Directory to save plots
            
        Returns:
            Explanation results
        """
        if self.lime_explainer is None:
            self.init_lime()
        
        explanation = self.lime_explainer.explain_instance(
            instance,
            num_features=num_features
        )
        
        if plot:
            save_path = f"{save_dir}/lime_explanation.png" if save_dir else None
            self.lime_explainer.plot_explanation(explanation, save_path)
        
        label = 1 if self.mode == 'classification' else 0
        importance_df = self.lime_explainer.get_feature_importance(explanation, label)
        
        return {
            'explanation': explanation,
            'feature_importance': importance_df
        }
    
    def generate_report(
        self,
        X: np.ndarray,
        sample_indices: List[int] = None,
        save_dir: str = './explainability_reports'
    ):
        """
        Generate comprehensive explainability report
        
        Args:
            X: Input data
            sample_indices: Specific samples to explain
            save_dir: Directory to save reports
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        log.info("Generating explainability report...")
        
        shap_results = self.explain_with_shap(X, plot_summary=True, save_dir=save_dir)
        
        if sample_indices is None:
            sample_indices = [0, len(X)//2, len(X)-1]  # First, middle, last
        
        lime_results = []
        for idx in sample_indices:
            result = self.explain_with_lime(
                X[idx],
                plot=True,
                save_dir=f"{save_dir}/lime_sample_{idx}"
            )
            lime_results.append(result)
        
        shap_results['feature_importance'].to_csv(
            f"{save_dir}/shap_feature_importance.csv",
            index=False
        )
        
        log.info(f"Explainability report saved to {save_dir}")
        
        return {
            'shap': shap_results,
            'lime': lime_results
        }
