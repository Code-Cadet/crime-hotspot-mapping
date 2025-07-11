"""
Model evaluation and performance metrics for crime hotspot prediction.

This module provides comprehensive evaluation tools for assessing
the performance of crime prediction models, including spatial and
temporal accuracy metrics.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation framework for crime prediction models.
    
    Provides standard ML metrics plus specialized spatial and temporal
    evaluation methods for crime hotspot prediction.
    """
    
    def __init__(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model instance
            X_test: Test feature matrix
            y_test: Test target values
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        self.evaluation_results = {}
        
        self._make_predictions()
        
    def _make_predictions(self) -> None:
        """Generate predictions for evaluation."""
        self.y_pred = self.model.predict(self.X_test)
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            self.y_pred_proba = self.y_pred
            
    def calculate_standard_metrics(self) -> Dict[str, float]:
        """
        Calculate standard classification metrics.
        
        Returns:
            Dictionary of standard ML metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, average='binary'),
            'recall': recall_score(self.y_test, self.y_pred, average='binary'),
            'f1_score': f1_score(self.y_test, self.y_pred, average='binary'),
            'specificity': self._calculate_specificity(),
        }
        
        if self.y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
                'average_precision': average_precision_score(self.y_test, self.y_pred_proba)
            })
            
        self.evaluation_results['standard_metrics'] = metrics
        return metrics
        
    def _calculate_specificity(self) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
    def calculate_spatial_metrics(self, coordinates: np.ndarray, 
                                 grid_size: float = 0.01) -> Dict[str, float]:
        """
        Calculate spatial accuracy metrics for crime prediction.
        
        Args:
            coordinates: Array of (lat, lon) coordinates for test points
            grid_size: Size of spatial grid cells for analysis
            
        Returns:
            Dictionary of spatial metrics
        """
        spatial_metrics = {}
        
        # Spatial autocorrelation of predictions
        spatial_metrics['prediction_moran_i'] = self._calculate_spatial_autocorrelation(
            coordinates, self.y_pred_proba
        )
        
        # Spatial accuracy within neighborhoods
        spatial_metrics['neighborhood_accuracy'] = self._calculate_neighborhood_accuracy(
            coordinates, grid_size
        )
        
        # Hit rate for top-k predictions
        spatial_metrics['hit_rate_top_10'] = self._calculate_hit_rate(0.1)
        spatial_metrics['hit_rate_top_20'] = self._calculate_hit_rate(0.2)
        
        # Spatial concentration metrics
        spatial_metrics['gini_coefficient'] = self._calculate_gini_coefficient()
        
        self.evaluation_results['spatial_metrics'] = spatial_metrics
        return spatial_metrics
        
    def _calculate_spatial_autocorrelation(self, coordinates: np.ndarray, 
                                         values: np.ndarray) -> float:
        """Calculate Moran's I for spatial autocorrelation."""
        # TODO: Implement proper Moran's I calculation
        # This requires spatial weights matrix
        return 0.0
        
    def _calculate_neighborhood_accuracy(self, coordinates: np.ndarray, 
                                       grid_size: float) -> float:
        """Calculate accuracy within spatial neighborhoods."""
        # TODO: Implement neighborhood-based accuracy calculation
        return 0.0
        
    def _calculate_hit_rate(self, top_percentile: float) -> float:
        """
        Calculate hit rate for top-k predictions.
        
        Args:
            top_percentile: Percentile threshold for top predictions
            
        Returns:
            Hit rate (proportion of actual crimes in top predictions)
        """
        if self.y_pred_proba is None:
            return 0.0
            
        # Get threshold for top percentile
        threshold = np.percentile(self.y_pred_proba, (1 - top_percentile) * 100)
        
        # Calculate hit rate
        top_predictions = self.y_pred_proba >= threshold
        hit_rate = np.sum(self.y_test[top_predictions]) / np.sum(self.y_test)
        
        return hit_rate
        
    def _calculate_gini_coefficient(self) -> float:
        """Calculate Gini coefficient for prediction concentration."""
        if self.y_pred_proba is None:
            return 0.0
            
        # Sort predictions
        sorted_proba = np.sort(self.y_pred_proba)
        n = len(sorted_proba)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_proba)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
        
    def calculate_temporal_metrics(self, timestamps: pd.Series) -> Dict[str, float]:
        """
        Calculate temporal stability metrics.
        
        Args:
            timestamps: Time series of prediction timestamps
            
        Returns:
            Dictionary of temporal metrics
        """
        temporal_metrics = {}
        
        # Temporal consistency
        temporal_metrics['temporal_consistency'] = self._calculate_temporal_consistency(
            timestamps
        )
        
        # Prediction stability over time
        temporal_metrics['prediction_stability'] = self._calculate_prediction_stability(
            timestamps
        )
        
        self.evaluation_results['temporal_metrics'] = temporal_metrics
        return temporal_metrics
        
    def _calculate_temporal_consistency(self, timestamps: pd.Series) -> float:
        """Calculate temporal consistency of predictions."""
        # TODO: Implement temporal consistency calculation
        return 0.0
        
    def _calculate_prediction_stability(self, timestamps: pd.Series) -> float:
        """Calculate stability of predictions over time."""
        # TODO: Implement prediction stability calculation
        return 0.0
        
    def perform_cross_validation(self, model_class: Any, X: pd.DataFrame, 
                                y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model_class: Model class to evaluate
            X: Full feature matrix
            y: Full target array
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        # Standard cross-validation
        cv_scores = cross_val_score(model_class, X, y, cv=cv_folds, scoring='roc_auc')
        
        # Time series cross-validation (if temporal data available)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        ts_scores = cross_val_score(model_class, X, y, cv=tscv, scoring='roc_auc')
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'ts_cv_scores': ts_scores,
            'ts_cv_mean': np.mean(ts_scores),
            'ts_cv_std': np.std(ts_scores)
        }
        
        self.evaluation_results['cross_validation'] = cv_results
        return cv_results
        
    def generate_confusion_matrix_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Generate confusion matrix visualization."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def generate_roc_curve_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Generate ROC curve visualization."""
        if self.y_pred_proba is None:
            raise ValueError("Probability predictions required for ROC curve")
            
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def generate_precision_recall_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """Generate precision-recall curve visualization."""
        if self.y_pred_proba is None:
            raise ValueError("Probability predictions required for PR curve")
            
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        ap_score = average_precision_score(self.y_test, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def generate_feature_importance_plot(self, feature_names: List[str],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Generate feature importance visualization."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not provide feature importance")
            
        importance = self.model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1][:20]  # Top 20 features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importance[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importances')
        ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'model_type': type(self.model).__name__,
            'test_set_size': len(self.y_test),
            'class_distribution': {
                'positive_class': np.sum(self.y_test),
                'negative_class': len(self.y_test) - np.sum(self.y_test),
                'positive_ratio': np.mean(self.y_test)
            },
            'evaluation_results': self.evaluation_results
        }
        
        return report
        
    def save_evaluation_results(self, filepath: str) -> None:
        """Save evaluation results to file."""
        report = self.generate_comprehensive_report()
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=convert_numpy)


class ModelComparison:
    """
    Compare multiple models on the same dataset.
    """
    
    def __init__(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: np.ndarray):
        """
        Initialize model comparison.
        
        Args:
            models: Dictionary of model_name: model_instance
            X_test: Test features
            y_test: Test targets
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.comparison_results = {}
        
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models and return results DataFrame.
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_name, model in self.models.items():
            evaluator = ModelEvaluator(model, self.X_test, self.y_test)
            metrics = evaluator.calculate_standard_metrics()
            
            result_row = {'model': model_name}
            result_row.update(metrics)
            results.append(result_row)
            
        comparison_df = pd.DataFrame(results)
        self.comparison_results = comparison_df
        
        return comparison_df
        
    def generate_comparison_plots(self, save_dir: str) -> None:
        """Generate comparison visualizations."""
        if self.comparison_results is None:
            self.compare_models()
            
        # Metric comparison bar plot
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.comparison_results.columns:
                ax = axes[i]
                self.comparison_results.plot(x='model', y=metric, kind='bar', ax=ax)
                ax.set_title(f'{metric.title()} Comparison')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.title())
                ax.tick_params(axis='x', rotation=45)
                
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Example usage of model evaluation."""
    print("Model evaluation module initialized")
    print("Available evaluation metrics:")
    print("- Standard ML metrics (accuracy, precision, recall, F1, ROC-AUC)")
    print("- Spatial metrics (hit rate, spatial autocorrelation, Gini coefficient)")
    print("- Temporal metrics (consistency, stability)")
    

if __name__ == "__main__":
    main()
