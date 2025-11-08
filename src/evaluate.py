"""
ECG Model Evaluation Module

Addresses Reviewer Requirements:
- A-2, D-6: Comprehensive metrics
- D-16, D-26: Statistical significance testing
- D-26: Mean ± std reporting with confidence intervals

Author: Abdullah Hasan Dafa
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger(__name__)


class ECGEvaluator:
    """
    Handles comprehensive evaluation and statistical testing.
    
    Addresses Reviewer D-16, D-26:
    "No statistical significance tests. Single-run results insufficient."
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['data']['results_dir'])
        self.metrics_dir = self.results_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ECGEvaluator initialized")
    
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict:
        """
        Calculate all metrics for a single evaluation.
        
        Addresses Reviewer A-2, D-6:
        "Report comprehensive metrics."
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC)
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_normal'] = precision[0]
        metrics['precision_abnormal'] = precision[1]
        metrics['recall_normal'] = recall[0]
        metrics['recall_abnormal'] = recall[1]
        metrics['f1_normal'] = f1[0]
        metrics['f1_abnormal'] = f1[1]
        
        # Macro averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Specificity (for normal class)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def aggregate_cv_results(self, model_name: str) -> pd.DataFrame:
        """
        Aggregate cross-validation results with mean ± std.
        
        Addresses Reviewer D-26:
        "Report mean ± SD across cross-validation folds"
        
        Args:
            model_name: Name of the model
        
        Returns:
            DataFrame with aggregated results
        """
        # Load CV results
        cv_path = self.metrics_dir / f'{model_name}_cv_results.csv'
        if not cv_path.exists():
            raise FileNotFoundError(f"CV results not found: {cv_path}")
        
        df = pd.read_csv(cv_path)
        
        # Calculate mean and std for each metric
        metrics = [col for col in df.columns if col not in ['model', 'fold', 'n_train', 'n_val']]
        
        aggregated = {'model': model_name}
        
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                
                aggregated[f'{metric}_mean'] = mean_val
                aggregated[f'{metric}_std'] = std_val
                
                # Calculate 95% confidence interval
                if len(df) > 1:
                    ci = stats.t.interval(
                        0.95,
                        len(df) - 1,
                        loc=mean_val,
                        scale=stats.sem(df[metric])
                    )
                    aggregated[f'{metric}_ci_lower'] = ci[0]
                    aggregated[f'{metric}_ci_upper'] = ci[1]
        
        return pd.DataFrame([aggregated])
    
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray
    ) -> Dict:
        """
        Perform McNemar's test for comparing two models.
        
        Addresses Reviewer D-16:
        "No statistical significance tests."
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
        
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        # Count disagreements
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # Model 1 correct, Model 2 wrong
        n10 = np.sum(correct1 & ~correct2)
        # Model 1 wrong, Model 2 correct
        n01 = np.sum(~correct1 & correct2)
        
        # Create 2x2 table
        table = np.array([[0, n10], [n01, 0]])
        
        # Perform McNemar's test
        result = mcnemar(table, exact=True)
        
        return {
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'n10': int(n10),  # Model 1 better
            'n01': int(n01),  # Model 2 better
            'significant': result.pvalue < 0.05
        }
    
    def create_statistical_comparisons(self) -> pd.DataFrame:
        """
        Create statistical comparisons between all models.
        Uses paired t-test on accuracy across CV folds.
        
        Addresses Reviewer D-16: Statistical significance tests
        
        Returns:
            DataFrame with pairwise statistical comparisons
        """
        print(f"\n{'='*70}")
        print("STATISTICAL SIGNIFICANCE TESTING")
        print(f"{'='*70}")
        print("Addresses Reviewer D-16, D-26: Pairwise model comparisons")
        print("Method: Paired t-test on accuracy across 5-fold CV\n")
        
        # Load all CV results
        cv_results = {}
        model_files = list(self.metrics_dir.glob('*_cv_results.csv'))
        
        for model_file in model_files:
            if model_file.stem != 'all_models_cv_results':
                model_name = model_file.stem.replace('_cv_results', '')
                cv_results[model_name] = pd.read_csv(model_file)
                print(f"✓ Loaded CV results for: {model_name}")
        
        if len(cv_results) < 2:
            print("\n⚠️  Warning: Need at least 2 models for comparison")
            return pd.DataFrame()
        
        print(f"\nPerforming pairwise comparisons ({len(cv_results)} models)...\n")
        
        # Pairwise comparisons
        comparisons = []
        models = sorted(list(cv_results.keys()))
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Get accuracies from each fold
                acc1 = cv_results[model1]['accuracy'].values
                acc2 = cv_results[model2]['accuracy'].values
                
                # Paired t-test
                t_stat, p_val = stats.ttest_rel(acc1, acc2)
                
                # Mean difference
                mean_diff = acc1.mean() - acc2.mean()
                
                # Determine significance level
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                comparisons.append({
                    'model_1': model1,
                    'model_2': model2,
                    'mean_diff': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'significance_level': sig
                })
                
                # Print comparison
                print(f"{model1:15} vs {model2:15} | "
                      f"Δ={mean_diff:+.4f} | p={p_val:.4f} {sig}")
        
        # Create DataFrame
        comp_df = pd.DataFrame(comparisons)
        
        # Save to CSV
        comp_path = self.metrics_dir / 'statistical_comparisons.csv'
        comp_df.to_csv(comp_path, index=False)
        
        print(f"\n✅ Saved: {comp_path}")
        print(f"{'='*70}\n")
        
        # Summary statistics
        n_comparisons = len(comp_df)
        n_significant = comp_df['significant'].sum()
        
        print(f"Summary:")
        print(f"  Total comparisons: {n_comparisons}")
        print(f"  Significant (p<0.05): {n_significant} ({n_significant/n_comparisons*100:.1f}%)")
        print(f"  Non-significant: {n_comparisons - n_significant}")
        print(f"{'='*70}\n")
        
        return comp_df
    
    def compare_all_models(self, results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform pairwise statistical comparisons between all models.
        
        DEPRECATED: Use create_statistical_comparisons() instead.
        
        Args:
            results_dict: Dictionary mapping model names to CV results
        
        Returns:
            DataFrame with pairwise comparison results
        """
        print("\n⚠️  Warning: compare_all_models() is deprecated.")
        print("    Use create_statistical_comparisons() instead.\n")
        
        return self.create_statistical_comparisons()
    
    def create_summary_report(self) -> pd.DataFrame:
        """
        Create comprehensive summary report of all models.
        
        Returns:
            DataFrame with summary statistics
        """
        print(f"\n{'='*70}")
        print("Creating Summary Report")
        print(f"{'='*70}\n")
        
        # Load all CV results
        all_results = []
        
        for model_file in self.metrics_dir.glob('*_cv_results.csv'):
            if model_file.stem != 'all_models_cv_results':
                model_name = model_file.stem.replace('_cv_results', '')
                df = self.aggregate_cv_results(model_name)
                all_results.append(df)
        
        summary_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by mean accuracy (descending)
        summary_df = summary_df.sort_values('accuracy_mean', ascending=False)
        
        # Format for display
        print("Model Performance Summary (Mean ± Std):")
        print("="*70)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['model'].upper()}")
            print(f"  Accuracy:    {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
            if 'auc_mean' in row:
                print(f"  AUC:         {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
            if 'f1_macro_mean' in row:
                print(f"  F1 (macro):  {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f}")
        
        # Save summary
        summary_path = self.metrics_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✅ Summary saved to: {summary_path}")
        print(f"{'='*70}\n")
        
        return summary_df


if __name__ == '__main__':
    """
    Main evaluation execution.
    """
    print("\n" + "="*70)
    print("ECG Model Evaluation")
    print("="*70 + "\n")
    
    evaluator = ECGEvaluator('config.yaml')
    
    # Create summary report
    summary = evaluator.create_summary_report()
    
    # Create statistical comparisons
    comparisons = evaluator.create_statistical_comparisons()
    
    print("\n✅ Evaluation complete!")
    print("="*70 + "\n")