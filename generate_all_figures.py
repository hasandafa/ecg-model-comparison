"""
ECG Classification - Complete Figure Generation Script
Generate ALL publication-ready figures addressing reviewer requirements

Addresses:
- Reviewer A-8: Professional figures (no code screenshots)
- Reviewer D-19: Proper figure numbering and labeling
- All reviewers: Publication-quality visualizations (300 DPI)

Author: Abdullah Hasan Dafa
Date: November 2025
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import cv2
from scipy import stats

# Set publication-quality parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Colors for consistency
COLORS = {
    'normal': '#2ecc71',
    'abnormal': '#e74c3c',
    'svm': '#3498db',
    'custom_cnn': '#9b59b6',
    'vgg16': '#e67e22',
    'resnet50': '#1abc9c',
    'inception_v3': '#f39c12',
    'xception': '#e91e63'
}

MODEL_NAMES = {
    'svm': 'SVM',
    'custom_cnn': 'Custom CNN',
    'vgg16': 'VGG16',
    'resnet50': 'ResNet50',
    'inception_v3': 'InceptionV3',
    'xception': 'Xception'
}


class FigureGenerator:
    """Generate all publication-ready figures"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize figure generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.data_dir = Path('data')
        self.results_dir = Path('results')
        self.figures_dir = self.results_dir / 'figures'
        self.metrics_dir = self.results_dir / 'metrics'
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("ECG CLASSIFICATION - FIGURE GENERATION")
        print(f"{'='*70}\n")
        print(f"Output directory: {self.figures_dir}")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # FIGURE 1: Dataset Overview and Distribution
    # =========================================================================
    
    def fig1_dataset_distribution(self):
        """
        Figure 1: Dataset Distribution
        - Overall class distribution (bar chart)
        - Distribution per split (grouped bar chart)
        - Patient vs images comparison
        
        Addresses Reviewer A-3, D-10: Clear dataset documentation
        """
        print("Generating Figure 1: Dataset Distribution...")
        
        # Load patient mapping
        df = pd.read_csv(self.data_dir / 'metadata' / 'patient_mapping.csv')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel A: Overall distribution
        ax = axes[0]
        counts = df['label'].value_counts()
        bars = ax.bar(
            ['Normal', 'Abnormal'],
            [counts['normal'], counts['abnormal']],
            color=[COLORS['normal'], COLORS['abnormal']],
            edgecolor='black',
            linewidth=1.5
        )
        ax.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
        ax.set_title('(A) Overall Dataset Distribution', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 300)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Panel B: Distribution per split
        ax = axes[1]
        split_data = df.groupby(['split', 'label']).size().unstack()
        split_data = split_data.reindex(['train', 'val', 'test'])
        
        x = np.arange(len(split_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, split_data['normal'], width,
                      label='Normal', color=COLORS['normal'],
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, split_data['abnormal'], width,
                      label='Abnormal', color=COLORS['abnormal'],
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
        ax.set_title('(B) Distribution by Data Split', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Train', 'Validation', 'Test'], fontsize=12)
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
        
        # Panel C: Patients vs Images
        ax = axes[2]
        
        # Get patient counts
        patient_stats = []
        for label in ['normal', 'abnormal']:
            df_label = df[df['label'] == label]
            n_images = len(df_label)
            n_patients = df_label['patient_id'].nunique()
            patient_stats.append({
                'label': label.capitalize(),
                'images': n_images,
                'patients': n_patients
            })
        
        stats_df = pd.DataFrame(patient_stats)
        
        x = np.arange(len(stats_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, stats_df['patients'], width,
                      label='Unique Patients', color='#95a5a6',
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, stats_df['images'], width,
                      label='Total Images', color='#34495e',
                      edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.set_title('(C) Patients vs Images', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['label'], fontsize=12)
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = self.figures_dir / 'Figure_1_Dataset_Distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # FIGURE 2: Patient Distribution Analysis
    # =========================================================================
    
    def fig2_patient_distribution(self):
        """
        Figure 2: Patient Distribution Analysis
        - Histogram of images per patient
        - Cumulative distribution
        
        Addresses Reviewer A-4, D-8: Patient-wise splitting justification
        """
        print("Generating Figure 2: Patient Distribution Analysis...")
        
        df = pd.read_csv(self.data_dir / 'metadata' / 'patient_mapping.csv')
        
        # Count images per patient
        images_per_patient = df.groupby('patient_id').size()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel A: Histogram
        ax = axes[0]
        ax.hist(images_per_patient, bins=range(1, images_per_patient.max()+2),
               color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Number of Images per Patient', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Patients', fontsize=14, fontweight='bold')
        ax.set_title('(A) Distribution of Images per Patient', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add statistics text
        stats_text = f"Mean: {images_per_patient.mean():.2f}\n"
        stats_text += f"Median: {images_per_patient.median():.0f}\n"
        stats_text += f"Max: {images_per_patient.max()}\n"
        stats_text += f"Total Patients: {len(images_per_patient)}"
        
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel B: Cumulative distribution
        ax = axes[1]
        sorted_counts = np.sort(images_per_patient.values)
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
        
        ax.plot(sorted_counts, cumulative, 'o-', color='#e74c3c',
               linewidth=2, markersize=4, alpha=0.7)
        ax.set_xlabel('Number of Images per Patient', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=14, fontweight='bold')
        ax.set_title('(B) Cumulative Distribution', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add reference lines
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50th percentile')
        ax.axhline(y=75, color='gray', linestyle=':', alpha=0.5, label='75th percentile')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.figures_dir / 'Figure_2_Patient_Distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # FIGURE 3: Preprocessing Pipeline Flowchart
    # =========================================================================
    
    def fig3_preprocessing_pipeline(self):
        """
        Figure 3: Preprocessing Pipeline
        Show original → grayscale → CLAHE → normalized
        
        Addresses Reviewer D-13, D-14, D-15: Preprocessing documentation
        """
        print("Generating Figure 3: Preprocessing Pipeline...")
        
        # Load a sample image
        sample_normal = list((self.data_dir / 'raw' / 'normal').glob('*.jpg'))[0]
        
        # Read image
        img_original = cv2.imread(str(sample_normal))
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        # Step 1: Grayscale
        img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        # Step 2: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        
        # Step 3: Resize
        img_resized = cv2.resize(img_clahe, (224, 224))
        
        # Step 4: Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original RGB
        axes[0, 0].imshow(img_original_rgb)
        axes[0, 0].set_title('(A) Original RGB Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 0].text(0.5, -0.05, 'Input: Color ECG Chart',
                       transform=axes[0, 0].transAxes,
                       ha='center', fontsize=12, style='italic')
        
        # Grayscale
        axes[0, 1].imshow(img_gray, cmap='gray')
        axes[0, 1].set_title('(B) Grayscale Conversion', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        axes[0, 1].text(0.5, -0.05, 'Eliminates color bias (Reviewer D-14)',
                       transform=axes[0, 1].transAxes,
                       ha='center', fontsize=12, style='italic', color='#e74c3c')
        
        # CLAHE
        axes[0, 2].imshow(img_clahe, cmap='gray')
        axes[0, 2].set_title('(C) CLAHE Enhancement', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        axes[0, 2].text(0.5, -0.05, 'Enhances waveform visibility',
                       transform=axes[0, 2].transAxes,
                       ha='center', fontsize=12, style='italic')
        
        # Resized
        axes[1, 0].imshow(img_resized, cmap='gray')
        axes[1, 0].set_title('(D) Resized to 224×224', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, -0.05, 'Standard CNN input size',
                       transform=axes[1, 0].transAxes,
                       ha='center', fontsize=12, style='italic')
        
        # Normalized
        axes[1, 1].imshow(img_normalized, cmap='gray')
        axes[1, 1].set_title('(E) Normalized [0, 1]', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, -0.05, 'Final preprocessed image',
                       transform=axes[1, 1].transAxes,
                       ha='center', fontsize=12, style='italic')
        
        # Flowchart
        axes[1, 2].text(0.5, 0.9, 'Preprocessing Pipeline', 
                       ha='center', fontsize=16, fontweight='bold',
                       transform=axes[1, 2].transAxes)
        
        pipeline_text = """
1. Load RGB Image
   ↓
2. Convert to Grayscale
   (Eliminate color bias)
   ↓
3. Apply CLAHE
   (clip_limit=2.0)
   ↓
4. Resize to 224×224
   ↓
5. Normalize to [0, 1]
   ↓
6. Ready for Model Input
"""
        axes[1, 2].text(0.5, 0.5, pipeline_text,
                       ha='center', va='center',
                       fontsize=12, family='monospace',
                       transform=axes[1, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        axes[1, 2].axis('off')
        
        plt.suptitle('Figure 3: Complete Preprocessing Pipeline', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.figures_dir / 'Figure_3_Preprocessing_Pipeline.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # FIGURE 4: Data Augmentation Examples
    # =========================================================================
    
    def fig4_augmentation_examples(self):
        """
        Figure 4: Data Augmentation Examples
        Show original + 5 augmented versions
        
        Addresses Reviewer A-5: Data augmentation documentation
        """
        print("Generating Figure 4: Data Augmentation Examples...")
        
        # Load a sample
        sample_img = list((self.data_dir / 'processed' / 'train' / 'normal').glob('*.jpg'))[0]
        img = cv2.imread(str(sample_img), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # Augmentation functions
        def rotate(img, angle):
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        def translate(img, tx, ty):
            h, w = img.shape
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        def zoom(img, factor):
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), 0, factor)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        def add_noise(img, std=0.02):
            noise = np.random.normal(0, std, img.shape)
            return np.clip(img + noise, 0, 1)
        
        # Generate augmentations
        aug_examples = [
            ('Original', img),
            ('Rotation (+5°)', rotate(img, 5)),
            ('Translation', translate(img, 10, 5)),
            ('Zoom (1.05×)', zoom(img, 1.05)),
            ('Gaussian Noise', add_noise(img)),
            ('Combined', add_noise(zoom(rotate(img, 3), 1.03)))
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (title, aug_img) in enumerate(aug_examples):
            axes[idx].imshow(aug_img, cmap='gray')
            axes[idx].set_title(f'({chr(65+idx)}) {title}', 
                               fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            
            # Add border for original
            if idx == 0:
                for spine in axes[idx].spines.values():
                    spine.set_edgecolor('#e74c3c')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
        
        plt.suptitle('Figure 4: Data Augmentation Techniques\n'
                    'Addresses Class Imbalance (Reviewer A-5)',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.figures_dir / 'Figure_4_Augmentation_Examples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # FIGURE 5: Model Performance Comparison
    # =========================================================================
    
    def fig5_performance_comparison(self):
        """
        Figure 5: Overall Model Performance Comparison
        Bar chart with error bars (mean ± std)
        
        Addresses Reviewer A-7, D-26: Mean±std reporting
        """
        print("Generating Figure 5: Model Performance Comparison...")
        
        # Load summary statistics
        summary = pd.read_csv(self.metrics_dir / 'summary_statistics.csv')
        
        # Prepare data
        models = [MODEL_NAMES.get(m, m) for m in summary['model']]
        metrics_to_plot = ['accuracy', 'auc']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            means = summary[f'{metric}_mean'].values * (100 if metric == 'accuracy' else 1)
            stds = summary[f'{metric}_std'].values * (100 if metric == 'accuracy' else 1)
            
            colors_list = [COLORS.get(m.lower().replace(' ', '_'), '#95a5a6') 
                          for m in summary['model']]
            
            bars = ax.bar(models, means, yerr=stds, capsize=8,
                         color=colors_list, edgecolor='black', linewidth=1.5,
                         alpha=0.8, error_kw={'linewidth': 2, 'ecolor': 'black'})
            
            # Labels
            if metric == 'accuracy':
                ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
                ax.set_title(f'(A) Model Accuracy\n(Mean ± SD across 5 folds)',
                            fontsize=16, fontweight='bold')
                ax.set_ylim(0, 100)
            else:
                ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
                ax.set_title(f'(B) Model AUC\n(Mean ± SD across 5 folds)',
                            fontsize=16, fontweight='bold')
                ax.set_ylim(0, 1)
            
            ax.set_xlabel('Model', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                if metric == 'accuracy':
                    label = f'{mean:.1f}±{std:.1f}%'
                else:
                    label = f'{mean:.3f}±{std:.3f}'
                
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Figure 5: Model Performance Comparison\n'
                    'Patient-wise 5-Fold Cross-Validation (Reviewer D-26)',
                    fontsize=18, fontweight='bold', y=1.00)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.figures_dir / 'Figure_5_Performance_Comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # FIGURE 6: Cross-Validation Consistency (Box Plots)
    # =========================================================================
    
    def fig6_cv_consistency(self):
        """
        Figure 6: Cross-Validation Fold Consistency
        Box plots showing performance distribution across folds
        
        Addresses Reviewer D-7: Variance analysis
        """
        print("Generating Figure 6: Cross-Validation Consistency...")
        
        # Load all CV results
        cv_results = pd.read_csv(self.metrics_dir / 'all_models_cv_results.csv')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data for box plots
        metrics = ['accuracy', 'auc']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data
            data_to_plot = []
            labels = []
            colors_list = []
            
            for model in cv_results['model'].unique():
                model_data = cv_results[cv_results['model'] == model][metric].dropna()
                if len(model_data) > 0:
                    if metric == 'accuracy':
                        model_data = model_data * 100
                    data_to_plot.append(model_data)
                    labels.append(MODEL_NAMES.get(model, model))
                    colors_list.append(COLORS.get(model, '#95a5a6'))
            
            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True,
                           medianprops=dict(color='red', linewidth=2),
                           meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Labels
            if metric == 'accuracy':
                ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
                ax.set_title('(A) Accuracy Distribution Across Folds',
                            fontsize=16, fontweight='bold')
                ax.set_ylim(0, 100)
            else:
                ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
                ax.set_title('(B) AUC Distribution Across Folds',
                            fontsize=16, fontweight='bold')
                ax.set_ylim(0, 1)
            
            ax.set_xlabel('Model', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.tick_params(axis='x', rotation=45)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=2, label='Median'),
                Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        
        plt.suptitle('Figure 6: Cross-Validation Consistency Analysis\n'
                    'Performance Variance Across 5 Folds (Reviewer D-7)',
                    fontsize=18, fontweight='bold', y=1.00)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.figures_dir / 'Figure_6_CV_Consistency.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # FIGURE 7: Statistical Significance Heatmap
    # =========================================================================
    
    def fig7_statistical_significance(self):
        """
        Figure 7: Statistical Significance Testing
        Heatmap of p-values from pairwise comparisons
        
        Addresses Reviewer D-16: Statistical significance tests
        """
        print("Generating Figure 7: Statistical Significance Heatmap...")
        
        # Load statistical comparisons
        try:
            comparisons = pd.read_csv(self.metrics_dir / 'statistical_comparisons.csv')
        except:
            print("⚠️  Statistical comparisons not found, skipping...\n")
            return
        
        # Get unique models
        all_models = list(set(comparisons['model_1'].tolist() + 
                             comparisons['model_2'].tolist()))
        all_models = sorted(all_models)
        
        # Create p-value matrix
        n = len(all_models)
        p_matrix = np.ones((n, n))
        
        for _, row in comparisons.iterrows():
            i = all_models.index(row['model_1'])
            j = all_models.index(row['model_2'])
            p_matrix[i, j] = row['p_value']
            p_matrix[j, i] = row['p_value']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=0.1)
        
        # Set ticks
        model_labels = [MODEL_NAMES.get(m, m) for m in all_models]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.set_yticklabels(model_labels)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                if i != j:
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        text = f'{p_val:.4f}\n***'
                        color = 'white'
                    elif p_val < 0.01:
                        text = f'{p_val:.4f}\n**'
                        color = 'white'
                    elif p_val < 0.05:
                        text = f'{p_val:.4f}\n*'
                        color = 'black'
                    else:
                        text = f'{p_val:.4f}\nns'
                        color = 'black'
                    
                    ax.text(j, i, text, ha='center', va='center',
                           color=color, fontsize=9, fontweight='bold')
                else:
                    ax.text(j, i, '—', ha='center', va='center',
                           color='gray', fontsize=16, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('P-value (McNemar Test)', fontsize=12, fontweight='bold')
        
        # Title
        ax.set_title('Figure 7: Statistical Significance Matrix\n'
                    'Pairwise Model Comparisons (McNemar\'s Test)\n'
                    '*p<0.05, **p<0.01, ***p<0.001, ns=not significant',
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = self.figures_dir / 'Figure_7_Statistical_Significance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}\n")
    
    # =========================================================================
    # INDIVIDUAL MODEL FIGURES
    # =========================================================================
    
    def generate_individual_model_figures(self):
        """
        Generate individual figures for each model:
        - Training curves (loss & accuracy)
        - Confusion matrix
        - ROC curve
        
        Each model gets its own figure
        """
        print("\nGenerating Individual Model Figures...")
        print("="*70)
        
        models = ['svm', 'custom_cnn', 'vgg16', 'resnet50', 'inception_v3', 'xception']
        
        for model in models:
            self._generate_single_model_figure(model)
    
    def _generate_single_model_figure(self, model_name):
        """Generate comprehensive figure for a single model"""
        print(f"\nGenerating Figure for {MODEL_NAMES.get(model_name, model_name)}...")
        
        # Load CV results
        cv_file = self.metrics_dir / f'{model_name}_cv_results.csv'
        if not cv_file.exists():
            print(f"⚠️  Results not found for {model_name}, skipping...")
            return
        
        cv_results = pd.read_csv(cv_file)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # Panel A: Fold-wise Performance
        ax1 = fig.add_subplot(gs[0, 0])
        folds = cv_results['fold'].values
        accuracy = cv_results['accuracy'].values * 100
        auc = cv_results['auc'].values
        
        ax1.plot(folds, accuracy, 'o-', label='Accuracy', 
                color=COLORS.get(model_name, '#3498db'),
                linewidth=2, markersize=8)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(folds, auc, 's--', label='AUC',
                     color='#e74c3c', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=COLORS.get(model_name, '#3498db'))
        ax1_twin.set_ylabel('AUC Score', fontsize=12, fontweight='bold', color='#e74c3c')
        ax1.set_title('(A) Performance Across Folds', fontsize=14, fontweight='bold')
        ax1.set_xticks(folds)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor=COLORS.get(model_name, '#3498db'))
        ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
        
        # Add mean lines
        ax1.axhline(y=accuracy.mean(), color=COLORS.get(model_name, '#3498db'),
                   linestyle=':', alpha=0.5, label=f'Mean Acc: {accuracy.mean():.1f}%')
        ax1_twin.axhline(y=auc.mean(), color='#e74c3c',
                        linestyle=':', alpha=0.5, label=f'Mean AUC: {auc.mean():.3f}')
        
        # Panel B: Metrics Summary
        ax2 = fig.add_subplot(gs[0, 1])
        
        metrics_data = []
        if 'precision' in cv_results.columns:
            metrics_data.append(('Accuracy', accuracy.mean(), accuracy.std()))
            metrics_data.append(('Precision', cv_results['precision'].mean()*100, 
                                cv_results['precision'].std()*100))
            metrics_data.append(('Recall', cv_results['recall'].mean()*100,
                                cv_results['recall'].std()*100))
            metrics_data.append(('AUC', auc.mean()*100, auc.std()*100))
        else:
            metrics_data.append(('Accuracy', accuracy.mean(), accuracy.std()))
            metrics_data.append(('AUC', auc.mean()*100, auc.std()*100))
        
        metric_names = [m[0] for m in metrics_data]
        metric_means = [m[1] for m in metrics_data]
        metric_stds = [m[2] for m in metrics_data]
        
        bars = ax2.barh(metric_names, metric_means, xerr=metric_stds,
                       color=COLORS.get(model_name, '#3498db'),
                       edgecolor='black', linewidth=1.5, alpha=0.7,
                       capsize=5, error_kw={'linewidth': 2})
        
        ax2.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('(B) Average Metrics (Mean ± SD)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(0, 100)
        
        # Add value labels
        for bar, mean, std in zip(bars, metric_means, metric_stds):
            width = bar.get_width()
            ax2.text(width + std, bar.get_y() + bar.get_height()/2,
                    f'{mean:.1f}±{std:.1f}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Panel C: Model Info
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # Model information
        info_text = f"Model: {MODEL_NAMES.get(model_name, model_name)}\n"
        info_text += "="*40 + "\n\n"
        
        if model_name == 'svm':
            info_text += "Architecture:\n"
            info_text += "• Feature Extraction: VGG16 fc2\n"
            info_text += "• Feature Dimension: 4096\n"
            info_text += "• Scaler: StandardScaler\n"
            info_text += "• Kernel: RBF\n"
            info_text += "• C: 10.0\n"
            info_text += "• Gamma: scale\n\n"
        elif model_name == 'custom_cnn':
            info_text += "Architecture:\n"
            info_text += "• Conv2D(32) + MaxPool + Dropout(0.25)\n"
            info_text += "• Conv2D(64) + MaxPool + Dropout(0.25)\n"
            info_text += "• Conv2D(128) + MaxPool + Dropout(0.25)\n"
            info_text += "• Dense(256) + Dropout(0.5)\n"
            info_text += "• Output: Dense(1, sigmoid)\n\n"
        else:
            info_text += "Architecture:\n"
            info_text += f"• Base: {MODEL_NAMES.get(model_name, model_name)}\n"
            info_text += "• Pretrained: ImageNet\n"
            info_text += "• Frozen: Convolutional base\n"
            info_text += "• GlobalAveragePooling2D\n"
            info_text += "• Dense(256, relu) + Dropout(0.5)\n"
            info_text += "• Output: Dense(1, sigmoid)\n\n"
        
        info_text += "Training:\n"
        info_text += f"• Validation: 5-fold patient-wise CV\n"
        info_text += f"• Optimizer: Adam (lr=0.0001)\n"
        info_text += f"• Batch Size: {self.config['training']['batch_size']}\n"
        info_text += f"• Class Weights: Balanced\n\n"
        
        info_text += "Performance:\n"
        info_text += f"• Accuracy: {accuracy.mean():.2f} ± {accuracy.std():.2f}%\n"
        info_text += f"• AUC: {auc.mean():.3f} ± {auc.std():.3f}\n"
        
        ax3.text(0.05, 0.95, info_text,
                transform=ax3.transAxes,
                fontsize=11, family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
        
        # Overall title
        fig.suptitle(f'Figure: {MODEL_NAMES.get(model_name, model_name)} - Complete Analysis',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = self.figures_dir / f'Figure_Model_{MODEL_NAMES.get(model_name, model_name).replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def generate_all(self):
        """Generate all figures"""
        print("\n" + "="*70)
        print("GENERATING ALL FIGURES")
        print("="*70 + "\n")
        
        try:
            # Main figures
            self.fig1_dataset_distribution()
            self.fig2_patient_distribution()
            self.fig3_preprocessing_pipeline()
            self.fig4_augmentation_examples()
            self.fig5_performance_comparison()
            self.fig6_cv_consistency()
            self.fig7_statistical_significance()
            
            # Individual model figures
            self.generate_individual_model_figures()
            
            print("\n" + "="*70)
            print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
            print("="*70)
            print(f"\nOutput directory: {self.figures_dir}")
            print(f"Total figures created: {len(list(self.figures_dir.glob('*.png')))}")
            print("\nAll figures are:")
            print("✓ 300 DPI (publication quality)")
            print("✓ Properly labeled and numbered")
            print("✓ Addressing specific reviewer requirements")
            print("✓ Individual files (not combined)")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error generating figures: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution"""
    generator = FigureGenerator()
    generator.generate_all()


if __name__ == '__main__':
    main()
