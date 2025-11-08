"""
ECG Classification Pipeline - Main Execution Script

Complete pipeline addressing all reviewer requirements.

Usage:
    python run_pipeline.py                    # Run complete pipeline
    python run_pipeline.py --step preprocess  # Run specific step

Author: Abdullah Hasan Dafa
Date: November 2025
"""

import argparse
import logging
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_utils import ECGDataManager, ECGPreprocessor, test_patient_extraction
from train import ECGTrainer
from evaluate import ECGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ECGPipeline:
    """
    Complete ECG classification pipeline.
    
    Addresses all reviewer requirements in systematic workflow.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize pipeline."""
        self.config_path = config_path
        self.data_manager = ECGDataManager(config_path)
        self.preprocessor = ECGPreprocessor(config_path)
        self.trainer = ECGTrainer(config_path)
        self.evaluator = ECGEvaluator(config_path)
        
        logger.info("ECG Pipeline initialized")
    
    def step1_create_patient_mapping(self):
        """
        Step 1: Create patient mapping from raw images.
        
        Addresses Reviewer A-4, D-8: Document patient-image relationships
        """
        print("\n" + "="*70)
        print("PIPELINE STEP 1: Create Patient Mapping")
        print("="*70)
        print("Addresses Reviewer A-4, D-8: Patient-wise data organization\n")
        
        try:
            # Test patient extraction first
            test_patient_extraction()
            
            # Create mapping
            df = self.data_manager.create_patient_mapping(save=True)
            
            print("✅ Step 1 Complete: Patient mapping created")
            return df
            
        except Exception as e:
            logger.error(f"Error in Step 1: {e}")
            raise
    
    def step2_create_splits(self):
        """
        Step 2: Create patient-wise train/val/test splits.
        
        Addresses Reviewer A-4, D-8: Patient-wise splitting (no leakage)
        """
        print("\n" + "="*70)
        print("PIPELINE STEP 2: Create Patient-Wise Splits")
        print("="*70)
        print("CRITICAL: Addresses Reviewer A-4, D-8: Prevent data leakage\n")
        
        try:
            # Load patient mapping
            df = self.data_manager.create_patient_mapping(save=False)
            
            # Create splits
            train_df, val_df, test_df = self.data_manager.create_patient_wise_splits(
                df, save=True
            )
            
            print("✅ Step 2 Complete: Patient-wise splits created")
            print("   Zero patient overlap verified between train/val/test")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            raise
    
    def step3_preprocess(self):
        """
        Step 3: Preprocess all images.
        
        Addresses Reviewer D-13, D-14, D-15: Comprehensive preprocessing
        """
        print("\n" + "="*70)
        print("PIPELINE STEP 3: Preprocess Images")
        print("="*70)
        print("Addresses Reviewer D-13, D-14, D-15:")
        print("  - Grayscale conversion (eliminate color bias)")
        print("  - CLAHE (enhance waveforms, reduce artifacts)")
        print("  - Normalization\n")
        
        try:
            # Load patient mapping with splits
            import pandas as pd
            df = pd.read_csv(self.data_manager.metadata_dir / 'patient_mapping.csv')
            
            # Preprocess each split
            for split in ['train', 'val', 'test']:
                split_df = df[df['split'] == split]
                output_dir = self.data_manager.processed_dir / split
                
                self.preprocessor.preprocess_dataset(
                    split_df,
                    output_dir,
                    desc=f"Preprocessing {split} set"
                )
            
            print("✅ Step 3 Complete: All images preprocessed")
            
        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            raise
    
    def step4_train_cv(self):
        """
        Step 4: Train all models with cross-validation.
        
        Addresses Reviewer A-7, D-26: Patient-wise CV with mean±std
        """
        print("\n" + "="*70)
        print("PIPELINE STEP 4: Train Models with Cross-Validation")
        print("="*70)
        print("Addresses Reviewer A-7, D-26:")
        print("  - 5-fold patient-wise cross-validation")
        print("  - Mean ± std reporting")
        print("  - Comprehensive hyperparameter documentation\n")
        
        try:
            # Train all models
            results = self.trainer.train_all_models()
            
            print("✅ Step 4 Complete: All models trained with CV")
            return results
            
        except Exception as e:
            logger.error(f"Error in Step 4: {e}")
            raise
    
    def step5_evaluate(self):
        """
        Step 5: Evaluate and compare models.
        
        Addresses Reviewer D-16, D-26: Statistical significance testing
        """
        print("\n" + "="*70)
        print("PIPELINE STEP 5: Evaluate and Compare Models")
        print("="*70)
        print("Addresses Reviewer D-16, D-26:")
        print("  - Comprehensive metrics")
        print("  - Statistical significance tests")
        print("  - Confidence intervals\n")
        
        try:
            # Create summary report
            summary = self.evaluator.create_summary_report()
            
            # CREATE STATISTICAL COMPARISONS
            stat_comparisons = self.evaluator.create_statistical_comparisons()
            
            print("✅ Step 5 Complete: Evaluation and statistical testing done")
            return summary, stat_comparisons
            
        except Exception as e:
            logger.error(f"Error in Step 5: {e}")
            raise
    
    def step6_visualize(self):
        """
        Step 6: Generate all figures.
        """
        print("\n" + "="*70)
        print("PIPELINE STEP 6: Generate Figures")
        print("="*70)
        print("Creating publication-ready visualizations...\n")
        
        try:
            # Import visualization module
            from visualize import create_all_figures
            
            create_all_figures(self.config_path)
            
            print("✅ Step 6 Complete: All figures generated")
            
        except Exception as e:
            logger.error(f"Error in Step 6: {e}")
            logger.warning("Continuing without visualizations...")
    
    def run_complete_pipeline(self):
        """Run complete pipeline from start to finish."""
        print("\n" + "="*70)
        print("ECG CLASSIFICATION PIPELINE - COMPLETE EXECUTION")
        print("="*70)
        print("Addresses ALL reviewer requirements systematically\n")
        
        try:
            # Step 1: Patient mapping
            self.step1_create_patient_mapping()
            
            # Step 2: Splits
            self.step2_create_splits()
            
            # Step 3: Preprocessing
            self.step3_preprocess()
            
            # Step 4: Training with CV
            self.step4_train_cv()
            
            # Step 5: Evaluation
            self.step5_evaluate()
            
            # Step 6: Visualization
            self.step6_visualize()
            
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETE!")
            print("="*70)
            print("\nResults saved in:")
            print(f"  - data/metadata/      # Patient mapping and splits")
            print(f"  - data/processed/     # Preprocessed images")
            print(f"  - results/models/     # Trained model weights")
            print(f"  - results/metrics/    # Performance metrics")
            print(f"  - results/figures/    # Publication-ready figures")
            print(f"  - pipeline.log        # Execution log")
            print("\nNext steps:")
            print("  1. Review results/metrics/summary_statistics.csv")
            print("  2. Check results/figures/ for visualizations")
            print("  3. Examine results/metrics/statistical_comparisons.csv")
            print("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='ECG Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                          # Run complete pipeline
  python run_pipeline.py --step patient_mapping   # Create patient mapping
  python run_pipeline.py --step splits            # Create splits
  python run_pipeline.py --step preprocess        # Preprocess images
  python run_pipeline.py --step train             # Train with CV
  python run_pipeline.py --step evaluate          # Evaluate models
  python run_pipeline.py --step visualize         # Generate figures
        """
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['patient_mapping', 'splits', 'preprocess', 'train', 'evaluate', 'visualize', 'all'],
        default='all',
        help='Pipeline step to execute (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ECGPipeline(args.config)
    
    # Execute requested step
    if args.step == 'all':
        pipeline.run_complete_pipeline()
    elif args.step == 'patient_mapping':
        pipeline.step1_create_patient_mapping()
    elif args.step == 'splits':
        pipeline.step2_create_splits()
    elif args.step == 'preprocess':
        pipeline.step3_preprocess()
    elif args.step == 'train':
        pipeline.step4_train_cv()
    elif args.step == 'evaluate':
        pipeline.step5_evaluate()
    elif args.step == 'visualize':
        pipeline.step6_visualize()


if __name__ == '__main__':
    main()