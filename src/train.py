"""
ECG Model Training Module with Cross-Validation

Addresses Reviewer Requirements:
- A-7, D-26: Patient-wise 5-fold cross-validation with mean±std reporting
- A-6, D-3: Complete hyperparameter documentation
- A-5: Class imbalance handling

Author: Abdullah Hasan Dafa
Date: November 2025
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import tensorflow as tf

from data_utils import ECGDataManager, ECGPreprocessor, ECGAugmentor
from models import ModelFactory, compile_model, create_callbacks

logger = logging.getLogger(__name__)


class ECGTrainer:
    """
    Handles training with patient-wise cross-validation.
    
    Addresses Reviewer A-7, D-26:
    "No cross-validation or uncertainty reporting. Use patient-wise k-fold 
    cross-validation, report mean ± SD."
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.seed = self.config['seed']
        self.n_folds = self.config['split']['n_folds']
        
        # Initialize components
        self.data_manager = ECGDataManager(config_path)
        self.preprocessor = ECGPreprocessor(config_path)
        self.augmentor = ECGAugmentor(config_path)
        self.model_factory = ModelFactory(config_path)
        
        # Results storage
        self.results_dir = Path(self.config['data']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ECGTrainer initialized with {self.n_folds}-fold CV")
    
    def load_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed images and labels.
        
        Returns:
            X: Images array (N, H, W, C)
            y: Labels array (N,)
            patient_ids: Patient IDs array (N,)
        """
        print(f"\n{'='*70}")
        print("Loading Preprocessed Data")
        print(f"{'='*70}")
        
        # Load patient mapping with splits
        df = pd.read_csv(self.data_manager.metadata_dir / 'patient_mapping.csv')
        
        # Use training data (will be split into folds)
        df_train = df[df['split'] == 'train'].copy()
        
        print(f"Loading {len(df_train)} training images for cross-validation...")
        
        X = []
        y = []
        patient_ids = []
        
        processed_dir = Path(self.config['data']['processed_dir']) / 'train'
        
        for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Loading images"):
            img_path = processed_dir / row['label'] / row['filename']
            
            # Load preprocessed image
            img = self.preprocessor.preprocess_image(str(img_path))
            
            X.append(img)
            y.append(row['label_numeric'])
            patient_ids.append(row['patient_id'])
        
        X = np.array(X)
        y = np.array(y)
        patient_ids = np.array(patient_ids)
        
        print(f"✅ Loaded: X shape={X.shape}, y shape={y.shape}")
        print(f"   Unique patients: {len(np.unique(patient_ids))}")
        print(f"   Normal: {np.sum(y==0)}, Abnormal: {np.sum(y==1)}")
        print(f"{'='*70}\n")
        
        return X, y, patient_ids
    
    def train_deep_learning_model(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold: int
    ) -> Dict:
        """
        Train a deep learning model (CNN-based).
        
        Args:
            model: Keras model
            model_name: Name of the model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            fold: Fold number
        
        Returns:
            Dictionary with training history and best metrics
        """
        # Convert grayscale to 3-channel for pretrained models
        if model_name in ['vgg16', 'resnet50', 'inception_v3', 'xception']:
            X_train = np.array([self.preprocessor.grayscale_to_3channel(img) 
                               for img in X_train])
            X_val = np.array([self.preprocessor.grayscale_to_3channel(img) 
                             for img in X_val])
        
        # Compile model
        learning_rate = self.config['training']['optimizer']['learning_rate']
        class_weights = self.data_manager.get_class_weights(y_train)
        
        compile_model(model, learning_rate, class_weights)
        
        # Create callbacks
        callbacks = create_callbacks(
            f"{model_name}_fold{fold}",
            self.config['training'],
            str(self.results_dir / 'models')
        )
        
        # Training
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        
        print(f"\n  Training {model_name} - Fold {fold}")
        print(f"    Batch size: {batch_size}, Epochs: {epochs}")
        print(f"    Learning rate: {learning_rate}")
        print(f"    Class weights: {class_weights}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )
        
        # Get best metrics
        best_epoch = np.argmax(history.history['val_auc'])
        best_metrics = {
            'loss': history.history['val_loss'][best_epoch],
            'accuracy': history.history['val_accuracy'][best_epoch],
            'auc': history.history['val_auc'][best_epoch],
            'precision': history.history['val_precision'][best_epoch],
            'recall': history.history['val_recall'][best_epoch]
        }
        
        print(f"    Best epoch: {best_epoch + 1}")
        print(f"    Val Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"    Val AUC: {best_metrics['auc']:.4f}")
        
        return {
            'history': history.history,
            'best_metrics': best_metrics,
            'best_epoch': best_epoch
        }
    
    def train_svm_model(
        self,
        svm_pipeline: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fold: int
    ) -> Dict:
        """
        Train SVM with deep feature extraction.
        
        Addresses Reviewer D-11, D-25: "Feature representation undefined"
        
        Args:
            svm_pipeline: Dictionary with feature_extractor, scaler, classifier
            X_train, y_train: Training data
            X_val, y_val: Validation data
            fold: Fold number
        
        Returns:
            Dictionary with predictions and metrics
        """
        print(f"\n  Training SVM - Fold {fold}")
        print(f"    Feature extraction: VGG16 fc2 layer (4096-dim)")
        print(f"    Addresses Reviewer D-11, D-25: Deep features documented")
        
        # Convert grayscale to 3-channel for VGG16
        X_train_3ch = np.array([self.preprocessor.grayscale_to_3channel(img) 
                               for img in X_train])
        X_val_3ch = np.array([self.preprocessor.grayscale_to_3channel(img) 
                             for img in X_val])
        
        # Preprocess for VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        X_train_vgg = preprocess_input(X_train_3ch * 255.0)
        X_val_vgg = preprocess_input(X_val_3ch * 255.0)
        
        # Extract features
        print(f"    Extracting features from {len(X_train)} training images...")
        features_train = svm_pipeline['feature_extractor'].predict(
            X_train_vgg, 
            batch_size=32, 
            verbose=0
        )
        
        print(f"    Extracting features from {len(X_val)} validation images...")
        features_val = svm_pipeline['feature_extractor'].predict(
            X_val_vgg, 
            batch_size=32, 
            verbose=0
        )
        
        # Scale features
        print(f"    Scaling features with StandardScaler...")
        features_train_scaled = svm_pipeline['scaler'].fit_transform(features_train)
        features_val_scaled = svm_pipeline['scaler'].transform(features_val)
        
        # Train SVM
        print(f"    Training SVM classifier...")
        svm_pipeline['classifier'].fit(features_train_scaled, y_train)
        
        # Predictions
        y_pred_train = svm_pipeline['classifier'].predict(features_train_scaled)
        y_pred_val = svm_pipeline['classifier'].predict(features_val_scaled)
        
        # Probabilities for AUC
        y_prob_train = svm_pipeline['classifier'].predict_proba(features_train_scaled)[:, 1]
        y_prob_val = svm_pipeline['classifier'].predict_proba(features_val_scaled)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        train_auc = roc_auc_score(y_train, y_prob_train)
        val_auc = roc_auc_score(y_val, y_prob_val)
        
        print(f"    Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"    Val Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Save features and model
        features_dir = self.results_dir / 'svm_features' / f'fold{fold}'
        features_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(features_dir / 'features_train.npy', features_train_scaled)
        np.save(features_dir / 'features_val.npy', features_val_scaled)
        
        with open(features_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(svm_pipeline['scaler'], f)
        
        with open(features_dir / 'svm_classifier.pkl', 'wb') as f:
            pickle.dump(svm_pipeline['classifier'], f)
        
        return {
            'predictions': {
                'train': y_pred_train,
                'val': y_pred_val,
                'train_proba': y_prob_train,
                'val_proba': y_prob_val
            },
            'best_metrics': {
                'accuracy': val_acc,
                'auc': val_auc
            }
        }
    
    def train_with_cross_validation(self, model_name: str) -> pd.DataFrame:
        """
        Train model with patient-wise cross-validation.
        
        Addresses Reviewer A-7, D-26:
        "Use patient-wise k-fold cross-validation, report mean ± SD"
        
        Args:
            model_name: Name of model to train
        
        Returns:
            DataFrame with results from all folds
        """
        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()} with {self.n_folds}-Fold Cross-Validation")
        print(f"{'='*70}")
        print(f"Addresses Reviewer A-7, D-26: Patient-wise CV with mean±std reporting\n")
        
        # Load data
        X, y, patient_ids = self.load_preprocessed_data()
        
        # Initialize StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed
        )
        
        # Results storage
        fold_results = []
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=patient_ids)):
            print(f"\n{'='*70}")
            print(f"Fold {fold + 1}/{self.n_folds}")
            print(f"{'='*70}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            patients_train = patient_ids[train_idx]
            patients_val = patient_ids[val_idx]
            
            # Verify no patient overlap (CRITICAL)
            overlap = set(patients_train) & set(patients_val)
            if overlap:
                raise ValueError(f"Patient overlap detected in fold {fold}: {len(overlap)} patients!")
            
            print(f"  Train: {len(X_train)} images from {len(np.unique(patients_train))} patients")
            print(f"  Val: {len(X_val)} images from {len(np.unique(patients_val))} patients")
            print(f"  ✅ No patient overlap verified")
            
            # Determine input shape
            if model_name == 'custom_cnn':
                input_shape = (224, 224, 1)
            elif model_name in ['vgg16', 'resnet50', 'inception_v3', 'xception']:
                input_shape = (224, 224, 3)
            else:
                input_shape = (224, 224, 1)
            
            # Create model
            model = self.model_factory.create_model(model_name, input_shape)
            
            # Train
            if model_name == 'svm':
                results = self.train_svm_model(
                    model, X_train, y_train, X_val, y_val, fold
                )
            else:
                results = self.train_deep_learning_model(
                    model, model_name, X_train, y_train, X_val, y_val, fold
                )
            
            # Store results
            fold_result = {
                'model': model_name,
                'fold': fold,
                'n_train': len(X_train),
                'n_val': len(X_val),
                **results['best_metrics']
            }
            fold_results.append(fold_result)
            
            print(f"\n  Fold {fold + 1} Complete:")
            print(f"    Accuracy: {results['best_metrics']['accuracy']:.4f}")
            print(f"    AUC: {results['best_metrics']['auc']:.4f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(fold_results)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Cross-Validation Summary for {model_name.upper()}")
        print(f"{'='*70}")
        print(f"\nPer-Fold Results:")
        print(results_df.to_string(index=False))
        
        # Calculate mean ± std (Reviewer D-26)
        metrics = ['accuracy', 'auc']
        if 'precision' in results_df.columns:
            metrics.extend(['precision', 'recall'])
        
        print(f"\nMean ± Std (Addresses Reviewer D-26):")
        for metric in metrics:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                print(f"  {metric.capitalize():12}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save results
        results_path = self.results_dir / 'metrics' / f'{model_name}_cv_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        print(f"{'='*70}\n")
        
        return results_df
    
    def train_all_models(self) -> Dict[str, pd.DataFrame]:
        """
        Train all enabled models with cross-validation.
        
        Returns:
            Dictionary mapping model names to results DataFrames
        """
        enabled_models = self.config['models']['enabled_models']
        
        print(f"\n{'='*70}")
        print(f"Training All Models")
        print(f"{'='*70}")
        print(f"Models to train: {', '.join(enabled_models)}")
        print(f"Cross-validation: {self.n_folds} folds (patient-wise)")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for model_name in enabled_models:
            try:
                results_df = self.train_with_cross_validation(model_name)
                all_results[model_name] = results_df
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                raise
        
        # Save combined results
        combined_results = pd.concat(all_results.values(), ignore_index=True)
        combined_path = self.results_dir / 'metrics' / 'all_models_cv_results.csv'
        combined_results.to_csv(combined_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"All Models Training Complete")
        print(f"{'='*70}")
        print(f"✅ Combined results saved to: {combined_path}")
        print(f"{'='*70}\n")
        
        return all_results


if __name__ == '__main__':
    """
    Main execution for training.
    """
    print("\n" + "="*70)
    print("ECG Model Training with Cross-Validation")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = ECGTrainer('config.yaml')
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\n✅ Training pipeline complete!")
    print("="*70 + "\n")