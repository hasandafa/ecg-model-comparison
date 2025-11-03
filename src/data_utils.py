"""
ECG Data Utilities Module

Handles all data-related operations with focus on addressing reviewer requirements:
- Reviewer A-4, D-8: Patient-wise splitting (no data leakage)
- Reviewer D-14: Grayscale conversion (eliminate color bias)
- Reviewer D-13, D-15: Comprehensive preprocessing pipeline
- Reviewer A-5: Data augmentation for small dataset

Author: ECG Classification Team
Date: 2025
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ECGDataManager:
    """
    Manages ECG dataset operations with patient-wise integrity.
    
    Addresses Reviewer Requirements:
    - A-4, D-8: Patient-wise splitting to prevent data leakage
    - D-18: Complete reproducibility
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize data manager with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.seed = self.config['seed']
        self._set_seeds()
        
        # Setup paths
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.metadata_dir = Path(self.config['data']['metadata_dir'])
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ECGDataManager initialized with seed={self.seed}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError(
                    f"❌ Configuration file is empty or invalid: {config_path}\n"
                    f"💡 Tip: Make sure config.yaml has valid YAML content.\n"
                    f"   The file should start with '# ECG Configuration' and contain key-value pairs."
                )
            
            # Verify essential keys
            required_keys = ['seed', 'data', 'split', 'preprocessing', 'training', 'models']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                raise ValueError(
                    f"❌ Missing required configuration keys: {missing_keys}\n"
                    f"💡 Tip: Check your config.yaml file has all required sections."
                )
            
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            raise ValueError(
                f"❌ Invalid YAML syntax in {config_path}\n"
                f"Error: {e}\n"
                f"💡 Tip: Check for proper indentation and no tabs in YAML file."
            )
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            raise
    
    def _set_seeds(self):
        """
        Set random seeds for reproducibility.
        
        Addresses Reviewer D-18: Reproducibility requirements
        """
        import random
        import tensorflow as tf
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        logger.info(f"✅ All random seeds set to {self.seed}")
    
    def extract_patient_id(self, filename: str) -> str:
        """
        Extract patient ID from filename.
        
        Handles multiple filename patterns:
        - Ahmad_1.jpg → "Ahmad"
        - Ahmad_2.jpg → "Ahmad" (same patient)
        - 15_1.jpg → "15"
        - 15_2.jpg → "15" (same patient)
        - Ali Suryana_2.jpg → "Ali Suryana"
        - 4_1(1).jpg → "4_1"
        - Marjuki(1)_2.jpg → "Marjuki(1)"
        
        Args:
            filename: Image filename
        
        Returns:
            patient_id: Extracted patient identifier
        """
        # Remove file extension
        name = Path(filename).stem
        
        # Remove trailing: _digit or _digit(digit)
        # Pattern: _\d+(\(\d+\))?$ means:
        # _ followed by digits, optionally followed by (digits), at end of string
        patient_id = re.sub(r'_\d+(\(\d+\))?$', '', name)
        
        # Special case: if format is digit_digit (e.g., 15_1)
        # Extract only the first number as patient ID
        if re.match(r'^\d+_\d+', name):
            patient_id = name.split('_')[0]
        
        # Clean up any trailing underscores
        patient_id = patient_id.strip('_')
        
        return patient_id
    
    def create_patient_mapping(self, save: bool = True) -> pd.DataFrame:
        """
        Create patient mapping from raw images.
        
        Addresses Reviewer A-4, D-8: Document patient-image relationships
        
        Returns:
            DataFrame with columns: filename, patient_id, label, label_numeric, path
        """
        print(f"\n{'='*70}")
        print("STEP 1: Creating Patient Mapping")
        print(f"{'='*70}")
        
        records = []
        
        # Process normal images
        normal_dir = self.raw_dir / self.config['data']['normal_folder']
        if not normal_dir.exists():
            raise FileNotFoundError(
                f"❌ Normal images folder not found: {normal_dir}\n"
                f"💡 Tip: Create folder structure:\n"
                f"   data/raw/normal/  (place 285 normal ECG images here)\n"
                f"   data/raw/abnormal/  (place 201 abnormal ECG images here)"
            )
        
        normal_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg')) + list(normal_dir.glob('*.png'))
        print(f"Processing normal folder: {len(normal_files)} images")
        
        for img_path in tqdm(normal_files, desc="Processing normal images"):
            filename = img_path.name
            patient_id = self.extract_patient_id(filename)
            records.append({
                'filename': filename,
                'patient_id': patient_id,
                'label': 'normal',
                'label_numeric': 0,
                'path': str(img_path)
            })
        
        # Process abnormal images
        abnormal_dir = self.raw_dir / self.config['data']['abnormal_folder']
        if not abnormal_dir.exists():
            raise FileNotFoundError(
                f"❌ Abnormal images folder not found: {abnormal_dir}\n"
                f"💡 Tip: Create folder data/raw/abnormal/ and place 201 abnormal ECG images"
            )
        
        abnormal_files = list(abnormal_dir.glob('*.jpg')) + list(abnormal_dir.glob('*.jpeg')) + list(abnormal_dir.glob('*.png'))
        print(f"Processing abnormal folder: {len(abnormal_files)} images")
        
        for img_path in tqdm(abnormal_files, desc="Processing abnormal images"):
            filename = img_path.name
            patient_id = self.extract_patient_id(filename)
            records.append({
                'filename': filename,
                'patient_id': patient_id,
                'label': 'abnormal',
                'label_numeric': 1,
                'path': str(img_path)
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Statistics
        total_images = len(df)
        n_patients = df['patient_id'].nunique()
        n_normal = len(df[df['label'] == 'normal'])
        n_abnormal = len(df[df['label'] == 'abnormal'])
        n_patients_normal = df[df['label'] == 'normal']['patient_id'].nunique()
        n_patients_abnormal = df[df['label'] == 'abnormal']['patient_id'].nunique()
        avg_images_per_patient = total_images / n_patients
        
        print(f"\n✅ Patient mapping created:")
        print(f"   Total images: {total_images}")
        print(f"   Unique patients: {n_patients}")
        print(f"   Images per patient (avg): {avg_images_per_patient:.2f}")
        print(f"   Normal: {n_normal} images ({n_patients_normal} patients)")
        print(f"   Abnormal: {n_abnormal} images ({n_patients_abnormal} patients)")
        
        # Show patients with multiple images
        multi_image_patients = df.groupby('patient_id').size()
        multi_image_patients = multi_image_patients[multi_image_patients > 1]
        print(f"\n📊 Patients with multiple images: {len(multi_image_patients)}")
        print(f"   Max images per patient: {multi_image_patients.max()}")
        
        if save:
            output_path = self.metadata_dir / 'patient_mapping.csv'
            df.to_csv(output_path, index=False)
            print(f"✅ Saved to: {output_path}")
        
        print(f"{'='*70}\n")
        
        return df
    
    def create_patient_wise_splits(
        self, 
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create patient-wise train/val/test splits.
        
        CRITICAL: Addresses Reviewer A-4, D-8
        "A random 80/20 image split can place images from the same patient 
        in both train and test, inflating estimates."
        
        Ensures:
        - No patient appears in multiple splits
        - All images from same patient stay together
        - Stratified by label (balanced normal/abnormal ratio)
        - Uses ALL 486 images (no deduplication)
        
        Args:
            df: Patient mapping DataFrame
            test_size: Proportion for test set (default 0.15 = 15%)
            val_size: Proportion of remaining for validation (default 0.15)
            save: Whether to save split assignments
        
        Returns:
            train_df, val_df, test_df
        """
        print(f"\n{'='*70}")
        print("STEP 2: Creating Patient-Wise Splits")
        print(f"{'='*70}")
        print(f"Addresses Reviewer A-4, D-8: Patient-wise splitting to prevent data leakage")
        print(f"Strategy: All images from same patient go to SAME split\n")
        
        # Get unique patients with their labels
        patient_labels = df.groupby('patient_id')['label_numeric'].first().reset_index()
        patient_labels.columns = ['patient_id', 'label']
        
        # First split: separate test set by patients
        patients_train_val, patients_test, labels_train_val, labels_test = train_test_split(
            patient_labels['patient_id'].values,
            patient_labels['label'].values,
            test_size=test_size,
            stratify=patient_labels['label'].values,
            random_state=self.seed
        )
        
        # Second split: separate validation set from remaining patients
        # Adjust val_size to be proportion of train_val set
        adjusted_val_size = val_size / (1 - test_size)
        
        patients_train, patients_val, labels_train, labels_val = train_test_split(
            patients_train_val,
            labels_train_val,
            test_size=adjusted_val_size,
            stratify=labels_train_val,
            random_state=self.seed
        )
        
        # Assign splits to all images based on patient_id
        df['split'] = None
        df.loc[df['patient_id'].isin(patients_train), 'split'] = 'train'
        df.loc[df['patient_id'].isin(patients_val), 'split'] = 'val'
        df.loc[df['patient_id'].isin(patients_test), 'split'] = 'test'
        
        # Create split DataFrames
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        # Verify no patient overlap (CRITICAL CHECK)
        train_patients = set(train_df['patient_id'].unique())
        val_patients = set(val_df['patient_id'].unique())
        test_patients = set(test_df['patient_id'].unique())
        
        overlap_train_val = train_patients & val_patients
        overlap_train_test = train_patients & test_patients
        overlap_val_test = val_patients & test_patients
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                f"❌ CRITICAL ERROR: Patient overlap detected!\n"
                f"   Train-Val overlap: {len(overlap_train_val)} patients\n"
                f"   Train-Test overlap: {len(overlap_train_test)} patients\n"
                f"   Val-Test overlap: {len(overlap_val_test)} patients\n"
                f"This violates patient-wise splitting requirement!"
            )
        
        # Print statistics
        print(f"Split Statistics:")
        print(f"  Train: {len(train_df)} images, {len(train_patients)} patients")
        print(f"         Normal: {len(train_df[train_df['label']=='normal'])}, "
              f"Abnormal: {len(train_df[train_df['label']=='abnormal'])}")
        print(f"  Val:   {len(val_df)} images, {len(val_patients)} patients")
        print(f"         Normal: {len(val_df[val_df['label']=='normal'])}, "
              f"Abnormal: {len(val_df[val_df['label']=='abnormal'])}")
        print(f"  Test:  {len(test_df)} images, {len(test_patients)} patients")
        print(f"         Normal: {len(test_df[test_df['label']=='normal'])}, "
              f"Abnormal: {len(test_df[test_df['label']=='abnormal'])}")
        
        print(f"\n✅ Patient-wise split verification:")
        print(f"   No overlap between train/val: {len(overlap_train_val) == 0}")
        print(f"   No overlap between train/test: {len(overlap_train_test) == 0}")
        print(f"   No overlap between val/test: {len(overlap_val_test) == 0}")
        print(f"   All images assigned: {len(df)} == {len(train_df) + len(val_df) + len(test_df)}")
        
        if save:
            # Save updated patient mapping with split column
            mapping_path = self.metadata_dir / 'patient_mapping.csv'
            df.to_csv(mapping_path, index=False)
            
            # Save split information as JSON
            splits_info = {
                'train': {
                    'patients': list(train_patients),
                    'n_images': len(train_df),
                    'n_patients': len(train_patients),
                    'n_normal': int(len(train_df[train_df['label']=='normal'])),
                    'n_abnormal': int(len(train_df[train_df['label']=='abnormal']))
                },
                'val': {
                    'patients': list(val_patients),
                    'n_images': len(val_df),
                    'n_patients': len(val_patients),
                    'n_normal': int(len(val_df[val_df['label']=='normal'])),
                    'n_abnormal': int(len(val_df[val_df['label']=='abnormal']))
                },
                'test': {
                    'patients': list(test_patients),
                    'n_images': len(test_df),
                    'n_patients': len(test_patients),
                    'n_normal': int(len(test_df[test_df['label']=='normal'])),
                    'n_abnormal': int(len(test_df[test_df['label']=='abnormal']))
                },
                'verification': {
                    'no_patient_overlap': True,
                    'all_images_assigned': len(df) == len(train_df) + len(val_df) + len(test_df)
                }
            }
            
            splits_path = self.metadata_dir / 'splits.json'
            with open(splits_path, 'w') as f:
                json.dump(splits_info, f, indent=2)
            
            print(f"✅ Saved to: {mapping_path}")
            print(f"✅ Saved to: {splits_path}")
        
        print(f"{'='*70}\n")
        
        return train_df, val_df, test_df
    
    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for handling imbalance.
        
        Addresses Reviewer A-5: "Class imbalance acknowledged but not handled"
        
        Dataset: 285 normal vs 201 abnormal (ratio 1.42:1)
        
        Args:
            labels: Array of labels (0=normal, 1=abnormal)
        
        Returns:
            Dictionary mapping class to weight
        """
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = {
            0: class_weights[0],  # Normal
            1: class_weights[1]   # Abnormal
        }
        
        print(f"\n📊 Class Distribution & Weights:")
        print(f"   Normal (0): {np.sum(labels == 0)} samples, weight: {class_weight_dict[0]:.4f}")
        print(f"   Abnormal (1): {np.sum(labels == 1)} samples, weight: {class_weight_dict[1]:.4f}")
        print(f"   Addresses Reviewer A-5: Handling class imbalance with balanced weights\n")
        
        return class_weight_dict


class ECGPreprocessor:
    """
    Handles ECG image preprocessing pipeline.
    
    Addresses Reviewer Requirements:
    - D-14: Grayscale conversion (eliminate color bias)
    - D-13, D-15: Comprehensive preprocessing (remove artifacts)
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.target_size = tuple(self.config['preprocessing']['target_size'])
        self.use_grayscale = self.config['preprocessing']['grayscale']
        self.use_clahe = self.config['preprocessing']['clahe']['enabled']
        self.clahe_clip_limit = self.config['preprocessing']['clahe']['clip_limit']
        self.clahe_grid_size = tuple(self.config['preprocessing']['clahe']['tile_grid_size'])
        self.normalize = self.config['preprocessing']['normalize']
        
        # Initialize CLAHE
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )
        
        logger.info("ECGPreprocessor initialized")
        logger.info(f"  Grayscale: {self.use_grayscale} (Reviewer D-14: Eliminate color bias)")
        logger.info(f"  CLAHE: {self.use_clahe} (Reviewer D-13: Enhance waveforms, reduce artifacts)")
        logger.info(f"  Target size: {self.target_size}")
    
    def preprocess_image(self, img_path: str, return_steps: bool = False) -> np.ndarray:
        """
        Preprocess single ECG image.
        
        Pipeline (Reviewer D-13, D-14, D-15):
        1. Load image
        2. Convert to grayscale (MANDATORY - Reviewer D-14)
        3. Apply CLAHE (enhance waveforms)
        4. Resize to target size
        5. Normalize to [0, 1]
        
        Args:
            img_path: Path to image file
            return_steps: If True, return dict with intermediate steps
        
        Returns:
            Preprocessed image array (H, W) if grayscale, (H, W, 1) for consistency
        """
        # Step 1: Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        steps = {'original': img.copy()} if return_steps else None
        
        # Step 2: Convert to grayscale (CRITICAL - Reviewer D-14)
        # "The ECG plots are color images (RGB), though color carries no 
        # physiological meaning. CNNs may exploit color biases as class cues."
        if self.use_grayscale:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if return_steps:
                steps['grayscale'] = img.copy()
        
        # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Purpose: Enhance ECG waveform visibility, reduce grid influence
        if self.use_clahe and len(img.shape) == 2:
            img = self.clahe.apply(img)
            if return_steps:
                steps['clahe'] = img.copy()
        
        # Step 4: Resize to target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        if return_steps:
            steps['resized'] = img.copy()
        
        # Step 5: Normalize to [0, 1]
        if self.normalize:
            img = img.astype(np.float32) / 255.0
            if return_steps:
                steps['normalized'] = img.copy()
        
        # Add channel dimension if grayscale for consistency
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        if return_steps:
            return img, steps
        return img
    
    def grayscale_to_3channel(self, img: np.ndarray) -> np.ndarray:
        """
        Convert grayscale to 3-channel for pre-trained models.
        
        Addresses Reviewer D-14: "All images converted to grayscale,
        then replicated to 3 channels for pre-trained CNN compatibility"
        
        Args:
            img: Grayscale image (H, W) or (H, W, 1)
        
        Returns:
            RGB image (H, W, 3) with identical channels
        """
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        
        if len(img.shape) == 2:
            img_3ch = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[-1] == 3:
            # Already 3-channel
            img_3ch = img
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        return img_3ch
    
    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        desc: str = "Preprocessing"
    ):
        """
        Preprocess entire dataset and save to disk.
        
        Args:
            df: DataFrame with image paths and labels
            output_dir: Directory to save preprocessed images
            desc: Description for progress bar
        """
        print(f"\n{'='*70}")
        print(f"STEP 3: {desc}")
        print(f"{'='*70}")
        print(f"Processing {len(df)} images...")
        print(f"Pipeline: Load → Grayscale → CLAHE → Resize → Normalize")
        print(f"Addresses Reviewer D-13, D-14, D-15: Comprehensive preprocessing\n")
        
        # Create output directories
        for label in ['normal', 'abnormal']:
            (output_dir / label).mkdir(parents=True, exist_ok=True)
        
        # Preprocess each image
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            try:
                # Preprocess
                img = self.preprocess_image(row['path'])
                
                # Save preprocessed image
                output_path = output_dir / row['label'] / row['filename']
                
                # Convert to uint8 for saving
                if img.max() <= 1.0:
                    img_save = (img * 255).astype(np.uint8)
                else:
                    img_save = img.astype(np.uint8)
                
                # Remove channel dimension if present
                if len(img_save.shape) == 3 and img_save.shape[-1] == 1:
                    img_save = np.squeeze(img_save, axis=-1)
                
                cv2.imwrite(str(output_path), img_save)
                
            except Exception as e:
                logger.error(f"Error preprocessing {row['filename']}: {e}")
                raise
        
        print(f"✅ Preprocessing complete!")
        print(f"   Saved to: {output_dir}")
        print(f"{'='*70}\n")




class ECGAugmentor:
    """
    Handles data augmentation for training set.
    
    Addresses Reviewer A-5, D-8: 
    "Class imbalance acknowledged but not handled. Small dataset constrains deep learning."
    
    Strategy:
    - Geometric: rotation, shift, zoom (small values for ECG)
    - Intensity: brightness, noise, blur
    - More aggressive for minority class (abnormal)
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize augmentor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config['augmentation']
        self.enabled = self.aug_config['enabled']
        
        if self.enabled:
            logger.info("ECGAugmentor initialized")
            logger.info(f"  Rotation: ±{self.aug_config['rotation_range']}°")
            logger.info(f"  Shift: ±{self.aug_config['width_shift_range']*100}%")
            logger.info(f"  Zoom: ±{self.aug_config['zoom_range']*100}%")
            logger.info(f"  Gaussian noise: {self.aug_config['gaussian_noise']['enabled']}")
            logger.info(f"  Gaussian blur: {self.aug_config['gaussian_blur']['enabled']}")
    
    def add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise for robustness.
        
        Args:
            img: Input image (normalized to [0, 1])
        
        Returns:
            Noisy image
        """
        if not self.aug_config['gaussian_noise']['enabled']:
            return img
        
        noise = np.random.normal(
            self.aug_config['gaussian_noise']['mean'],
            self.aug_config['gaussian_noise']['stddev'],
            img.shape
        )
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 1)
    
    def add_gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Add Gaussian blur to simulate image quality variations.
        
        Args:
            img: Input image
        
        Returns:
            Blurred image
        """
        if not self.aug_config['gaussian_blur']['enabled']:
            return img
        
        ksize = self.aug_config['gaussian_blur']['kernel_size']
        sigma = np.random.uniform(*self.aug_config['gaussian_blur']['sigma_range'])
        
        # Handle different image formats
        if img.max() <= 1.0:
            img_uint8 = (img * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)
            return blurred.astype(np.float32) / 255.0
        else:
            return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    def augment_image(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Apply augmentation to single image.
        
        Args:
            img: Input image (H, W) or (H, W, 1), normalized to [0, 1]
            aggressive: If True, apply stronger augmentation (for minority class)
        
        Returns:
            Augmented image
        """
        if not self.enabled:
            return img
        
        # Ensure 2D for cv2 operations
        if len(img.shape) == 3:
            img_2d = np.squeeze(img, axis=-1)
        else:
            img_2d = img
        
        h, w = img_2d.shape
        center = (w // 2, h // 2)
        
        # Augmentation strength multiplier
        strength = 1.5 if aggressive else 1.0
        
        # Rotation
        angle = np.random.uniform(
            -self.aug_config['rotation_range'] * strength,
            self.aug_config['rotation_range'] * strength
        )
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_aug = cv2.warpAffine(img_2d, M_rot, (w, h), 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        # Translation
        tx = np.random.uniform(
            -self.aug_config['width_shift_range'] * strength,
            self.aug_config['width_shift_range'] * strength
        ) * w
        ty = np.random.uniform(
            -self.aug_config['height_shift_range'] * strength,
            self.aug_config['height_shift_range'] * strength
        ) * h
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img_aug = cv2.warpAffine(img_aug, M_trans, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        
        # Zoom
        zoom = 1.0 + np.random.uniform(
            -self.aug_config['zoom_range'] * strength,
            self.aug_config['zoom_range'] * strength
        )
        M_zoom = cv2.getRotationMatrix2D(center, 0, zoom)
        img_aug = cv2.warpAffine(img_aug, M_zoom, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        
        # Brightness (if configured)
        if 'brightness_range' in self.aug_config:
            brightness = np.random.uniform(*self.aug_config['brightness_range'])
            img_aug = img_aug * brightness
            img_aug = np.clip(img_aug, 0, 1)
        
        # Add noise
        img_aug = self.add_gaussian_noise(img_aug)
        
        # Add blur (occasionally)
        if np.random.random() < 0.3:  # 30% chance
            img_aug = self.add_gaussian_blur(img_aug)
        
        # Restore channel dimension if needed
        if len(img.shape) == 3:
            img_aug = np.expand_dims(img_aug, axis=-1)
        
        return img_aug
    
    def create_augmented_generator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ):
        """
        Create generator that yields augmented batches.
        
        Args:
            X: Images array (N, H, W, C)
            y: Labels array (N,)
            batch_size: Batch size
        
        Yields:
            (X_batch, y_batch) with augmented images
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        while True:
            # Shuffle
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = []
                y_batch = []
                
                for idx in batch_indices:
                    img = X[idx]
                    label = y[idx]
                    
                    # Apply more aggressive augmentation to minority class
                    aggressive = (label == 1) and self.aug_config.get('augment_minority_more', False)
                    
                    # Augment
                    img_aug = self.augment_image(img, aggressive=aggressive)
                    
                    X_batch.append(img_aug)
                    y_batch.append(label)
                
                yield np.array(X_batch), np.array(y_batch)


def load_patient_mapping(metadata_dir: str = 'data/metadata') -> pd.DataFrame:
    """
    Load existing patient mapping.
    
    Args:
        metadata_dir: Directory containing patient_mapping.csv
    
    Returns:
        Patient mapping DataFrame
    """
    mapping_path = Path(metadata_dir) / 'patient_mapping.csv'
    
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"❌ Patient mapping not found: {mapping_path}\n"
            f"💡 Tip: Run create_patient_mapping() first"
        )
    
    df = pd.read_csv(mapping_path)
    logger.info(f"✅ Loaded patient mapping: {len(df)} images, {df['patient_id'].nunique()} patients")
    
    return df


def load_splits(metadata_dir: str = 'data/metadata') -> Dict:
    """
    Load existing split information.
    
    Args:
        metadata_dir: Directory containing splits.json
    
    Returns:
        Dictionary with split information
    """
    splits_path = Path(metadata_dir) / 'splits.json'
    
    if not splits_path.exists():
        raise FileNotFoundError(
            f"❌ Split information not found: {splits_path}\n"
            f"💡 Tip: Run create_patient_wise_splits() first"
        )
    
    with open(splits_path, 'r') as f:
        splits_info = json.load(f)
    
    logger.info(f"✅ Loaded split information")
    logger.info(f"   Train: {splits_info['train']['n_images']} images, {splits_info['train']['n_patients']} patients")
    logger.info(f"   Val: {splits_info['val']['n_images']} images, {splits_info['val']['n_patients']} patients")
    logger.info(f"   Test: {splits_info['test']['n_images']} images, {splits_info['test']['n_patients']} patients")
    
    return splits_info


def test_patient_extraction():
    """Test patient ID extraction with various filename patterns."""
    test_cases = [
        ('Ahmad_1.jpg', 'Ahmad'),
        ('Ahmad_2.jpg', 'Ahmad'),
        ('15_1.jpg', '15'),
        ('15_2.jpg', '15'),
        ('Ali Suryana_2.jpg', 'Ali Suryana'),
        ('Ali Suryana_3.jpg', 'Ali Suryana'),
        ('4_1(1).jpg', '4_1'),
        ('Marjuki(1)_2.jpg', 'Marjuki(1)'),
        ('Asiah_1.jpg', 'Asiah'),
    ]
    
    print("\n" + "="*70)
    print("Testing Patient ID Extraction")
    print("="*70)
    
    manager = ECGDataManager()
    all_passed = True
    
    for filename, expected in test_cases:
        result = manager.extract_patient_id(filename)
        passed = result == expected
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"{status} {filename:30} → {result:20} (expected: {expected})")
    
    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed!'}")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == '__main__':
    """
    Demo usage of data utilities.
    """
    print("\n" + "="*70)
    print("ECG Data Utilities - Demo")
    print("="*70 + "\n")
    
    # Test patient extraction
    test_patient_extraction()
    
    # Create data manager
    manager = ECGDataManager('config.yaml')
    
    # Create patient mapping
    df = manager.create_patient_mapping()
    
    # Create splits
    train_df, val_df, test_df = manager.create_patient_wise_splits(df)
    
    # Compute class weights
    class_weights = manager.get_class_weights(train_df['label_numeric'].values)
    
    print("\n✅ Data utilities demo complete!")
    print("Next step: Run preprocessing with ECGPreprocessor")
    print("="*70 + "\n")