"""
ECG Classification Models Module

Implements all 6 models with clear documentation:
1. SVM with deep feature extraction (Reviewer D-11, D-25)
2. Custom CNN
3. VGG16 (transfer learning)
4. ResNet50 (transfer learning)
5. InceptionV3 (transfer learning)
6. Xception (transfer learning)

Author: Abdullah Hasan Dafa
Date: November 2025
"""

import logging
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import yaml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, Xception
)
from tensorflow.keras.regularizers import l2

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating all ECG classification models.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize model factory with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.l2_reg = self.training_config['regularization']['l2_weight_decay']
        self.dropout = self.training_config['regularization']['dropout']
        
        logger.info("ModelFactory initialized")
    
    def create_model(self, model_name: str, input_shape: Tuple[int, int, int]):
        """
        Create model by name.
        
        Args:
            model_name: One of ['svm', 'custom_cnn', 'vgg16', 'resnet50', 'inception_v3', 'xception']
            input_shape: Input shape (H, W, C)
        
        Returns:
            Model instance
        """
        model_creators = {
            'svm': lambda: self.create_svm_pipeline(),
            'custom_cnn': lambda: self.create_custom_cnn(input_shape),
            'vgg16': lambda: self.create_vgg16(input_shape),
            'resnet50': lambda: self.create_resnet50(input_shape),
            'inception_v3': lambda: self.create_inception_v3(input_shape),
            'xception': lambda: self.create_xception(input_shape)
        }
        
        if model_name not in model_creators:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Creating model: {model_name}")
        return model_creators[model_name]()
    
    # =========================================================================
    # SVM WITH DEEP FEATURE EXTRACTION
    # =========================================================================
    
    def create_svm_pipeline(self):
        """
        Create SVM classifier with deep feature extraction.
        
        Addresses Reviewer D-11, D-25:
        "Feature representation undefined. The SVM input features are never described."
        
        Implementation:
        - Feature extraction: VGG16 fc2 layer (4096-dimensional)
        - Scaling: StandardScaler
        - Classifier: SVM with RBF kernel
        
        Returns:
            Dictionary with feature_extractor, scaler, classifier
        """
        logger.info("Creating SVM pipeline with VGG16 feature extraction")
        logger.info("  Addresses Reviewer D-11, D-25: Deep feature extraction documented")
        
        svm_config = self.model_config['svm']
        
        # Step 1: Feature extractor (VGG16 fc2 layer)
        base_model = VGG16(
            weights=svm_config['feature_extraction']['pretrained_weights'],
            include_top=True
        )
        
        # Extract features from fc2 layer (4096-dim)
        feature_extractor = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output,
            name='vgg16_feature_extractor'
        )
        feature_extractor.trainable = False  # Freeze weights
        
        logger.info(f"  Feature extractor: VGG16 {svm_config['feature_extraction']['layer_name']} layer")
        logger.info(f"  Feature dimension: {svm_config['feature_extraction']['feature_dim']}")
        
        # Step 2: Scaler
        scaler = StandardScaler()
        logger.info(f"  Scaler: {svm_config['scaler']}")
        
        # Step 3: SVM classifier
        classifier = SVC(
            kernel=svm_config['kernel'],
            C=svm_config['C'],
            gamma=svm_config['gamma'],
            probability=svm_config['probability'],
            class_weight=svm_config['class_weight'],
            random_state=svm_config['random_state'],
            max_iter=svm_config['max_iter']
        )
        
        logger.info(f"  SVM kernel: {svm_config['kernel']}")
        logger.info(f"  SVM C: {svm_config['C']}")
        logger.info(f"  SVM gamma: {svm_config['gamma']}")
        logger.info(f"  Class weight: {svm_config['class_weight']}")
        
        return {
            'feature_extractor': feature_extractor,
            'scaler': scaler,
            'classifier': classifier,
            'type': 'svm_pipeline'
        }
    
    # =========================================================================
    # CUSTOM CNN
    # =========================================================================
    
    def create_custom_cnn(self, input_shape: Tuple[int, int, int]) -> Model:
        """
        Create custom CNN architecture for ECG classification.
        
        Architecture:
        - Input: (224, 224, 1) - Grayscale
        - Conv Block 1: 32 filters → BatchNorm → MaxPool → Dropout(0.25)
        - Conv Block 2: 64 filters → BatchNorm → MaxPool → Dropout(0.25)
        - Conv Block 3: 128 filters → BatchNorm → MaxPool → Dropout(0.25)
        - Dense: 256 units → Dropout(0.5)
        - Output: 1 unit (Sigmoid)
        
        Args:
            input_shape: Input shape (H, W, C)
        
        Returns:
            Keras Model
        """
        logger.info(f"Creating Custom CNN with input shape: {input_shape}")
        
        cnn_config = self.model_config['custom_cnn']
        
        inputs = layers.Input(shape=input_shape, name='input')
        x = inputs
        
        # Convolutional blocks
        for i, block_config in enumerate(cnn_config['conv_blocks']):
            # Conv layer
            x = layers.Conv2D(
                filters=block_config['filters'],
                kernel_size=block_config['kernel_size'],
                activation=block_config['activation'],
                padding='same',
                kernel_regularizer=l2(self.l2_reg),
                name=f'conv_{i+1}'
            )(x)
            
            # Batch normalization
            if block_config.get('batch_norm', False):
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # Pooling
            x = layers.MaxPooling2D(
                pool_size=block_config['pool_size'],
                name=f'pool_{i+1}'
            )(x)
            
            # Dropout
            x = layers.Dropout(
                block_config['dropout'],
                name=f'dropout_{i+1}'
            )(x)
        
        # Flatten
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layers
        for i, dense_config in enumerate(cnn_config['dense_layers']):
            x = layers.Dense(
                units=dense_config['units'],
                activation=dense_config['activation'],
                kernel_regularizer=l2(self.l2_reg),
                name=f'dense_{i+1}'
            )(x)
            
            x = layers.Dropout(
                dense_config['dropout'],
                name=f'dropout_dense_{i+1}'
            )(x)
        
        # Output layer
        outputs = layers.Dense(
            units=1,
            activation=cnn_config['output_activation'],
            name='output'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CustomCNN')
        
        logger.info(f"  Custom CNN created with {model.count_params():,} parameters")
        
        return model
    
    # =========================================================================
    # TRANSFER LEARNING MODELS
    # =========================================================================
    
    def _create_transfer_learning_model(
        self,
        base_model_func,
        model_name: str,
        input_shape: Tuple[int, int, int]
    ) -> Model:
        """
        Generic function to create transfer learning model.
        
        Args:
            base_model_func: Function to create base model (e.g., VGG16)
            model_name: Name of the model
            input_shape: Input shape (H, W, 3) - Must be 3-channel
        
        Returns:
            Keras Model
        """
        logger.info(f"Creating {model_name} with transfer learning")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Note: Grayscale images replicated to 3 channels (Reviewer D-14)")
        
        tl_config = self.model_config['transfer_learning']
        
        # Load pretrained base model
        base_model = base_model_func(
            weights=tl_config['pretrained_weights'],
            include_top=tl_config['include_top'],
            input_shape=input_shape
        )
        
        # Freeze base model
        if tl_config['freeze_base']:
            base_model.trainable = False
            logger.info(f"  Convolutional base frozen")
        
        # Build model
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Base model
        x = base_model(inputs, training=False)
        
        # Pooling
        if tl_config['pooling'] == 'global_average':
            x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        elif tl_config['pooling'] == 'global_max':
            x = layers.GlobalMaxPooling2D(name='global_max_pool')(x)
        
        # Custom classification head
        x = layers.Dense(
            units=tl_config['dense_units'],
            activation=tl_config['dense_activation'],
            kernel_regularizer=l2(self.l2_reg),
            name='dense'
        )(x)
        
        x = layers.Dropout(
            tl_config['dropout'],
            name='dropout'
        )(x)
        
        # Output
        outputs = layers.Dense(
            units=1,
            activation=tl_config['output_activation'],
            name='output'
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=model_name)
        
        logger.info(f"  {model_name} created")
        logger.info(f"  Total parameters: {model.count_params():,}")
        logger.info(f"  Trainable parameters: {np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def create_vgg16(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
        """
        Create VGG16 model with transfer learning.
        
        Args:
            input_shape: Input shape (must be 3-channel)
        
        Returns:
            Keras Model
        """
        return self._create_transfer_learning_model(
            VGG16,
            'VGG16_Transfer',
            input_shape
        )
    
    def create_resnet50(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
        """
        Create ResNet50 model with transfer learning.
        
        Args:
            input_shape: Input shape (must be 3-channel)
        
        Returns:
            Keras Model
        """
        return self._create_transfer_learning_model(
            ResNet50,
            'ResNet50_Transfer',
            input_shape
        )
    
    def create_inception_v3(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
        """
        Create InceptionV3 model with transfer learning.
        
        Note: InceptionV3 officially requires 299x299 input, but we use 224x224
        for consistency across models. This is acceptable for transfer learning.
        
        Args:
            input_shape: Input shape (must be 3-channel)
        
        Returns:
            Keras Model
        """
        logger.info("Note: InceptionV3 officially uses 299x299, using 224x224 for consistency")
        return self._create_transfer_learning_model(
            InceptionV3,
            'InceptionV3_Transfer',
            input_shape
        )
    
    def create_xception(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
        """
        Create Xception model with transfer learning.
        
        Note: Xception officially requires minimum 71x71 input.
        
        Args:
            input_shape: Input shape (must be 3-channel)
        
        Returns:
            Keras Model
        """
        return self._create_transfer_learning_model(
            Xception,
            'Xception_Transfer',
            input_shape
        )


def compile_model(
    model: Model,
    learning_rate: float = 0.0001,
    class_weight: Optional[dict] = None
):
    """
    Compile Keras model with optimizer and loss.
    
    Addresses Reviewer A-6, D-3: Document all hyperparameters
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        class_weight: Class weights for handling imbalance
    """
    logger.info(f"Compiling model: {model.name}")
    logger.info(f"  Optimizer: Adam (lr={learning_rate})")
    logger.info(f"  Loss: binary_crossentropy")
    logger.info(f"  Metrics: accuracy, AUC, precision, recall")
    
    if class_weight:
        logger.info(f"  Class weights: {class_weight}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )


def create_callbacks(
    model_name: str,
    config: dict,
    checkpoint_dir: str = 'results/models'
) -> list:
    """
    Create training callbacks.
    
    Addresses Reviewer D-3: Document early stopping, LR scheduling
    
    Args:
        model_name: Name of the model
        config: Training configuration dictionary
        checkpoint_dir: Directory to save model checkpoints
    
    Returns:
        List of Keras callbacks
    """
    from pathlib import Path
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks_list = []
    
    # Early stopping
    if config['early_stopping']['enabled']:
        early_stop = keras.callbacks.EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            mode='min',
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta'],
            restore_best_weights=config['early_stopping']['restore_best_weights'],
            verbose=config['early_stopping']['verbose']
        )
        callbacks_list.append(early_stop)
        logger.info(f"  Early stopping: monitor={config['early_stopping']['monitor']}, "
                   f"patience={config['early_stopping']['patience']}")
    
    # Learning rate scheduler
    if config['lr_scheduler']['enabled']:
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor=config['lr_scheduler']['monitor'],
            mode=config['lr_scheduler']['mode'],
            factor=config['lr_scheduler']['factor'],
            patience=config['lr_scheduler']['patience'],
            min_lr=config['lr_scheduler']['min_lr'],
            verbose=config['lr_scheduler']['verbose']
        )
        callbacks_list.append(lr_scheduler)
        logger.info(f"  LR scheduler: monitor={config['lr_scheduler']['monitor']}, "
                   f"factor={config['lr_scheduler']['factor']}, "
                   f"patience={config['lr_scheduler']['patience']}")
    
    # Model checkpoint
    if config['checkpoint']['enabled']:
        checkpoint_path = Path(checkpoint_dir) / f"{model_name}_best.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config['checkpoint']['monitor'],
            mode=config['checkpoint']['mode'],
            save_best_only=config['checkpoint']['save_best_only'],
            save_weights_only=config['checkpoint']['save_weights_only'],
            verbose=1
        )
        callbacks_list.append(checkpoint)
        logger.info(f"  Model checkpoint: {checkpoint_path}")
    
    return callbacks_list


def get_model_summary(model):
    """
    Get formatted model summary.
    
    Args:
        model: Keras model or SVM pipeline dict
    
    Returns:
        String with model summary
    """
    if isinstance(model, dict) and model.get('type') == 'svm_pipeline':
        # SVM pipeline
        summary = []
        summary.append("="*70)
        summary.append("SVM Pipeline Summary")
        summary.append("="*70)
        summary.append(f"Feature Extraction: {model['feature_extractor'].name}")
        summary.append(f"  Output shape: (None, 4096)")
        summary.append(f"Scaler: StandardScaler")
        summary.append(f"Classifier: SVM (RBF kernel)")
        summary.append("="*70)
        return "\n".join(summary)
    else:
        # Keras model
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        model.summary()
        summary = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        return summary


if __name__ == '__main__':
    """
    Demo: Create and display all models.
    """
    print("\n" + "="*70)
    print("ECG Models - Demo")
    print("="*70 + "\n")
    
    factory = ModelFactory('config.yaml')
    
    # Create all models
    models_to_create = [
        ('svm', (224, 224, 1)),
        ('custom_cnn', (224, 224, 1)),
        ('vgg16', (224, 224, 3)),
        ('resnet50', (224, 224, 3)),
        ('inception_v3', (224, 224, 3)),
        ('xception', (224, 224, 3))
    ]
    
    for model_name, input_shape in models_to_create:
        print(f"\n{'='*70}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*70}")
        
        model = factory.create_model(model_name, input_shape)
        
        if model_name != 'svm':
            print(get_model_summary(model))
        else:
            print(get_model_summary(model))
    
    print("\n✅ All models created successfully!")
    print("="*70 + "\n")