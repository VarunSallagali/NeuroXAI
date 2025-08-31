"""
Model training module for Parkinson's Disease Prediction using XAI.
Implements various ML and DL models with training and evaluation capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, learning_curve
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelTrainer:
    """Class for training traditional machine learning models."""
    
    def __init__(self):
        """Initialize the ML model trainer."""
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.training_history = {}
        
    def get_models(self):
        """Get a dictionary of available ML models."""
        return {
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'mlp': MLPClassifier(random_state=42, max_iter=1000)
        }
    
    def train_model(self, model_name, X_train, y_train, **kwargs):
        """
        Train a specific ML model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            **kwargs: Additional parameters for the model
            
        Returns:
            tuple: (trained_model, training_time)
        """
        models_dict = self.get_models()
        
        if model_name not in models_dict:
            raise ValueError(f"Model {model_name} not available. Available models: {list(models_dict.keys())}")
        
        model = models_dict[model_name]
        
        # Update model parameters if provided
        if kwargs:
            model.set_params(**kwargs)
        
        logger.info(f"Training {model_name} model...")
        start_time = datetime.now()
        
        model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        
        self.models[model_name] = model
        return model, training_time
    
    def train_all_models(self, X_train, y_train, **kwargs):
        """
        Train all available ML models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            **kwargs: Additional parameters for models
        """
        models_dict = self.get_models()
        
        for model_name in models_dict.keys():
            try:
                self.train_model(model_name, X_train, y_train, **kwargs)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store metrics
        if model_name not in self.training_history:
            self.training_history[model_name] = {}
        self.training_history[model_name]['test_metrics'] = metrics
        
        logger.info(f"{model_name} evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """
        Perform cross-validation for a model.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        models_dict = self.get_models()
        
        if model_name not in models_dict:
            raise ValueError(f"Model {model_name} not available")
        
        model = models_dict[model_name]
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        logger.info(f"{model_name} cross-validation: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        return results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, method='grid', cv=5):
        """
        Perform hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            param_grid (dict): Parameter grid for tuning
            method (str): Tuning method ('grid' or 'random')
            cv (int): Number of cross-validation folds
            
        Returns:
            tuple: (best_model, best_params, best_score)
        """
        models_dict = self.get_models()
        
        if model_name not in models_dict:
            raise ValueError(f"Model {model_name} not available")
        
        base_model = models_dict[model_name]
        
        if method == 'grid':
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        elif method == 'random':
            search = RandomizedSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, n_iter=20)
        else:
            raise ValueError("method must be 'grid' or 'random'")
        
        logger.info(f"Performing {method} search for {model_name}...")
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Store the best model
        self.models[f"{model_name}_tuned"] = best_model
        
        logger.info(f"Best {model_name} parameters: {best_params}")
        logger.info(f"Best {model_name} score: {best_score:.4f}")
        
        return best_model, best_params, best_score
    
    def save_model(self, model_name, filepath):
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name, filepath):
        """Load a saved model from disk."""
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")

class DLModelTrainer:
    """Class for training deep learning models."""
    
    def __init__(self, input_shape=None):
        """
        Initialize the DL model trainer.
        
        Args:
            input_shape (tuple): Shape of input features
        """
        self.input_shape = input_shape
        self.models = {}
        self.training_history = {}
        
    def create_dnn_model(self, input_shape, layers_config=None):
        """
        Create a deep neural network model.
        
        Args:
            input_shape (tuple): Shape of input features
            layers_config (list): Configuration for hidden layers
            
        Returns:
            keras.Model: Compiled DNN model
        """
        if layers_config is None:
            layers_config = [64, 32, 16]
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(layers_config[0], activation='relu', input_shape=input_shape))
        model.add(layers.Dropout(0.3))
        
        # Hidden layers
        for units in layers_config[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info(f"DNN model created with {len(layers_config)} hidden layers")
        return model
    
    def train_dnn_model(self, model_name, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train a DNN model.
        
        Args:
            model_name (str): Name for the model
            X_train (np.array): Training features
            y_train (np.array): Training labels
            X_val (np.array): Validation features
            y_val (np.array): Validation labels
            **kwargs: Additional training parameters
            
        Returns:
            keras.Model: Trained model
        """
        # Create model if input_shape is not set
        if self.input_shape is None:
            self.input_shape = (X_train.shape[1],)
        
        model = self.create_dnn_model(self.input_shape)
        
        # Set default training parameters
        training_params = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'callbacks': [
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        }
        
        # Update with provided parameters
        training_params.update(kwargs)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            training_params['validation_data'] = (X_val, y_val)
            training_params.pop('validation_split', None)
        
        logger.info(f"Training DNN model {model_name}...")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            **training_params
        )
        
        # Store model and history
        self.models[model_name] = model
        self.training_history[model_name] = history.history
        
        logger.info(f"DNN model {model_name} training completed")
        return model
    
    def evaluate_dnn_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained DNN model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        # Evaluate model
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate additional metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        # Store metrics
        if model_name not in self.training_history:
            self.training_history[model_name] = {}
        self.training_history[model_name]['test_metrics'] = metrics
        
        logger.info(f"DNN model {model_name} evaluation completed. Accuracy: {test_accuracy:.4f}")
        return metrics
    
    def plot_training_history(self, model_name, figsize=(15, 5)):
        """
        Plot training history for a DNN model.
        
        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size
        """
        if model_name not in self.training_history:
            logger.warning(f"No training history available for {model_name}")
            return
        
        history = self.training_history[model_name]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Plot loss
        axes[1].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        # Plot precision and recall
        if 'precision' in history and 'recall' in history:
            axes[2].plot(history['precision'], label='Training Precision')
            axes[2].plot(history['recall'], label='Training Recall')
            if 'val_precision' in history and 'val_recall' in history:
                axes[2].plot(history['val_precision'], label='Validation Precision')
                axes[2].plot(history['val_recall'], label='Validation Recall')
            axes[2].set_title('Model Precision & Recall')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Score')
            axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Training history plot displayed for {model_name}")
    
    def save_model(self, model_name, filepath):
        """Save a trained DNN model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        self.models[model_name].save(filepath)
        logger.info(f"DNN model {model_name} saved to {filepath}")
    
    def load_model(self, model_name, filepath):
        """Load a saved DNN model from disk."""
        self.models[model_name] = models.load_model(filepath)
        logger.info(f"DNN model {model_name} loaded from {filepath}")

def main():
    """Example usage of the model training classes."""
    print("Model training modules created successfully!")
    print("\nMLModelTrainer provides:")
    print("1. Traditional ML models (Random Forest, SVM, Logistic Regression, etc.)")
    print("2. Hyperparameter tuning with GridSearch and RandomizedSearch")
    print("3. Cross-validation and model evaluation")
    print("4. Model saving and loading")
    
    print("\nDLModelTrainer provides:")
    print("1. Deep Neural Network model creation and training")
    print("2. Training history visualization")
    print("3. Model evaluation and metrics")
    print("4. Model saving and loading")

if __name__ == "__main__":
    main()
