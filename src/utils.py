"""
Utility functions for Parkinson's Disease Prediction using XAI.
Provides helper functions for visualization, metrics, and data handling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import learning_curve, validation_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """Class for creating comprehensive data visualizations."""
    
    def __init__(self, style='default'):
        """
        Initialize the data visualizer.
        
        Args:
            style (str): Plotting style to use
        """
        self.style = style
        self.set_plotting_style()
    
    def set_plotting_style(self):
        """Set the plotting style for matplotlib and seaborn."""
        if self.style == 'default':
            plt.style.use('default')
            sns.set_theme(style="whitegrid")
        elif self.style == 'seaborn':
            plt.style.use('seaborn-v0_8')
            sns.set_theme(style="darkgrid")
        elif self.style == 'ggplot':
            plt.style.use('ggplot')
        
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
    
    def plot_data_overview(self, data, target_column='status', figsize=(20, 15)):
        """
        Create a comprehensive data overview visualization.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of the target variable
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Parkinson\'s Disease Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        if target_column in data.columns:
            target_counts = data[target_column].value_counts()
            axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Target Distribution')
        
        # 2. Dataset shape info
        axes[0, 1].text(0.1, 0.8, f'Samples: {data.shape[0]}', fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].text(0.1, 0.6, f'Features: {data.shape[1]}', fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].text(0.1, 0.4, f'Missing: {data.isnull().sum().sum()}', fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].text(0.1, 0.2, f'Duplicates: {data.duplicated().sum()}', fontsize=12, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Dataset Information')
        axes[0, 1].axis('off')
        
        # 3. Data types distribution
        data_types = data.dtypes.value_counts()
        axes[0, 2].bar(data_types.index.astype(str), data_types.values)
        axes[0, 2].set_title('Data Types Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Missing values heatmap
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0]
            axes[1, 0].bar(range(len(missing_data)), missing_data.values)
            axes[1, 0].set_title('Missing Values by Feature')
            axes[1, 0].set_xticks(range(len(missing_data)))
            axes[1, 0].set_xticklabels(missing_data.index, rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Missing Values')
        
        # 5. Feature correlation heatmap (top 10 features)
        if target_column in data.columns:
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                target_corr = corr_matrix[target_column].abs().sort_values(ascending=False).head(10)
                axes[1, 1].bar(range(len(target_corr)), target_corr.values)
                axes[1, 1].set_title('Top 10 Features by Target Correlation')
                axes[1, 1].set_xticks(range(len(target_corr)))
                axes[1, 1].set_xticklabels(target_corr.index, rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient numeric features', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Feature Correlation')
        
        # 6. Feature value ranges
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            feature_ranges = numeric_data.max() - numeric_data.min()
            top_features = feature_ranges.nlargest(10)
            axes[1, 2].bar(range(len(top_features)), top_features.values)
            axes[1, 2].set_title('Top 10 Features by Value Range')
            axes[1, 2].set_xticks(range(len(top_features)))
            axes[1, 2].set_xticklabels(top_features.index, rotation=45)
        
        # 7. Feature distributions (histograms for top 6 features)
        if len(numeric_data.columns) > 0:
            top_features = numeric_data.columns[:6]
            for i, feature in enumerate(top_features):
                row, col = 2, i % 3
                axes[row, col].hist(data[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Data overview visualization completed")
    
    def plot_feature_distributions(self, data, features=None, target_column='status', n_features=6):
        """
        Plot feature distributions by target class.
        
        Args:
            data (pd.DataFrame): Input dataset
            features (list): List of features to plot (None for automatic selection)
            target_column (str): Name of the target variable
            n_features (int): Number of features to plot
        """
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return
        
        # Select features to plot
        if features is None:
            numeric_features = data.select_dtypes(include=[np.number]).columns
            # Exclude target column
            numeric_features = [f for f in numeric_features if f != target_column]
            # Select top features by variance
            variances = data[numeric_features].var().sort_values(ascending=False)
            features = variances.head(n_features).index.tolist()
        
        # Create subplots
        n_cols = 3
        n_rows = (len(features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Feature Distributions by Target Class', fontsize=16, fontweight='bold')
        
        # Flatten axes if needed
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature in enumerate(features):
            row, col = i // n_cols, i % n_cols
            
            # Plot distributions for each class
            for target_value in sorted(data[target_column].unique()):
                subset = data[data[target_column] == target_value][feature].dropna()
                axes[row, col].hist(subset, bins=30, alpha=0.7, label=f'Class {target_value}')
            
            axes[row, col].set_title(f'{feature}')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(features), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Feature distributions plotted for {len(features)} features")
    
    def plot_correlation_matrix(self, data, target_column='status', top_n=20, figsize=(12, 10)):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of the target variable
            top_n (int): Number of top correlated features to show
            figsize (tuple): Figure size
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_column not in numeric_data.columns:
            logger.error(f"Target column '{target_column}' not found in numeric data")
            return
        
        # Calculate correlations with target
        target_correlations = numeric_data.corr()[target_column].abs().sort_values(ascending=False)
        top_features = target_correlations.head(top_n + 1).index.tolist()  # +1 to include target
        
        # Create correlation matrix for top features
        corr_matrix = numeric_data[top_features].corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Matrix (Top {top_n} Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Correlation matrix plotted for top {top_n} features")

class ModelEvaluator:
    """Class for comprehensive model evaluation and visualization."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_pred_proba (array-like): Predicted probabilities
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, figsize=(8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list): Names of the classes
            figsize (tuple): Figure size
        """
        if class_names is None:
            class_names = ['No Parkinson', 'Parkinson']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        logger.info("Confusion matrix plotted")
    
    def plot_roc_curve(self, y_true, y_pred_proba, class_names=None, figsize=(8, 6)):
        """
        Plot ROC curve.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            class_names (list): Names of the classes
            figsize (tuple): Figure size
        """
        if class_names is None:
            class_names = ['No Parkinson', 'Parkinson']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"ROC curve plotted with AUC: {roc_auc:.3f}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, class_names=None, figsize=(8, 6)):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            class_names (list): Names of the classes
            figsize (tuple): Figure size
        """
        if class_names is None:
            class_names = ['No Parkinson', 'Parkinson']
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkgreen', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Precision-recall curve plotted with AUC: {pr_auc:.3f}")
    
    def plot_learning_curves(self, estimator, X, y, cv=5, figsize=(15, 5)):
        """
        Plot learning curves for model evaluation.
        
        Args:
            estimator: The model to evaluate
            X (array-like): Features
            y (array-like): Labels
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size
        """
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        # Plot learning curves
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Plot score distribution
        plt.subplot(1, 2, 2)
        plt.boxplot([train_scores[-1], val_scores[-1]], labels=['Training', 'Validation'])
        plt.ylabel('Score', fontsize=12)
        plt.title('Final Score Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Learning curves plotted")
    
    def create_evaluation_report(self, y_true, y_pred, y_pred_proba=None, 
                                class_names=None, output_file=None):
        """
        Create a comprehensive evaluation report.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_pred_proba (array-like): Predicted probabilities
            class_names (list): Names of the classes
            output_file (str): Path to save the report
        """
        if class_names is None:
            class_names = ['No Parkinson', 'Parkinson']
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Create classification report
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create report
        report = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'n_samples': len(y_true),
            'n_classes': len(class_names)
        }
        
        # Save report if file path provided
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {output_file}")
        
        return report

def save_model_results(model_name, results, output_dir='results'):
    """
    Save model results to disk.
    
    Args:
        model_name (str): Name of the model
        results (dict): Results to save
        output_dir (str): Output directory
    """
    import os
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model results saved to {filepath}")
    return filepath

def main():
    """Example usage of the utility classes."""
    print("Utility modules created successfully!")
    print("\nDataVisualizer provides:")
    print("1. Comprehensive data overview visualizations")
    print("2. Feature distribution plots")
    print("3. Correlation matrix heatmaps")
    
    print("\nModelEvaluator provides:")
    print("1. Comprehensive evaluation metrics")
    print("2. Confusion matrix visualization")
    print("3. ROC and precision-recall curves")
    print("4. Learning curves analysis")
    print("5. Evaluation report generation")
    
    print("\nAdditional utilities:")
    print("1. Model results saving")
    print("2. Plotting style configuration")
    print("3. Data handling helpers")

if __name__ == "__main__":
    main()
