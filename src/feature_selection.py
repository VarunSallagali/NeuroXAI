"""
Feature selection module for Parkinson's Disease Prediction using XAI.
Implements various feature selection techniques and importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """Class for feature selection and importance analysis."""
    
    def __init__(self):
        """Initialize the feature selector."""
        self.selected_features = None
        self.feature_importance_scores = None
        self.selector = None
        
    def statistical_feature_selection(self, X, y, k=10, method='f_classif'):
        """
        Perform statistical feature selection.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            k (int): Number of top features to select
            method (str): Selection method ('f_classif' or 'mutual_info_classif')
            
        Returns:
            tuple: (X_selected, selected_features, scores)
        """
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info_classif':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError("method must be 'f_classif' or 'mutual_info_classif'")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        scores = selector.scores_
        
        logger.info(f"Statistical feature selection completed. Selected {len(selected_features)} features.")
        return X_selected, selected_features, scores
    
    def recursive_feature_elimination(self, X, y, n_features=10, estimator=None):
        """
        Perform Recursive Feature Elimination (RFE).
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_features (int): Number of features to select
            estimator: Base estimator for RFE
            
        Returns:
            tuple: (X_selected, selected_features, ranking)
        """
        if estimator is None:
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        selected_features = X.columns[rfe.support_].tolist()
        ranking = rfe.ranking_
        
        logger.info(f"RFE completed. Selected {len(selected_features)} features.")
        return X_selected, selected_features, ranking
    
    def random_forest_importance(self, X, y, n_estimators=100, random_state=42):
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_estimators (int): Number of trees in the forest
            random_state (int): Random seed
            
        Returns:
            tuple: (importance_scores, feature_names)
        """
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importance_scores = rf.feature_importances_
        feature_names = X.columns.tolist()
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_scores = importance_df
        
        logger.info("Random Forest feature importance calculated.")
        return importance_scores, feature_names
    
    def select_top_features(self, X, y, method='random_forest', n_features=10, **kwargs):
        """
        Select top features using specified method.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Selection method
            n_features (int): Number of features to select
            **kwargs: Additional arguments for specific methods
            
        Returns:
            tuple: (X_selected, selected_features)
        """
        if method == 'random_forest':
            importance_scores, _ = self.random_forest_importance(X, y, **kwargs)
            top_indices = np.argsort(importance_scores)[-n_features:]
            selected_features = X.columns[top_indices].tolist()
            X_selected = X[selected_features]
            
        elif method == 'statistical':
            X_selected, selected_features, _ = self.statistical_feature_selection(
                X, y, k=n_features, **kwargs
            )
            
        elif method == 'rfe':
            X_selected, selected_features, _ = self.recursive_feature_elimination(
                X, y, n_features=n_features, **kwargs
            )
            
        else:
            raise ValueError("method must be 'random_forest', 'statistical', or 'rfe'")
        
        self.selected_features = selected_features
        logger.info(f"Feature selection completed using {method}. Selected {len(selected_features)} features.")
        
        return X_selected, selected_features
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance scores.
        
        Args:
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
        """
        if self.feature_importance_scores is None:
            logger.warning("No feature importance scores available. Run random_forest_importance first.")
            return
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance_scores.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance Scores')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Feature importance plot displayed for top {top_n} features.")
    
    def compare_feature_selection_methods(self, X, y, n_features=10):
        """
        Compare different feature selection methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_features (int): Number of features to select
            
        Returns:
            dict: Comparison results
        """
        results = {}
        
        # Random Forest importance
        X_rf, features_rf = self.select_top_features(
            X, y, method='random_forest', n_features=n_features
        )
        results['random_forest'] = features_rf
        
        # Statistical selection
        X_stat, features_stat = self.select_top_features(
            X, y, method='statistical', n_features=n_features
        )
        results['statistical'] = features_stat
        
        # RFE
        X_rfe, features_rfe = self.select_top_features(
            X, y, method='rfe', n_features=n_features
        )
        results['rfe'] = features_rfe
        
        # Find common features
        common_features = set(features_rf) & set(features_stat) & set(features_rfe)
        results['common_features'] = list(common_features)
        
        logger.info("Feature selection methods comparison completed.")
        return results
    
    def evaluate_feature_selection(self, X, y, selected_features, estimator=None):
        """
        Evaluate the performance with selected features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            selected_features (list): List of selected features
            estimator: Classifier to use for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        if estimator is None:
            estimator = RandomForestClassifier(random_state=42, n_estimators=100)
        
        # Select features
        X_selected = X[selected_features]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train and evaluate
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'n_features': len(selected_features)
        }
        
        logger.info(f"Feature selection evaluation completed. Accuracy: {accuracy:.4f}")
        return results

def main():
    """Example usage of the FeatureSelector class."""
    # This would typically be used after preprocessing
    print("FeatureSelector class created successfully!")
    print("Use this class to:")
    print("1. Select top features using Random Forest importance")
    print("2. Perform statistical feature selection")
    print("3. Use Recursive Feature Elimination (RFE)")
    print("4. Compare different feature selection methods")
    print("5. Evaluate the impact of feature selection on model performance")

if __name__ == "__main__":
    main()
