"""
Flask web application for Parkinson's Disease Prediction using XAI.
Provides a web interface for model inference and explanations.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.model_training import MLModelTrainer, DLModelTrainer
from src.explainability import ModelExplainer
from src.utils import DataVisualizer, ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for loaded models and preprocessors
models = {}
preprocessor = None
feature_selector = None
explainer = None

# Configuration
MODELS_DIR = 'models'
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_models():
    """Load trained models from disk."""
    global models, preprocessor, feature_selector
    
    try:
        # Load preprocessor
        if os.path.exists(os.path.join(MODELS_DIR, 'preprocessor.pkl')):
            preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
            logger.info("Preprocessor loaded successfully")
        
        # Load feature selector
        if os.path.exists(os.path.join(MODELS_DIR, 'feature_selector.pkl')):
            feature_selector = joblib.load(os.path.join(MODELS_DIR, 'feature_selector.pkl'))
            logger.info("Feature selector loaded successfully")
        
        # Load ML models
        ml_models = ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting']
        for model_name in ml_models:
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                logger.info(f"Model {model_name} loaded successfully")
        
        # Load DNN model if available
        dnn_path = os.path.join(MODELS_DIR, 'dnn_model.h5')
        if os.path.exists(dnn_path):
            try:
                from tensorflow import keras
                models['dnn'] = keras.models.load_model(dnn_path)
                logger.info("DNN model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load DNN model: {e}")
        
        logger.info(f"Loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def initialize_explainer():
    """Initialize the model explainer with the best model."""
    global explainer
    
    if models:
        # Use the first available model for explanations
        best_model = list(models.values())[0]
        explainer = ModelExplainer(best_model)
        logger.info("Model explainer initialized")

@app.route('/')
def index():
    """Main page of the application."""
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """Get list of available models."""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            'type': type(model).__name__,
            'name': name
        }
    return jsonify(model_info)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the selected model."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'random_forest')
        features = data.get('features', {})
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Preprocess features if preprocessor is available
        if preprocessor:
            # Apply the same preprocessing steps
            feature_df = preprocessor.scale_features(feature_df, target_column=None)[0]
        
        # Make prediction
        model = models[model_name]
        
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(feature_df)[0]
            prediction = model.predict(feature_df)[0]
        else:
            prediction = model.predict(feature_df)[0]
            prediction_proba = [1.0, 0.0] if prediction == 1 else [0.0, 1.0]
        
        result = {
            'prediction': int(prediction),
            'prediction_class': 'Parkinson' if prediction == 1 else 'No Parkinson',
            'confidence': max(prediction_proba),
            'probabilities': {
                'No Parkinson': float(prediction_proba[0]),
                'Parkinson': float(prediction_proba[1])
            },
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Generate explanations for a prediction."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'random_forest')
        features = data.get('features', {})
        explanation_type = data.get('type', 'shap')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        if not explainer:
            return jsonify({'error': 'Model explainer not initialized'}), 400
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Preprocess features if preprocessor is available
        if preprocessor:
            feature_df = preprocessor.scale_features(feature_df, target_column=None)[0]
        
        # Generate explanations
        if explanation_type == 'shap':
            explanation = explainer.explain_with_shap(feature_df, sample_idx=0)
        elif explanation_type == 'lime':
            explanation = explainer.explain_with_lime(feature_df, sample_idx=0)
        else:
            return jsonify({'error': f'Explanation type {explanation_type} not supported'}), 400
        
        if explanation:
            # Convert numpy arrays to lists for JSON serialization
            if 'shap_values' in explanation:
                explanation['shap_values'] = explanation['shap_values'].tolist()
            if 'sample' in explanation:
                explanation['sample'] = explanation['sample'].to_dict('records')
            
            return jsonify({
                'explanation': explanation,
                'type': explanation_type,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Could not generate explanation'}), 500
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for batch predictions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        file.save(filepath)
        
        # Load and preprocess data
        data = pd.read_csv(filepath)
        
        # Basic validation
        if 'status' not in data.columns:
            return jsonify({'error': 'CSV must contain a "status" column'}), 400
        
        # Return data summary
        summary = {
            'filename': filename,
            'rows': len(data),
            'columns': len(data.columns),
            'features': list(data.columns),
            'target_distribution': data['status'].value_counts().to_dict(),
            'message': 'File uploaded successfully'
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Make batch predictions on uploaded data."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_name = data.get('model', 'random_forest')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Load the uploaded file
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        data = pd.read_csv(filepath)
        X = data.drop(columns=['status'])
        y_true = data['status']
        
        # Preprocess features if preprocessor is available
        if preprocessor:
            X = preprocessor.scale_features(data, target_column='status')[0]
        
        # Make predictions
        model = models[model_name]
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
            y_pred = model.predict(X)
        else:
            y_pred = model.predict(X)
            y_pred_proba = None
        
        # Calculate metrics
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Create results summary
        results = {
            'filename': filename,
            'model_used': model_name,
            'n_samples': len(data),
            'predictions': y_pred.tolist(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_file = save_model_results(model_name, results, RESULTS_DIR)
        
        return jsonify({
            'message': 'Batch prediction completed',
            'results_file': results_file,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain_batch', methods=['POST'])
def explain_batch():
    """Generate explanations for batch predictions."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_name = data.get('model', 'random_forest')
        sample_indices = data.get('sample_indices', [0, 1, 2])  # Explain first 3 samples
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        if not explainer:
            return jsonify({'error': 'Model explainer not initialized'}), 400
        
        # Load the uploaded file
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        data = pd.read_csv(filepath)
        X = data.drop(columns=['status'])
        
        # Preprocess features if preprocessor is available
        if preprocessor:
            X = preprocessor.scale_features(data, target_column='status')[0]
        
        # Generate explanations for selected samples
        explanations = {}
        for idx in sample_indices:
            if idx < len(X):
                sample_explanations = {}
                
                # SHAP explanation
                shap_expl = explainer.explain_with_shap(X, sample_idx=idx)
                if shap_expl:
                    sample_explanations['shap'] = {
                        'shap_values': shap_expl['shap_values'].tolist() if 'shap_values' in shap_expl else None,
                        'type': shap_expl.get('type', 'unknown')
                    }
                
                # LIME explanation
                lime_expl = explainer.explain_with_lime(X, sample_idx=idx)
                if lime_expl:
                    sample_explanations['lime'] = {
                        'feature_importance': lime_expl.get('feature_importance', []),
                        'sample_idx': lime_expl.get('sample_idx', idx)
                    }
                
                explanations[idx] = sample_explanations
        
        return jsonify({
            'explanations': explanations,
            'n_samples_explained': len(explanations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch explanation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a new model (for development/testing purposes)."""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'ml')  # 'ml' or 'dl'
        model_name = data.get('model_name', 'random_forest')
        
        # Load and preprocess data
        data_file = os.path.join(DATA_DIR, 'parkinsons_disease_data.csv')
        if not os.path.exists(data_file):
            return jsonify({'error': 'Training data not found'}), 400
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocessor.preprocess_pipeline(data_file)
        
        if model_type == 'ml':
            # Train ML model
            trainer = MLModelTrainer()
            model, training_time = trainer.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            metrics = trainer.evaluate_model(model_name, X_test, y_test)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
            trainer.save_model(model_name, model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
            
        elif model_type == 'dl':
            # Train DNN model
            trainer = DLModelTrainer()
            model = trainer.train_dnn_model(model_name, X_train.values, y_train.values)
            
            # Evaluate model
            metrics = trainer.evaluate_dnn_model(model_name, X_test.values, y_test.values)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.h5')
            trainer.save_model(model_name, model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
        
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Reload models
        load_models()
        initialize_explainer()
        
        return jsonify({
            'message': f'Model {model_name} trained successfully',
            'model_path': model_path,
            'metrics': metrics,
            'training_time': training_time if model_type == 'ml' else 'N/A'
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

def save_model_results(model_name, results, output_dir):
    """Save model results to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model results saved to {filepath}")
    return filepath

if __name__ == '__main__':
    # Load models on startup
    load_models()
    initialize_explainer()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
