import joblib
import logging
import os
from datetime import datetime
import json
import numpy as np

def save_model(model, model_name, model_type="sklearn", output_dir="saved_models", metadata=None):
    """
    Save trained model with metadata
    
    Args:
        model: Trained model object
        model_name: Name of the model
        model_type: Type of model ('sklearn', 'xgboost', 'keras')
        output_dir: Directory to save the model
        metadata: Additional metadata to save with the model
    
    Returns:
        str: Path to saved model file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model based on type
    if model_type == "keras":
        model_path = f"{output_dir}/{model_name}_{timestamp}.h5"
        model.save(model_path)
    else:
        model_path = f"{output_dir}/{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'save_date': datetime.now().isoformat()
    })
    
    metadata_path = f"{output_dir}/{model_name}_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Model saved: {model_path}")
    logging.info(f"Metadata saved: {metadata_path}")
    
    return model_path

def load_model(model_path, model_type="sklearn"):
    """
    Load saved model
    
    Args:
        model_path: Path to saved model file
        model_type: Type of model ('sklearn', 'xgboost', 'keras')
    
    Returns:
        Loaded model object
    """
    if model_type == "keras":
        from keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
    else:
        model = joblib.load(model_path)
    
    logging.info(f"Model loaded: {model_path}")
    return model

def save_model_results(results_dict, output_dir="saved_models"):
    """
    Save model training results and metrics
    
    Args:
        results_dict: Dictionary containing model results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = f"{output_dir}/training_results_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results for JSON serialization
    serializable_results = {}
    for key, value in results_dict.items():
        if hasattr(value, 'tolist'):  # numpy array
            serializable_results[key] = value.tolist()
        elif hasattr(value, 'item'):  # numpy scalar
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Training results saved: {results_path}")
    return results_path

def create_model_registry(output_dir="saved_models"):
    """
    Create a registry of all saved models
    
    Args:
        output_dir: Directory containing saved models
    
    Returns:
        dict: Registry of saved models
    """
    registry = {}
    
    if not os.path.exists(output_dir):
        return registry
    
    # Look for model files
    for file in os.listdir(output_dir):
        if file.endswith('.pkl') or file.endswith('.h5'):
            model_name = file.split('_')[0]
            timestamp = file.split('_')[1].split('.')[0]
            
            if model_name not in registry:
                registry[model_name] = []
            
            registry[model_name].append({
                'file': file,
                'timestamp': timestamp,
                'path': os.path.join(output_dir, file)
            })
    
    # Save registry
    registry_path = f"{output_dir}/model_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logging.info(f"Model registry created: {registry_path}")
    return registry
