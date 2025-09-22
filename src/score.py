"""
Scoring script for Azure ML managed online endpoint.

This script defines the init() and run() functions required by Azure ML
for model inference in a managed online endpoint.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def init():
    """
    Initialize the model. This function is called once when the container starts.
    """
    global model
    
    # Get the path to the registered model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), "model.pkl")
    
    # Load the model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def run(raw_data: str) -> str:
    """
    Run inference on input data.
    
    Args:
        raw_data: JSON string containing input data
        
    Returns:
        JSON string containing predictions
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Handle different input formats
        if "data" in data:
            # Standard format: {"data": [[feature1, feature2, ...], ...]}
            input_data = pd.DataFrame(data["data"])
        elif "inputs" in data:
            # Alternative format: {"inputs": {"feature1": [val1, val2], "feature2": [val3, val4]}}
            input_data = pd.DataFrame(data["inputs"])
        elif isinstance(data, list):
            # Direct list format: [[feature1, feature2, ...], ...]
            input_data = pd.DataFrame(data)
        elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            # Direct dict format: {"feature1": [val1, val2], "feature2": [val3, val4]}
            input_data = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported input format. Expected formats: "
                           "{'data': [...]} or {'inputs': {...}} or [...] or {...}")
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Handle different prediction types
        if hasattr(model, "predict_proba"):
            # If model supports probability prediction, include both
            probabilities = model.predict_proba(input_data)
            result = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist()
            }
        else:
            # Just predictions
            result = {
                "predictions": predictions.tolist()
            }
        
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(error_msg)
        return json.dumps({"error": error_msg})


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data if needed.
    
    Args:
        data: Input dataframe
        
    Returns:
        Preprocessed dataframe
    """
    # Add any preprocessing steps here
    # For example: scaling, encoding, feature engineering
    return data


def postprocess_predictions(predictions: np.ndarray) -> List:
    """
    Postprocess predictions if needed.
    
    Args:
        predictions: Raw model predictions
        
    Returns:
        Processed predictions
    """
    # Add any postprocessing steps here
    # For example: scaling back, decoding, formatting
    return predictions.tolist()