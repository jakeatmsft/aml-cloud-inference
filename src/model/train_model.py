"""
Sample model training script to create a simple model for demonstration.

This script creates a simple scikit-learn model that can be used
with the cloud inference example.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def create_sample_data(n_samples: int = 1000, n_features: int = 10, random_state: int = 42):
    """
    Create sample classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random state for reproducibility
        
    Returns:
        X, y: Features and target arrays
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    return X, y


def train_model(X, y):
    """
    Train a simple RandomForest model.
    
    Args:
        X: Features
        y: Target
        
    Returns:
        Trained model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model


def save_model_and_data(model, X, y, output_dir: str = "src/model"):
    """
    Save the trained model and sample data.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save sample data for testing
    sample_data = pd.DataFrame(X[:10], columns=[f"feature_{i}" for i in range(X.shape[1])])
    sample_data_path = os.path.join(output_dir, "sample_data.csv")
    sample_data.to_csv(sample_data_path, index=False)
    print(f"Sample data saved to: {sample_data_path}")
    
    # Save feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    feature_names_path = os.path.join(output_dir, "feature_names.txt")
    with open(feature_names_path, "w") as f:
        f.write("\n".join(feature_names))
    print(f"Feature names saved to: {feature_names_path}")


if __name__ == "__main__":
    print("Creating sample data...")
    X, y = create_sample_data()
    
    print("Training model...")
    model = train_model(X, y)
    
    print("Saving model and data...")
    save_model_and_data(model, X, y)
    
    print("Done! Model is ready for deployment.")