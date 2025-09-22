#!/usr/bin/env python3
"""
Example script showing how to invoke an existing Azure ML endpoint for inference.
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aml_cloud_inference import AMLCloudInference


def main():
    """Test endpoint inference with various input formats."""
    # Load environment variables
    load_dotenv()
    
    # Configuration
    config = {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
        "workspace_name": os.getenv("AZURE_ML_WORKSPACE_NAME"),
    }
    
    endpoint_name = "sample-classifier-endpoint"
    
    print("🧪 Testing Azure ML Endpoint Inference")
    print("=" * 40)
    
    try:
        # Initialize client
        client = AMLCloudInference(**config)
        
        # Check if endpoint exists
        endpoints = client.list_endpoints()
        endpoint_exists = any(ep["name"] == endpoint_name for ep in endpoints)
        
        if not endpoint_exists:
            print(f"❌ Endpoint '{endpoint_name}' not found!")
            print("Please run 'python examples/deploy_model.py' first to create the endpoint.")
            return 1
        
        print(f"✅ Found endpoint '{endpoint_name}'")
        
        # Create sample data for testing
        print("\n📊 Creating test data...")
        test_data = {
            "feature_0": [1.2, -0.5, 2.1],
            "feature_1": [0.8, 1.3, -1.2],
            "feature_2": [-0.3, 0.9, 0.4],
            "feature_3": [1.1, -0.8, 1.7],
            "feature_4": [0.2, 1.5, -0.6],
            "feature_5": [-1.0, 0.3, 0.8],
            "feature_6": [0.7, -1.2, 1.3],
            "feature_7": [1.4, 0.6, -0.9],
            "feature_8": [-0.2, 0.1, 1.6],
            "feature_9": [0.9, -0.4, 0.5]
        }
        
        # Test different input formats
        test_cases = [
            {
                "name": "Standard format with 'data' key",
                "input": {
                    "data": [
                        [1.2, 0.8, -0.3, 1.1, 0.2, -1.0, 0.7, 1.4, -0.2, 0.9],
                        [-0.5, 1.3, 0.9, -0.8, 1.5, 0.3, -1.2, 0.6, 0.1, -0.4],
                        [2.1, -1.2, 0.4, 1.7, -0.6, 0.8, 1.3, -0.9, 1.6, 0.5]
                    ]
                }
            },
            {
                "name": "Dictionary format with 'inputs' key",
                "input": {
                    "inputs": test_data
                }
            },
            {
                "name": "Direct list format",
                "input": [
                    [1.2, 0.8, -0.3, 1.1, 0.2, -1.0, 0.7, 1.4, -0.2, 0.9],
                    [-0.5, 1.3, 0.9, -0.8, 1.5, 0.3, -1.2, 0.6, 0.1, -0.4]
                ]
            },
            {
                "name": "Direct dictionary format",
                "input": {
                    key: values[:2] for key, values in test_data.items()
                }
            }
        ]
        
        # Test each format
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔬 Test {i}: {test_case['name']}")
            try:
                result = client.invoke_endpoint(endpoint_name, test_case["input"])
                
                if "error" in result:
                    print(f"  ❌ Error: {result['error']}")
                else:
                    predictions = result.get("predictions", [])
                    probabilities = result.get("probabilities", [])
                    
                    print(f"  ✅ Success!")
                    print(f"  📊 Predictions: {predictions}")
                    if probabilities:
                        print(f"  📈 Probabilities: {probabilities}")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        # Performance test
        print(f"\n⚡ Performance Test")
        print("Making 10 rapid requests...")
        
        simple_input = {"data": [[1.2, 0.8, -0.3, 1.1, 0.2, -1.0, 0.7, 1.4, -0.2, 0.9]]}
        
        import time
        start_time = time.time()
        successful_requests = 0
        
        for i in range(10):
            try:
                result = client.invoke_endpoint(endpoint_name, simple_input)
                if "predictions" in result:
                    successful_requests += 1
            except Exception as e:
                print(f"  Request {i+1} failed: {str(e)}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"  ✅ {successful_requests}/10 requests successful")
        print(f"  ⏱️  Total time: {duration:.2f} seconds")
        print(f"  📊 Average time per request: {duration/10:.2f} seconds")
        
        print("\n✨ Inference testing completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())