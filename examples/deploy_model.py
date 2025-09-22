#!/usr/bin/env python3
"""
Complete example of deploying a model to Azure ML for cloud inference.

This script demonstrates the full workflow:
1. Train a sample model
2. Register the model in Azure ML
3. Create an endpoint
4. Deploy the model
5. Test the endpoint
6. Clean up resources
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aml_cloud_inference import AMLCloudInference
from model.train_model import create_sample_data, train_model, save_model_and_data


def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP", 
        "AZURE_ML_WORKSPACE_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in a .env file or environment.")
        return 1
    
    # Configuration
    config = {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
        "workspace_name": os.getenv("AZURE_ML_WORKSPACE_NAME"),
    }
    
    model_name = "sample-classifier"
    endpoint_name = "sample-classifier-endpoint"
    deployment_name = "sample-classifier-deployment"
    
    print("🚀 Starting Azure ML Cloud Inference Example")
    print("=" * 50)
    
    try:
        # Step 1: Create and train a sample model
        print("\n📊 Step 1: Creating and training sample model...")
        X, y = create_sample_data()
        model = train_model(X, y)
        save_model_and_data(model, X, y)
        print("✅ Model training completed!")
        
        # Step 2: Initialize Azure ML client
        print("\n🔗 Step 2: Connecting to Azure ML workspace...")
        client = AMLCloudInference(**config)
        print("✅ Connected to Azure ML workspace!")
        
        # Step 3: Register the model
        print(f"\n📝 Step 3: Registering model '{model_name}'...")
        registered_model = client.register_model(
            model_name=model_name,
            model_path="src/model",
            description="Sample RandomForest classifier for cloud inference example"
        )
        print(f"✅ Model registered with version: {registered_model.version}")
        
        # Step 4: Create endpoint
        print(f"\n🌐 Step 4: Creating endpoint '{endpoint_name}'...")
        endpoint = client.create_endpoint(
            endpoint_name=endpoint_name,
            description="Sample endpoint for cloud inference example"
        )
        print(f"✅ Endpoint created: {endpoint.scoring_uri}")
        
        # Step 5: Create deployment
        print(f"\n🚀 Step 5: Creating deployment '{deployment_name}'...")
        deployment = client.create_deployment(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            model_name=model_name,
            model_version=registered_model.version,
            code_path="src",
            scoring_script="score.py",
            instance_type="Standard_DS3_v2",
            instance_count=1
        )
        print("✅ Deployment created successfully!")
        
        # Step 6: Set traffic to 100%
        print("\n🔄 Step 6: Setting traffic allocation...")
        client.set_traffic(endpoint_name, deployment_name, 100)
        print("✅ Traffic set to 100% for the deployment!")
        
        # Step 7: Test the endpoint
        print("\n🧪 Step 7: Testing the endpoint...")
        
        # Load sample data
        import pandas as pd
        sample_data = pd.read_csv("src/model/sample_data.csv")
        
        # Test with different input formats
        test_cases = [
            {
                "name": "Standard format",
                "data": {"data": sample_data.iloc[:3].values.tolist()}
            },
            {
                "name": "Dictionary format", 
                "data": {"inputs": sample_data.iloc[:2].to_dict("list")}
            },
            {
                "name": "Simple list format",
                "data": sample_data.iloc[:1].values.tolist()
            }
        ]
        
        for test_case in test_cases:
            print(f"\n  Testing {test_case['name']}...")
            try:
                result = client.invoke_endpoint(endpoint_name, test_case["data"])
                print(f"  ✅ Success! Got {len(result.get('predictions', []))} predictions")
                if 'probabilities' in result:
                    print(f"      Probabilities included: {len(result['probabilities'])} sets")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        # Step 8: Display endpoint information
        print("\n📋 Step 8: Endpoint Information")
        endpoints = client.list_endpoints()
        for ep in endpoints:
            if ep["name"] == endpoint_name:
                print(f"  Name: {ep['name']}")
                print(f"  Scoring URI: {ep['scoring_uri']}")
                print(f"  Auth Mode: {ep['auth_mode']}")
                print(f"  Status: {ep['provisioning_state']}")
                break
        
        print("\n✨ Example completed successfully!")
        print("\n📝 Next steps:")
        print("  1. Use the scoring URI to make predictions from your applications")
        print("  2. Monitor the endpoint in Azure ML Studio")
        print("  3. Scale the deployment as needed")
        print(f"  4. Remember to clean up resources when done:")
        print(f"     python examples/cleanup_resources.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🧹 Attempting to clean up any created resources...")
        
        try:
            client = AMLCloudInference(**config)
            client.delete_endpoint(endpoint_name)
            print("✅ Endpoint cleaned up")
        except:
            print("⚠️  Could not clean up endpoint")
        
        return 1


if __name__ == "__main__":
    exit(main())