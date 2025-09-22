#!/usr/bin/env python3
"""
Script to clean up Azure ML resources created by the examples.
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from aml_cloud_inference import AMLCloudInference


def main():
    """Clean up Azure ML resources."""
    # Load environment variables
    load_dotenv()
    
    # Configuration
    config = {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
        "workspace_name": os.getenv("AZURE_ML_WORKSPACE_NAME"),
    }
    
    endpoint_name = "sample-classifier-endpoint"
    
    print("🧹 Cleaning up Azure ML resources...")
    print("=" * 40)
    
    try:
        # Initialize client
        client = AMLCloudInference(**config)
        
        # List existing endpoints
        endpoints = client.list_endpoints()
        print(f"\nFound {len(endpoints)} endpoints:")
        for endpoint in endpoints:
            print(f"  - {endpoint['name']}")
        
        # Delete the sample endpoint if it exists
        endpoint_exists = any(ep["name"] == endpoint_name for ep in endpoints)
        if endpoint_exists:
            print(f"\n🗑️  Deleting endpoint '{endpoint_name}'...")
            client.delete_endpoint(endpoint_name)
            print("✅ Endpoint deleted successfully!")
        else:
            print(f"\n📋 Endpoint '{endpoint_name}' not found, nothing to clean up.")
        
        print("\n✨ Cleanup completed!")
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())