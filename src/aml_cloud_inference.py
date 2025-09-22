"""
Azure ML v2 Cloud Inference Example
===================================

This module provides an example of how to deploy and use models for cloud inference
using Azure Machine Learning SDK v2.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from typing import Dict, Any, List


class AMLCloudInference:
    """
    Azure ML Cloud Inference client for deploying and using models.
    """
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initialize the AML client.
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Azure resource group name
            workspace_name: Azure ML workspace name
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        # Initialize ML client with default Azure credentials
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
    
    def register_model(self, model_name: str, model_path: str, description: str = "") -> Model:
        """
        Register a model in Azure ML workspace.
        
        Args:
            model_name: Name for the registered model
            model_path: Local path to the model file
            description: Description of the model
            
        Returns:
            Registered model object
        """
        model = Model(
            name=model_name,
            path=model_path,
            description=description,
            type="custom_model",
        )
        
        return self.ml_client.models.create_or_update(model)
    
    def create_endpoint(self, endpoint_name: str, description: str = "") -> ManagedOnlineEndpoint:
        """
        Create a managed online endpoint.
        
        Args:
            endpoint_name: Name for the endpoint
            description: Description of the endpoint
            
        Returns:
            Created endpoint object
        """
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description,
            auth_mode="key",
        )
        
        return self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    def create_deployment(
        self,
        endpoint_name: str,
        deployment_name: str,
        model_name: str,
        model_version: str,
        code_path: str,
        scoring_script: str,
        environment_name: str = None,
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
    ) -> ManagedOnlineDeployment:
        """
        Create a deployment for the endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            deployment_name: Name for the deployment
            model_name: Name of the registered model
            model_version: Version of the model
            code_path: Path to the code directory
            scoring_script: Name of the scoring script
            environment_name: Name of the environment (optional)
            instance_type: VM instance type
            instance_count: Number of instances
            
        Returns:
            Created deployment object
        """
        # Create or use existing environment
        if environment_name:
            environment = Environment(name=environment_name)
        else:
            environment = Environment(
                name=f"{deployment_name}-env",
                conda_file=os.path.join(code_path, "conda.yml"),
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            )
        
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"{model_name}:{model_version}",
            environment=environment,
            code_configuration=CodeConfiguration(
                code=code_path,
                scoring_script=scoring_script,
            ),
            instance_type=instance_type,
            instance_count=instance_count,
        )
        
        return self.ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    def set_traffic(self, endpoint_name: str, deployment_name: str, traffic_percentage: int = 100):
        """
        Set traffic allocation for a deployment.
        
        Args:
            endpoint_name: Name of the endpoint
            deployment_name: Name of the deployment
            traffic_percentage: Percentage of traffic to route to this deployment
        """
        endpoint = self.ml_client.online_endpoints.get(name=endpoint_name)
        endpoint.traffic = {deployment_name: traffic_percentage}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    def invoke_endpoint(self, endpoint_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the endpoint with input data.
        
        Args:
            endpoint_name: Name of the endpoint
            input_data: Input data for inference
            
        Returns:
            Prediction results
        """
        response = self.ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=None,
            deployment_name=None,
            request_data=json.dumps(input_data),
        )
        
        return json.loads(response)
    
    def delete_endpoint(self, endpoint_name: str):
        """
        Delete an endpoint and all its deployments.
        
        Args:
            endpoint_name: Name of the endpoint to delete
        """
        self.ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all endpoints in the workspace.
        
        Returns:
            List of endpoint information
        """
        endpoints = self.ml_client.online_endpoints.list()
        return [
            {
                "name": endpoint.name,
                "description": endpoint.description,
                "scoring_uri": endpoint.scoring_uri,
                "auth_mode": endpoint.auth_mode,
                "provisioning_state": endpoint.provisioning_state,
            }
            for endpoint in endpoints
        ]


def load_environment_config():
    """
    Load Azure ML configuration from environment variables.
    
    Returns:
        Dictionary with Azure ML configuration
    """
    return {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
        "workspace_name": os.getenv("AZURE_ML_WORKSPACE_NAME"),
    }


if __name__ == "__main__":
    # Example usage
    config = load_environment_config()
    
    if not all(config.values()):
        print("Please set the following environment variables:")
        print("- AZURE_SUBSCRIPTION_ID")
        print("- AZURE_RESOURCE_GROUP") 
        print("- AZURE_ML_WORKSPACE_NAME")
        exit(1)
    
    # Initialize client
    client = AMLCloudInference(**config)
    
    # List existing endpoints
    endpoints = client.list_endpoints()
    print(f"Found {len(endpoints)} endpoints:")
    for endpoint in endpoints:
        print(f"  - {endpoint['name']}: {endpoint['scoring_uri']}")