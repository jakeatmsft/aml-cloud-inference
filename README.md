# Azure ML v2 Cloud Inference Example

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Azure ML SDK v2](https://img.shields.io/badge/Azure%20ML%20SDK-v2-green.svg)](https://docs.microsoft.com/en-us/azure/machine-learning/)

A comprehensive example demonstrating how to deploy machine learning models for cloud inference using Azure Machine Learning SDK v2. This repository provides end-to-end examples, from model training to deployment and testing.

## 🚀 Features

- **Complete ML Pipeline**: Train, register, deploy, and test models
- **Azure ML v2 SDK**: Uses the latest Azure ML SDK v2 features
- **Multiple Input Formats**: Supports various data input formats for inference
- **Production Ready**: Includes proper error handling, logging, and best practices
- **Interactive Examples**: Jupyter notebooks and Python scripts
- **Easy Cleanup**: Scripts to manage and clean up Azure resources

## 📋 Prerequisites

Before getting started, ensure you have:

1. **Azure Subscription**: An active Azure subscription
2. **Azure ML Workspace**: A configured Azure Machine Learning workspace
3. **Python 3.9+**: Python 3.9 or higher installed
4. **Azure CLI** (optional): For additional Azure operations

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jakeatmsft/aml-cloud-inference.git
   cd aml-cloud-inference
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Azure credentials**:
   ```bash
   # Copy the template and fill in your details
   cp .env.template .env
   
   # Edit .env with your Azure details:
   # AZURE_SUBSCRIPTION_ID=your-subscription-id
   # AZURE_RESOURCE_GROUP=your-resource-group
   # AZURE_ML_WORKSPACE_NAME=your-workspace-name
   ```

## 📁 Project Structure

```
aml-cloud-inference/
├── src/                           # Source code
│   ├── aml_cloud_inference.py     # Main AML client class
│   ├── score.py                   # Scoring script for endpoints
│   ├── conda.yml                  # Environment definition
│   └── model/                     # Model-related code
│       └── train_model.py         # Sample model training
├── examples/                      # Python script examples
│   ├── deploy_model.py            # Complete deployment example
│   ├── test_inference.py          # Endpoint testing example
│   └── cleanup_resources.py       # Resource cleanup script
├── notebooks/                     # Jupyter notebook examples
│   └── aml_cloud_inference_example.ipynb
├── requirements.txt               # Python dependencies
├── .env.template                  # Configuration template
└── README.md                      # This file
```

## 🎯 Quick Start

### Option 1: Using Python Scripts

1. **Deploy a complete example**:
   ```bash
   python examples/deploy_model.py
   ```

2. **Test the deployed endpoint**:
   ```bash
   python examples/test_inference.py
   ```

3. **Clean up resources when done**:
   ```bash
   python examples/cleanup_resources.py
   ```

### Option 2: Using Jupyter Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/aml_cloud_inference_example.ipynb
   ```

2. **Follow the interactive notebook** for step-by-step guidance.

### Option 3: Using the AML Client Directly

```python
from src.aml_cloud_inference import AMLCloudInference

# Initialize client
client = AMLCloudInference(
    subscription_id="your-subscription-id",
    resource_group="your-resource-group", 
    workspace_name="your-workspace-name"
)

# List existing endpoints
endpoints = client.list_endpoints()
print(f"Found {len(endpoints)} endpoints")

# Deploy a model (requires trained model)
endpoint = client.create_endpoint("my-endpoint")
deployment = client.create_deployment(
    endpoint_name="my-endpoint",
    deployment_name="my-deployment",
    model_name="my-model",
    model_version="1",
    code_path="src",
    scoring_script="score.py"
)

# Test inference
result = client.invoke_endpoint("my-endpoint", {
    "data": [[1.0, 2.0, 3.0, 4.0, 5.0]]
})
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_SUBSCRIPTION_ID` | Your Azure subscription ID | Yes |
| `AZURE_RESOURCE_GROUP` | Azure resource group name | Yes |
| `AZURE_ML_WORKSPACE_NAME` | Azure ML workspace name | Yes |
| `AZURE_CLIENT_ID` | Service principal client ID | No* |
| `AZURE_CLIENT_SECRET` | Service principal secret | No* |
| `AZURE_TENANT_ID` | Azure tenant ID | No* |

*Required for automated/CI deployments. For interactive use, Azure CLI or browser authentication is used.

### Supported Input Formats

The scoring script supports multiple input formats:

1. **Standard format**:
   ```json
   {
     "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
   }
   ```

2. **Dictionary with inputs key**:
   ```json
   {
     "inputs": {
       "feature1": [1.0, 4.0],
       "feature2": [2.0, 5.0],
       "feature3": [3.0, 6.0]
     }
   }
   ```

3. **Direct list format**:
   ```json
   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
   ```

4. **Direct dictionary format**:
   ```json
   {
     "feature1": [1.0, 4.0],
     "feature2": [2.0, 5.0], 
     "feature3": [3.0, 6.0]
   }
   ```

## 📖 API Reference

### AMLCloudInference Class

#### Methods

- `register_model(model_name, model_path, description)`: Register a model
- `create_endpoint(endpoint_name, description)`: Create a managed endpoint
- `create_deployment(...)`: Deploy a model to an endpoint
- `set_traffic(endpoint_name, deployment_name, percentage)`: Set traffic allocation
- `invoke_endpoint(endpoint_name, input_data)`: Make predictions
- `delete_endpoint(endpoint_name)`: Delete an endpoint
- `list_endpoints()`: List all endpoints

### Scoring Script (score.py)

#### Required Functions

- `init()`: Initialize the model (called once when container starts)
- `run(raw_data)`: Process inference requests (called for each request)

## 🧪 Testing

The repository includes comprehensive testing examples:

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test end-to-end workflows
3. **Performance Tests**: Measure endpoint response times
4. **Format Tests**: Validate different input formats

Run tests with:
```bash
python examples/test_inference.py
```

## 🚀 Deployment Options

### Instance Types

Common instance types for different workloads:

| Instance Type | vCPUs | RAM | Use Case |
|---------------|-------|-----|----------|
| `Standard_DS3_v2` | 4 | 14 GB | General purpose |
| `Standard_F4s_v2` | 4 | 8 GB | Compute optimized |
| `Standard_NC6s_v3` | 6 | 112 GB | GPU workloads |

### Scaling Options

- **Instance Count**: Scale horizontally by increasing replica count
- **Auto-scaling**: Configure automatic scaling based on metrics
- **Blue-Green Deployments**: Use traffic allocation for safe deployments

## 🔍 Monitoring and Troubleshooting

### Monitoring

- **Azure ML Studio**: Monitor endpoints, deployments, and metrics
- **Application Insights**: Detailed logging and telemetry
- **Custom Metrics**: Add custom monitoring in scoring script

### Common Issues

1. **Authentication Errors**: Ensure Azure credentials are properly configured
2. **Model Loading Errors**: Check model file paths and dependencies
3. **Timeout Issues**: Increase instance size or optimize model code
4. **Memory Errors**: Use larger instance types or optimize data processing

### Debugging

Enable detailed logging in scoring script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🧹 Resource Management

### Cleanup Resources

**Important**: Remember to clean up Azure resources to avoid unnecessary charges:

```bash
# Using the cleanup script
python examples/cleanup_resources.py

# Or manually
python -c "
from src.aml_cloud_inference import AMLCloudInference
client = AMLCloudInference('sub-id', 'rg', 'workspace')
client.delete_endpoint('endpoint-name')
"
```

### Cost Optimization

- Delete unused endpoints and deployments
- Use appropriate instance types for your workload
- Consider spot instances for development/testing
- Monitor usage with Azure Cost Management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML SDK v2 Reference](https://docs.microsoft.com/en-us/python/api/overview/azure/ai-ml-readme)
- [Managed Online Endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/concept-endpoints)
- [Azure ML Samples](https://github.com/Azure/azureml-examples)

## 📞 Support

If you encounter issues or have questions:

1. Check the [troubleshooting section](#monitoring-and-troubleshooting)
2. Review [Azure ML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
3. Open an issue in this repository
4. Contact Azure Support for Azure-specific issues

---

**Happy ML Deploying! 🚀**