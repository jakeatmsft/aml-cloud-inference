# Cluster Command Job for AutoML Bank Marketing Inference

This example demonstrates how to run the AutoML model trained in the Bank Marketing notebook as a remote command job on a compute cluster.

## Prerequisites
- An Azure ML workspace with a CPU or GPU cluster named `cpu-cluster` (or update the compute name in `job.yaml`).
- The trained AutoML model registered in the workspace (e.g., name: `automl-bankmarketing-model`, version: `1`).
- Azure ML Python SDK v2 installed and [configured](https://aka.ms/azureml-setup).

## Files
- `environment.yml`: Conda environment definition for the job.
- `src/score.py`: Scoring script that downloads the registered model locally (using MLflow) and runs predictions.
- `job.yaml`: Command job definition for Azure ML, now sets `LOCAL_MODEL_PATH` to specify where the model artifacts are downloaded.
- `job.yaml`: Command job definition for Azure ML.

## Submit the job with CLI
```bash
az ml job create --file job.yaml
```

## Submit the job with Python SDK
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential

ml_client = MLClient(DefaultAzureCredential(),
                      subscription_id="<SUBSCRIPTION_ID>",
                      resource_group_name="<RESOURCE_GROUP>",
                      workspace_name="<WORKSPACE_NAME>")

job = ml_client.jobs.create_or_update(
    Job.load("job.yaml")
)
ml_client.jobs.stream(job.name)
``` 