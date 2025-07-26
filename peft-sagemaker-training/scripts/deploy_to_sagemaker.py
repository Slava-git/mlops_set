import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
import wandb
import boto3
import tarfile
import os
import tempfile


def deploy_serverless_endpoint():
    """Deploy model as serverless endpoint to SageMaker"""
    
    print("Starting SageMaker Serverless deployment...")
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = "arn:aws:iam::262435404301:role/SageMaker-ExecutionRole"
    
    print("Downloading model from W&B...")
    wandb.init(project="peft-sagemaker", job_type="deployment")
    
    artifact = wandb.use_artifact("lmsys-monitored-model:latest")
    model_dir = artifact.download()
    
    wandb.finish()
    
    # Create model.tar.gz properly
    print("📦 Creating model.tar.gz...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "model.tar.gz")
        
        # Create tar.gz with all model files
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to tar with relative path
                    arcname = os.path.relpath(file_path, model_dir)
                    tar.add(file_path, arcname=arcname)
                    print(f"  Added: {arcname}")
        
        print("📤 Uploading model.tar.gz to S3...")
        bucket = sagemaker_session.default_bucket()
        key = "models/lmsys-full-classifier/model.tar.gz"
        
        s3_client = boto3.client('s3')
        s3_client.upload_file(tar_path, bucket, key)
        
        model_data = f"s3://{bucket}/{key}"
        print(f"📦 Model uploaded to: {model_data}")
    

    print("🤖 Creating PyTorch model...")
    pytorch_model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir="src",
        framework_version="2.0.0",
        py_version="py310",
        name=f"lmsys-full-classifier-{int(__import__('time').time())}"
    )
    
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=1
    )
    
    print("🚀 Deploying serverless endpoint...")
    endpoint_name = f"lmsys-serverless-{int(__import__('time').time())}"
    
    try:
        predictor = pytorch_model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name
        )
        
        print(f"✅ Serverless endpoint deployed: {predictor.endpoint_name}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return None

if __name__ == "__main__":
    predictor = deploy_serverless_endpoint()
    if predictor:
        print(f"\n🔗 Endpoint name: {predictor.endpoint_name}")
        print("📝 Update test script with this endpoint name")