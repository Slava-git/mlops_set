import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import os


def submit_peft_training_job():
    """Submit PEFT training job to SageMaker"""
    
    print("🚀 Preparing SageMaker training job...")

    session = boto3.Session(region_name='us-west-2')
    sagemaker_session = sagemaker.Session(boto_session=session)

    role = "arn:aws:iam::262435404301:role/SageMaker-ExecutionRole"
    
    bucket = "ml-bucket-vh-4543"
    s3_train_data = f"s3://{bucket}/data/train"
    s3_val_data = f"s3://{bucket}/data/validation"
    s3_output_path = f"s3://{bucket}/models"
    
    # Hyperparameters for your training
    hyperparameters = {
        'model-name': 'distilbert-base-uncased',
        'epochs': 2,
        'batch-size': 8,
        'learning-rate': 2e-4,
        'max-length': 256,
    }
    
    # Create PyTorch estimator
    pytorch_estimator = PyTorch(
        entry_point='train.py', 
        source_dir='src',
        role=role,
        instance_type='ml.m5.large',
        instance_count=1,
        framework_version='2.0.0',
        py_version='py310',
        hyperparameters=hyperparameters,
        output_path=s3_output_path,
        max_run=3600,
        volume_size=30,
        environment={
            'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
            'WANDB_API_KEY': os.environ.get('WANDB_API_KEY')
        }
    )
    
    training_inputs = {
        'train': TrainingInput(s3_train_data),
        'validation': TrainingInput(s3_val_data)
    }
    
    print("📋 Training Job Configuration:")
    print(f"  Instance type: ml.m5.large")
    print(f"  Training data: {s3_train_data}")
    print(f"  Validation data: {s3_val_data}")
    print(f"  Output path: {s3_output_path}")
    print(f"  Hyperparameters: {hyperparameters}")
    
    # Start training job
    print("🏗️ Starting training job...")
    pytorch_estimator.fit(training_inputs, wait=False)
    
    print(f"✅ Training job submitted!")
    print(f"📝 Job name: {pytorch_estimator.latest_training_job.name}")
    print(f"🔗 You can monitor it in the SageMaker console")
    
    return pytorch_estimator

if __name__ == "__main__":
    estimator = submit_peft_training_job()