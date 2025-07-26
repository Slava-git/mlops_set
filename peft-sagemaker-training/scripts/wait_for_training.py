import boto3
import time
import sys


def wait_for_training_completion():
    """Wait for the most recent training job to complete"""
    
    sagemaker = boto3.client('sagemaker')
    
    response = sagemaker.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not response['TrainingJobSummaries']:
        print("No training jobs found")
        sys.exit(1)
    
    job_name = response['TrainingJobSummaries'][0]['TrainingJobName']
    print(f"Waiting for training job: {job_name}")
    
    while True:
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        
        print(f"Status: {status}")
        
        if status == 'Completed':
            print("Training completed successfully!")
            break
        elif status in ['Failed', 'Stopped']:
            print(f"Training failed with status: {status}")
            sys.exit(1)
        
        time.sleep(60)

if __name__ == "__main__":
    wait_for_training_completion()