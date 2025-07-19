import boto3
import os

def upload_to_s3():
    s3 = boto3.client('s3')
    bucket_name = "ml-bucket-vh-4543"
    
    # Upload training data
    s3.upload_file(
        "data/local/train.json",
        bucket_name,
        "data/train/train.json"
    )
    
    # Upload validation data
    s3.upload_file(
        "data/local/validation.json",
        bucket_name,
        "data/validation/validation.json"
    )
    
    print("✅ Data uploaded to S3!")
    print(f"✅ Train data: s3://{bucket_name}/data/train/train.json")
    print(f"✅ Validation data: s3://{bucket_name}/data/validation/validation.json")

if __name__ == "__main__":
    upload_to_s3()