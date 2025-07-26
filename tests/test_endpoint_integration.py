import boto3
import json
import time


def test_endpoint():
    """Test deployed endpoint"""
    
    runtime = boto3.client('sagemaker-runtime')
    
    sagemaker = boto3.client('sagemaker')
    endpoints = sagemaker.list_endpoints(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not endpoints['Endpoints']:
        print("No endpoints found")
        return False
    
    endpoint_name = endpoints['Endpoints'][0]['EndpointName']
    print(f"Testing endpoint: {endpoint_name}")
    
    test_payload = {"text": "What is machine learning?"}
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(test_payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        assert "predicted_class" in result
        assert "predicted_label" in result
        assert "confidence" in result
        
        print(f"Endpoint test passed: {result['predicted_label']}")
        return True
        
    except Exception as e:
        print(f"Endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    test_endpoint()