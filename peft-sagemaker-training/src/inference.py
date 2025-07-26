import json
import torch
import os
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import boto3 


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    cloudwatch = boto3.client("cloudwatch")
    logger.info("Cloud watch initalized")
except:
    cloudwatch = None
    logger.info("Cloud watch not initialized")


def model_fn(model_dir):
    """
    Load the full fine-tuned model for inference.
    """
    print(f"Loading full fine-tuned model from {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load full fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=3,
        torch_dtype=torch.float32
    )
    model.eval()
    
    print("Full fine-tuned model loaded successfully")
    return {"model": model, "tokenizer": tokenizer}


def send_metrics_to_cloudwatch(processing_time, input_length, success=True):
    """Send metrics to CloudWatch"""
    if not cloudwatch:
        logger.warning("CloudWatch not available, skipping metrics")
        return
    
    try:
        timestamp = datetime.utcnow()
        
        metrics = [
            {
                'MetricName': 'ProcessingTime',
                'Value': processing_time,
                'Unit': 'Seconds',
                'Timestamp': timestamp
            },
            {
                'MetricName': 'PredictionsPerMinute',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': timestamp
            },
            {
                'MetricName': 'InputLength',
                'Value': input_length,
                'Unit': 'Count',
                'Timestamp': timestamp
            }
        ]
        
        if success:
            metrics.append({
                'MetricName': 'SuccessfulPredictions',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': timestamp
            })
        else:
            metrics.append({
                'MetricName': 'FailedPredictions',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': timestamp
            })
        
        cloudwatch.put_metric_data(
            Namespace='SageMaker/ModelMonitoring',
            MetricData=metrics
        )
        
        logger.info(f"Metrics sent to CloudWatch - Processing time: {processing_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Failed to send metrics to CloudWatch: {e}")

def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions."""

    start_time = time.time()

    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    
    # Get input text
    if isinstance(input_data, dict):
        text = input_data.get("text", input_data.get("prompt", ""))
    else:
        text = str(input_data)
    
    input_length = len(text)

    logger.info(f"Processing request - Input length: {input_length} chars")

    try:

        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        processing_time = time.time() - start_time

        # Map classes to labels
        class_names = {0: "Model A wins", 1: "Model B wins", 2: "Tie"}

        send_metrics_to_cloudwatch(processing_time, input_length, success=True)

        logger.info(f"✅ Prediction completed - Time: {processing_time:.3f}s, "
                f"Prediction: {class_names[predicted_class]}, "
                f"Confidence: {confidence:.3f}")
            
        return {
            "predicted_class": predicted_class,
            "predicted_label": class_names[predicted_class],
            "confidence": float(confidence),
            "probabilities": {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            },
            "monitoring": {
                "processing_time": round(processing_time, 3),
                "input_length": input_length,
            }
        }

    except Exception as e:
        processing_time = time.time() - start_time
        send_metrics_to_cloudwatch(processing_time, input_length, success=False)
        logger.error(f"Prediction failed: {e}")
        raise e


def output_fn(prediction, accept):
    """Format the output."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")