import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    
    print("✅ Full fine-tuned model loaded successfully")
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions."""
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    
    # Get input text
    if isinstance(input_data, dict):
        text = input_data.get("text", input_data.get("prompt", ""))
    else:
        text = str(input_data)
    
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
    
    # Map classes to labels
    class_names = {0: "Model A wins", 1: "Model B wins", 2: "Tie"}
    
    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": float(confidence),
        "probabilities": {
            class_names[i]: float(prob) 
            for i, prob in enumerate(probabilities[0])
        }
    }

def output_fn(prediction, accept):
    """Format the output."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")