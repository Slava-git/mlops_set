import json
import os

def test_data_exists():
    """Test that training data exists"""
    train_file = "peft-sagemaker-training/data/local/train.json"
    val_file = "peft-sagemaker-training/data/local/validation.json"
    
    assert os.path.exists(train_file), f"Training data not found: {train_file}"
    assert os.path.exists(val_file), f"Validation data not found: {val_file}"
    
    print("Data files exist")

def test_data_format():
    """Test data format is correct"""
    with open("peft-sagemaker-training/data/local/train.json", "r") as f:
        data = json.load(f)
    
    assert len(data) > 0, "Training data is empty"
    
    sample = data[0]
    assert "text" in sample, "Sample missing 'text' field"
    assert "label" in sample, "Sample missing 'label' field"
    assert sample["label"] in [0, 1, 2], f"Invalid label: {sample['label']}"
    
    print(f"Data format valid. {len(data)} samples found")

if __name__ == "__main__":
    test_data_exists()
    test_data_format()