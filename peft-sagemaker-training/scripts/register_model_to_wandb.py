import wandb
import os


def register_clean_peft_model():
    """Register clean PEFT model with only necessary files"""
    
    print("🔍 Checking model files...")
    
    model_path = "./model"
    if os.path.exists(model_path):
        print("📋 Files in model directory:")
        for file in sorted(os.listdir(model_path)):
            print(f"  {file}")
    else:
        print("❌ model directory not found")
        return
    
    # Initialize wandb
    wandb.init(project="peft-sagemaker")
    
    # Create artifact
    model_artifact = wandb.Artifact('lmsys-full-model', type='model', description='model fine-tuned on 3 classes')
    
    # Add model directory
    model_artifact.add_dir(model_path)
    
    # Log artifact
    wandb.log_artifact(model_artifact)
    
    print("✅ Clean PEFT model registered")
    wandb.finish()

if __name__ == "__main__":
    register_clean_peft_model()