import wandb
import tempfile
import sys
import os


sys.path.append('src')
from inference import model_fn, predict_fn, input_fn

def test_local_inference():
    """Test inference locally before SageMaker deployment"""
    
    print("🧪 Testing inference locally...")
    
    print("Downloading model from W&B...")
    wandb.init(project="peft-sagemaker", job_type="local-test")
    
    artifact = wandb.use_artifact("lmsys-full-model:latest")

    model_dir = artifact.download()
    
    wandb.finish()
    
    print(f"📂 Model downloaded to: {model_dir}")
    
    # Test model loading
    print("🤖 Loading model...")
    try:
        model_artifacts = model_fn(model_dir)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    
    test_prompts = [
        "[\"Explain quantum computing in simple terms\"]<Model A>: [\"Quantum computing is a new type of computing that uses quantum bits, or qubits, which can be both 0 and 1 at the same time. This allows quantum computers to solve certain problems much faster than regular computers.\"]<Model B>: [\"Quantum computing is a method of computing that uses the principles of quantum mechanics to process information. It is especially useful for tasks like cryptography, simulation, and optimization.\"]",

        "[\"Write a Python function to calculate fibonacci numbers\"]<Model A>: [\"def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    else:\\n        return fibonacci(n-1) + fibonacci(n-2)\"]<Model B>: [\"def fibonacci(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a\"]",

        "[\"What are the main causes of climate change?\"]<Model A>: [\"The main causes of climate change are the burning of fossil fuels, deforestation, and industrial activities that release greenhouse gases into the atmosphere.\"]<Model B>: [\"Climate change is primarily driven by human activities such as emissions from cars and factories, cutting down forests, and large-scale agriculture.\"]",

        "[\"How do neural networks learn from data?\"]<Model A>: [\"Neural networks learn from data by adjusting the weights of connections between neurons based on the error in their predictions, using a method called backpropagation.\"]<Model B>: [\"Neural networks learn by processing input data through layers of interconnected nodes, comparing outputs to known results, and refining their internal parameters to reduce errors.\"]"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        try:
            # Simulate SageMaker's input processing
            input_data = {"text": prompt}
            
            # Make prediction
            result = predict_fn(input_data, model_artifacts)
            
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.3f}")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Local testing completed!")

if __name__ == "__main__":
    test_local_inference()