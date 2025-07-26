import boto3
import json

def test_sagemaker_endpoint():
    """Test the deployed SageMaker endpoint"""
    
    runtime = boto3.client('sagemaker-runtime')
    
    endpoint_name = "lmsys-serverless-1753546624"
    
    test_prompts = [
        "[\"Explain quantum computing in simple terms\"]<Model A>: [\"Quantum computing is a new type of computing that uses quantum bits, or qubits, which can be both 0 and 1 at the same time. This allows quantum computers to solve certain problems much faster than regular computers.\"]<Model B>: [\"Quantum computing is a method of computing that uses the principles of quantum mechanics to process information. It is especially useful for tasks like cryptography, simulation, and optimization.\"]",

        "[\"Write a Python function to calculate fibonacci numbers\"]<Model A>: [\"def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    else:\\n        return fibonacci(n-1) + fibonacci(n-2)\"]<Model B>: [\"def fibonacci(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a\"]",

        "[\"What are the main causes of climate change?\"]<Model A>: [\"The main causes of climate change are the burning of fossil fuels, deforestation, and industrial activities that release greenhouse gases into the atmosphere.\"]<Model B>: [\"Climate change is primarily driven by human activities such as emissions from cars and factories, cutting down forests, and large-scale agriculture.\"]",

        "[\"How do neural networks learn from data?\"]<Model A>: [\"Neural networks learn from data by adjusting the weights of connections between neurons based on the error in their predictions, using a method called backpropagation.\"]<Model B>: [\"Neural networks learn by processing input data through layers of interconnected nodes, comparing outputs to known results, and refining their internal parameters to reduce errors.\"]"
    ]
    
    print("Testing SageMaker endpoint...")
    print(f"Endpoint: {endpoint_name}")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        try:
            # Prepare input
            payload = {"text": prompt}
            
            # Call endpoint
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n Endpoint testing completed!")

if __name__ == "__main__":
    test_sagemaker_endpoint()