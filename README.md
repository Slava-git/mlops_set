# LMSYS Human Labeling Dataset

## Annotation Tool

[Label Studio](https://labelstud.io/) was used for response annotation, an open-source data annotation platform. The annotation was performed manually using a custom interface that compares `response_a` and `response_b` responses and determines the better one (or records a tie).

## How to Launch/Open Annotation

1. Start Label Studio using Docker Compose:
   ```bash
   docker compose up -d
   ```

2. Open Label Studio in your browser:
   ```
   http://localhost:9008
   ```

3. Log in or create an account.
4. Open the `LMSYS` project and review the data or start new annotation.

## Video Demonstrations

- **Annotation Process**: [Label Studio Annotation Demo](https://www.loom.com/share/2b9fbea13de14d4cb421955e5f42fcbe?sid=66fe15ab-736a-47e6-8a4f-a8fa5b2d6dc5)
- **PEFT Training**: [SageMaker Training with W&B Tracking](https://www.loom.com/share/c164419704ca4a9c9de8f86dd5412499?sid=ac4d85df-9f33-4cfb-a396-c408c3b2a643)

## Data Versioning

Versioning of annotated data is implemented using DVC with MinIO as an S3-compatible storage.

### Basic Commands:

**Add a new version:**
```bash
dvc add data/lmsys-annotations-v2.json
git add data/lmsys-annotations-v2.json.dvc
git commit -m "Annotation v2"
dvc push
```

**Download a specific version:**
```bash
git checkout v1
dvc checkout
```

## PEFT Training Pipeline

### Quick Start
```bash
# Configure AWS credentials
aws configure

# Prepare data for training
python peft-sagemaker-training/scripts/prepare_lmsys_data.py

# Upload to S3
python peft-sagemaker-training/scripts/upload_data.py

# Set Weights & Biases API key
export WANDB_API_KEY="your-wandb-api-key"

# Submit training job
python peft-sagemaker-training/scripts/submit_training.py
```

### Project Structure
```
peft-sagemaker-training/
├── src/train.py              # PEFT training script
├── scripts/                  # Data prep and submission
├── data/local/              # Local data storage
└── requirements.txt         # Dependencies
```

## Model Monitoring and Observability

### CloudWatch Metrics Dashboard
```bash
# Create monitoring dashboard
python peft-sagemaker-training/scripts/create_dashboard.py

# Generate test data for metrics
python peft-sagemaker-training/scripts/generate_test_data.py
```

### Monitoring Features
- **Predictions per Minute**: Real-time request volume tracking
- **Average Processing Time**: Latency monitoring for performance optimization
- **Success/Failure Rates**: Error rate monitoring and alerting
- **Input Data Analysis**: Text length and word count distribution tracking
- **CloudWatch Integration**: Native AWS monitoring with custom metrics

### Dashboard Access
View your monitoring dashboard at:
```
AWS Console → CloudWatch → Dashboards → LMSYS-Model-Monitoring
```

## Model Inference Deployment

### SageMaker Serverless Inference
```bash
# Register trained model to W&B Model Registry
python peft-sagemaker-training/scripts/register_model_to_wandb.py

# Deploy to SageMaker serverless endpoint
python peft-sagemaker-training/scripts/deploy_to_sagemaker.py

# Test the deployed endpoint
python peft-sagemaker-training/scripts/test_sagemaker_endpoint.py
```

### Local Inference Testing
```bash
# Test inference locally before deployment
python peft-sagemaker-training/scripts/test_local_inference.py
```

### Inference Features
- **W&B Model Registry**: Model versioning and artifact management
- **Serverless Deployment**: Pay-per-request, cost-effective for testing
- **3-Class Classification**: Predicts Model A wins, Model B wins, or Tie
- **REST API**: JSON input/output for easy integration
- **Real-time Monitoring**: Comprehensive metrics and alerting

## Goal

This annotated dataset can be used for multiple machine learning applications:

- **Text Classification**: Multi-class classification of response quality (winner/loser/tie)
- **Alignment and Fine-tuning**: Reinforcement Learning from Human Feedback training or DPO approach
- **PEFT Training**: Parameter-efficient fine-tuning on AWS SageMaker with experiment tracking
- **Model Deployment**: Serverless inference endpoints with W&B model registry integration
- **Production Monitoring**: Real-time observability and performance tracking