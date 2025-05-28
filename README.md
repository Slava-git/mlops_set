# LMSYS Human Labeling Dataset

## 🔧 Annotation Tool

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

## Video Demonstration

Watch the video demonstration of the annotation process:
[Label Studio Annotation Demo](https://www.loom.com/share/2b9fbea13de14d4cb421955e5f42fcbe?sid=66fe15ab-736a-47e6-8a4f-a8fa5b2d6dc5)

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
## Goal
This annotated dataset can be used for multiple machine learning applications:
Text Classification:
Multi-class classification of response quality (winner/loser/tie)

Alignment and Fine-tuning:
Reinforcement Learning from Human Feedback training or Dpo approach

