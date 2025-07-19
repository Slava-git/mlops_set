
import json
import os


sample_data = {
    "train": [
        {"text": "This product is amazing! I love it.", "label": 1},
        {"text": "Great quality and fast shipping.", "label": 1},
        {"text": "Excellent customer service.", "label": 1},
        {"text": "Best purchase I've made this year.", "label": 1},
        {"text": "Terrible product, complete waste of money.", "label": 0},
        {"text": "Poor quality, broke after one day.", "label": 0},
        {"text": "Worst customer service ever.", "label": 0},
        {"text": "Would not recommend to anyone.", "label": 0},
    ],
    "validation": [
        {"text": "Pretty good product overall.", "label": 1},
        {"text": "Not bad but could be better.", "label": 0},
    ]
}

# Save as JSON files
# os.makedirs("data/local", exist_ok=True)

with open("data/local/train.json", "w") as f:
    json.dump(sample_data["train"], f, indent=2)

with open("data/local/validation.json", "w") as f:
    json.dump(sample_data["validation"], f, indent=2)

print("✅ Sample data created!")