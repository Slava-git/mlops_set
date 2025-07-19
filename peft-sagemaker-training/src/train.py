import argparse
import os
import json
import sys
import wandb


print("🐍 Python version:", sys.version)

try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError as e:
    print("PyTorch import error:", e)
    sys.exit(1)

try:
    import transformers
    print("Transformers version:", transformers.__version__)
except ImportError as e:
    print("Transformers import error:", e)
    sys.exit(1)

try:
    import peft
    print("PEFT version:", peft.__version__)
except ImportError as e:
    print("PEFT import error:", e)
    sys.exit(1)


from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Try to import PEFT components with error handling
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
    print("PEFT imports successful")
except ImportError as e:
    print(f"PEFT import error: {e}")
    PEFT_AVAILABLE = False

from datasets import Dataset
import numpy as np

try:
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available, using simple accuracy")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, 
                       default=os.environ.get('SM_MODEL_DIR', './local_output'))
    parser.add_argument('--train', type=str, 
                       default=os.environ.get('SM_CHANNEL_TRAIN', './data/local'))
    parser.add_argument('--validation', type=str, 
                       default=os.environ.get('SM_CHANNEL_VALIDATION', './data/local'))

    parser.add_argument('--wandb-project', type=str, default='peft-sagemaker')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true', default=True)
    
    # Hyperparameters
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--use-peft', action='store_true', default=True)
    
    return parser.parse_args()

def load_data(data_dir, split_name="train"):
    """Load data from directory"""
    print(f"📂 Loading {split_name} data from: {data_dir}")
    
    # Determine filename based on split
    if split_name == "train":
        filename = "train.json"
    elif split_name == "validation":
        filename = "validation.json"
    else:
        filename = f"{split_name}.json"
    
    data_file = os.path.join(data_dir, filename)
    
    if not os.path.exists(data_file):
        print(f"{split_name.capitalize()} file not found: {data_file}")
        print(f"Directory contents: {os.listdir(data_dir)}")
        raise FileNotFoundError(f"{split_name.capitalize()} file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data)} {split_name} samples")
    return Dataset.from_list(data)

def setup_model_and_tokenizer(model_name, use_peft=True):
    """Setup model with optional PEFT"""
    print(f"🤖 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        torch_dtype=torch.float32  # Explicit dtype to avoid issues
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        print("📝 Added padding token")
    
    # Apply PEFT if available and requested
    if use_peft and PEFT_AVAILABLE:
        try:
            print("📊 Applying PEFT (LoRA)...")
            
            # Conservative PEFT config
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=4,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=["q_lin", "v_lin"],
                bias="none",
                inference_mode=False,
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            print("✅ PEFT applied successfully")
            
        except Exception as e:
            print(f"⚠️ PEFT application failed: {e}")
            print("🔄 Falling back to full fine-tuning")
            use_peft = False
    
    if not use_peft or not PEFT_AVAILABLE:
        print("📚 Using full model fine-tuning")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer

def preprocess_data(examples, tokenizer, max_length):
    """Tokenize data"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None
    )
    tokenized["labels"] = examples["label"]
    return tokenized

def compute_metrics(eval_pred):
    """Compute accuracy"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(labels, predictions)
    else:
        # Simple accuracy calculation
        accuracy = (predictions == labels).mean()
    
    return {"accuracy": float(accuracy)}

def main():
    args = parse_args()
    
    print("Starting training...")
    print(f"Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Use PEFT: {args.use_peft}")
    print(f"   PEFT available: {PEFT_AVAILABLE}")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "use_peft": args.use_peft,
            }
        )
    
    try:
        # Load data
        print("\n📚 Loading datasets...")
        train_dataset = load_data(args.train, "train")
        val_dataset = load_data(args.validation, "validation")
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        
        # Show sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"📋 Sample: '{sample['text'][:100]}...' → {sample['label']}")
        
        # Setup model
        print("\n🤖 Setting up model...")
        model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_peft)
        
        # Preprocess data
        print("\n🔄 Preprocessing data...")
        train_dataset = train_dataset.map(
            lambda x: preprocess_data(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_data(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        print("Data preprocessing complete")
        
        # Training arguments - very conservative
        print("\nSetting up training...")
        training_args = TrainingArguments(
            output_dir=args.model_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            logging_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            disable_tqdm=False,  # Keep progress bars
            warmup_steps=0,
            report_to="wandb" if args.use_wandb else "none"
        )
        
        # Create trainer
        print("🏗️ Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        print("Training completed!")
        
        # Evaluate
        print("\nFinal evaluation...")
        eval_results = trainer.evaluate()
        print(f"📊 Results: {eval_results}")
        
        # Save
        print("\nSaving model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.model_dir)
        print(f"Model saved to: {args.model_dir}")
        
        print("\nTraining completed successfully!")

        if args.use_wandb:
            wandb.finish()
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()