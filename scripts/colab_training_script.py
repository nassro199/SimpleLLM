"""
Google Colab Training Script for SimpleLLM

This script provides a simplified way to train the SimpleLLM model in Google Colab.
Copy and paste this entire script into a Colab notebook cell and run it.
"""

import os
import torch
import logging
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Check GPU availability
print("Checking GPU availability...")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("No GPU available. Please change runtime type to include a GPU.")
    raise RuntimeError("GPU is required for training.")

# Clone repository and install dependencies
print("\nSetting up environment...")
# Run these commands in Colab:
# !git clone https://github.com/nassro199/SimpleLLM.git
# %cd SimpleLLM
# !pip install -q -r requirements.txt
# !pip install -q accelerate bitsandbytes sentencepiece datasets wandb

# Import necessary modules
print("\nImporting modules...")
from model.config import MoEConfig, TrainingConfig
from model.model import MoELLM
from data.dataset import load_and_prepare_datasets, create_dataloaders
from data.tokenizer import get_tokenizer
from training.trainer import MoETrainer, create_optimizer, create_lr_scheduler
from training.memory_utils import get_memory_stats, print_memory_stats, optimize_memory_efficiency, get_model_size
from utils.logging import configure_logging
from utils.checkpoint import save_model_for_inference

# Configure logging
logger = configure_logging(log_level="INFO", log_file="logs/training.log")

# Define model configuration
print("\nConfiguring model...")
model_config = MoEConfig(
    vocab_size=32000,
    hidden_size=768,  # Reduced for Colab
    intermediate_size=2048,  # Reduced for Colab
    num_hidden_layers=12,  # Reduced for Colab
    num_attention_heads=12,  # Reduced for Colab
    num_experts=8,
    num_experts_per_token=2,
    expert_capacity=0,  # Auto-calculate
    router_jitter_noise=0.1,
    router_z_loss_coef=0.001,
    router_aux_loss_coef=0.001,
    max_position_embeddings=2048,  # Reduced for Colab
    max_sequence_length=2048,  # Reduced for Colab
    hidden_dropout_prob=0.1,
    attention_dropout_prob=0.1,
    use_rms_norm=True,
    position_embedding_type="rotary",
    rotary_dim=64,  # Reduced for Colab
    use_mla=True,  # Multi-head Latent Attention
    mla_dim=64,  # Reduced for Colab
    use_mtp=True,  # Multi-Token Prediction
    mtp_num_tokens=2  # Reduced for Colab
)

# Define training configuration
training_config = TrainingConfig(
    batch_size=2,  # Small batch size for Colab
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch
    learning_rate=5e-5,
    weight_decay=0.01,
    max_steps=5000,  # Reduced for Colab
    warmup_steps=200,
    optimizer_type="8bit-adam",  # Memory-efficient optimizer
    lr_scheduler_type="cosine",
    use_gradient_checkpointing=True,  # Memory optimization
    mixed_precision="bf16",  # Memory optimization (use "fp16" if bf16 not supported)
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    max_seq_length=2048,  # Reduced for Colab
    preprocessing_num_workers=2  # Reduced for Colab
)

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = get_tokenizer(
    tokenizer_name_or_path="EleutherAI/gpt-neo-1.3B",  # Using an existing tokenizer
    use_fast=True
)

# Update model config with tokenizer vocab size
model_config.vocab_size = len(tokenizer)
print(f"Updated vocab size to {model_config.vocab_size}")

# Load dataset
print("\nLoading dataset...")
dataset_paths = ["roneneldan/TinyStories"]  # Small dataset for demonstration

datasets = load_and_prepare_datasets(
    tokenizer=tokenizer,
    dataset_paths=dataset_paths,
    max_seq_length=training_config.max_seq_length,
    streaming=False,  # Set to True for larger datasets
    text_column="text",
    preprocessing_num_workers=training_config.preprocessing_num_workers
)

# Create dataloaders
dataloaders = create_dataloaders(
    datasets=datasets,
    batch_size=training_config.batch_size,
    num_workers=training_config.preprocessing_num_workers
)

# Print dataset information
print("\nDataset Information:")
for split, dataset in datasets.items():
    print(f"  {split}: {len(dataset)} examples")

# Initialize model
print("\nInitializing model...")
model = MoELLM(model_config)

# Apply memory optimizations
model = optimize_memory_efficiency(model)

# Get model size information
model_size_info = get_model_size(model)
print("\nModel Size Information:")
print(f"  Total Parameters: {model_size_info['total_parameters'] / 1e6:.2f}M")
print(f"  Trainable Parameters: {model_size_info['trainable_parameters'] / 1e6:.2f}M")
print(f"  Parameter Size: {model_size_info['parameter_size_mb']:.2f} MB")
print(f"  Estimated Total Size: {model_size_info['estimated_total_size_mb']:.2f} MB")

# Create optimizer
print("\nSetting up training...")
optimizer = create_optimizer(
    model=model,
    learning_rate=training_config.learning_rate,
    weight_decay=training_config.weight_decay,
    optimizer_type=training_config.optimizer_type
)

# Calculate number of training steps
if hasattr(dataloaders["train"], "__len__"):
    num_training_steps = len(dataloaders["train"]) * training_config.max_steps
else:
    num_training_steps = training_config.max_steps

# Create learning rate scheduler
lr_scheduler = create_lr_scheduler(
    optimizer=optimizer,
    num_training_steps=num_training_steps,
    warmup_steps=training_config.warmup_steps,
    lr_scheduler_type=training_config.lr_scheduler_type
)

# Initialize trainer
trainer = MoETrainer(
    model=model,
    train_dataloader=dataloaders["train"],
    eval_dataloader=dataloaders.get("validation"),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    config=training_config,
    use_wandb=False  # Set to True if you want to use Weights & Biases
)

# Train the model
print("\nStarting training...")
trainer.train()

# Evaluate the model
print("\nEvaluating model...")
eval_metrics = trainer.evaluate()
print("\nEvaluation Metrics:")
for key, value in eval_metrics.items():
    print(f"  {key}: {value}")

# Generate text with the model
print("\nGenerating text samples...")
model.eval()

# Define prompts for text generation
prompts = [
    "Once upon a time, there was a",
    "The best way to learn is to",
    "In the future, artificial intelligence will"
]

# Generate text for each prompt
for prompt in prompts:
    print(f"\nPrompt: {prompt}")

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )

    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

# Save the model
print("\nSaving model...")
output_dir = "trained_model"
save_model_for_inference(
    model=model,
    tokenizer=tokenizer,
    output_dir=output_dir,
    save_format="pytorch",
    quantization="int8"  # Apply quantization for smaller model size
)
print(f"Model saved to {output_dir}")

# Zip the model directory for easier download
# Run these commands in Colab:
# !zip -r trained_model.zip trained_model
#
# # Download the model
# from google.colab import files
# files.download('trained_model.zip')

print("\nTraining complete! The model has been saved and is ready for download.")
