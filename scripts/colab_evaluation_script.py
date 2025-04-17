"""
Google Colab Evaluation Script for MoE LLM

This script provides a simplified way to evaluate a Mixture of Experts (MoE) LLM in Google Colab.
Copy and paste this entire script into a Colab notebook cell and run it.
"""

import os
import torch
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Check GPU availability
print("Checking GPU availability...")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("No GPU available. Please change runtime type to include a GPU.")
    raise RuntimeError("GPU is required for evaluation.")

# Clone repository and install dependencies
print("\nSetting up environment...")
!git clone https://github.com/your-username/moe-llm.git
%cd moe-llm
!pip install -q -r requirements.txt
!pip install -q accelerate bitsandbytes sentencepiece datasets matplotlib

# Import necessary modules
print("\nImporting modules...")
from model.config import MoEConfig
from model.model import MoELLM
from data.tokenizer import get_tokenizer
from evaluation.benchmarks import BenchmarkEvaluator
from evaluation.perplexity import calculate_perplexity
from utils.logging import configure_logging
from utils.visualization import plot_benchmark_results, plot_comparison_with_deepseek

# Configure logging
logger = configure_logging(log_level="INFO", log_file="logs/evaluation.log")

# Check if we have a trained model or need to upload one
print("\nChecking for model...")
if os.path.exists("trained_model"):
    print("Found existing model directory.")
    model_path = "trained_model"
elif os.path.exists("model.zip"):
    print("Found model zip file. Extracting...")
    !unzip -q model.zip
    model_path = "model"  # Adjust based on the extracted directory name
else:
    print("No model found. Please upload a model zip file or create a small demo model.")
    model_path = None
    
    # Option to upload a model
    from google.colab import files
    print("Please upload your model zip file:")
    uploaded = files.upload()
    
    if uploaded:
        print("Extracting uploaded model...")
        !unzip -q *.zip
        model_path = [d for d in os.listdir() if os.path.isdir(d) and (d.startswith("model") or d.startswith("trained"))][0]
    else:
        print("No model uploaded. Creating a small demo model.")

# Load or create model
if model_path and os.path.exists(model_path):
    print(f"\nLoading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = get_tokenizer(
        tokenizer_name_or_path=model_path,
        use_fast=True
    )
    
    # Load model configuration
    if os.path.exists(os.path.join(model_path, "config.json")):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        model_config = MoEConfig(**config_dict)
    else:
        # Create default config
        model_config = MoEConfig(
            vocab_size=len(tokenizer),
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_experts=8,
            num_experts_per_token=2
        )
    
    # Initialize model
    model = MoELLM(model_config)
    
    # Load model weights
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"))
    else:
        print("Warning: No model weights found. Using randomly initialized weights.")
else:
    print("\nCreating a small demo model...")
    
    # Load tokenizer from a pre-trained model
    tokenizer = get_tokenizer(
        tokenizer_name_or_path="EleutherAI/gpt-neo-1.3B",
        use_fast=True
    )
    
    # Create a smaller model for demonstration
    model_config = MoEConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,  # Very small for demonstration
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_experts=4,
        num_experts_per_token=2,
        max_position_embeddings=1024,
        max_sequence_length=1024
    )
    
    # Initialize model
    model = MoELLM(model_config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model to evaluation mode
model.eval()

# Print model information
print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

# Calculate perplexity
print("\nCalculating perplexity...")
try:
    from datasets import load_dataset
    validation_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Calculate perplexity on a subset of the data to save time
    perplexity_metrics = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=validation_dataset["text"][:100],  # Use only 100 examples
        device=device,
        batch_size=1,
        stride=512,
        max_length=1024
    )
    
    print(f"Perplexity: {perplexity_metrics['perplexity']:.4f}")
except Exception as e:
    print(f"Error calculating perplexity: {e}")
    perplexity_metrics = {"perplexity": float('nan')}

# Initialize benchmark evaluator
print("\nInitializing benchmark evaluator...")
evaluator = BenchmarkEvaluator(
    model=model,
    tokenizer=tokenizer,
    device=device,
    output_dir="benchmark_results"
)

# Initialize dictionary to store all metrics
all_metrics = {}
all_metrics["perplexity"] = perplexity_metrics["perplexity"]

# Evaluate on MMLU (using a small subset)
try:
    print("\nEvaluating on MMLU (subset)...")
    mmlu_metrics = evaluator.evaluate_mmlu(
        data_path="cais/mmlu",
        num_few_shot=2,  # Reduced for speed
        batch_size=2
    )
    
    print(f"MMLU accuracy: {mmlu_metrics['mmlu_accuracy']:.4f}")
    all_metrics.update(mmlu_metrics)
except Exception as e:
    print(f"Error evaluating on MMLU: {e}")
    all_metrics["mmlu_accuracy"] = float('nan')

# Evaluate on GSM8K (using a small subset)
try:
    print("\nEvaluating on GSM8K (subset)...")
    gsm8k_metrics = evaluator.evaluate_gsm8k(
        data_path="gsm8k",
        split="test",
        num_few_shot=2,  # Reduced for speed
        batch_size=2
    )
    
    print(f"GSM8K accuracy: {gsm8k_metrics['gsm8k_accuracy']:.4f}")
    all_metrics.update(gsm8k_metrics)
except Exception as e:
    print(f"Error evaluating on GSM8K: {e}")
    all_metrics["gsm8k_accuracy"] = float('nan')

# Generate text with the model
print("\nGenerating text samples...")
prompts = [
    "Explain the concept of a Mixture of Experts (MoE) architecture in simple terms.",
    "What are the advantages of using a Mixture of Experts model compared to a dense model?",
    "Solve the following math problem step by step: If a train travels at 60 mph for 3 hours and then at 80 mph for 2 hours, what is the average speed for the entire journey?"
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}\n")
    
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=min(input_ids.shape[1] + 200, model_config.max_sequence_length),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Print generated text
    print(f"Generated text:\n{generated_text}\n")
    print("-" * 80)

# Save all metrics to a file
os.makedirs("benchmark_results", exist_ok=True)
with open("benchmark_results/all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# Visualize results
print("\nVisualizing results...")
benchmark_metrics = {
    "MMLU": all_metrics.get("mmlu_accuracy", float('nan')),
    "GSM8K": all_metrics.get("gsm8k_accuracy", float('nan')),
    "Perplexity": 1.0 / all_metrics.get("perplexity", float('nan'))  # Invert perplexity for visualization
}

# Filter out NaN values
benchmark_metrics = {k: v for k, v in benchmark_metrics.items() if not np.isnan(v)}

if benchmark_metrics:
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(benchmark_metrics.keys(), benchmark_metrics.values())
    plt.ylim(0, 1)
    plt.title("Benchmark Results")
    plt.ylabel("Score")
    
    # Add value labels
    for i, (key, value) in enumerate(benchmark_metrics.items()):
        plt.text(i, value + 0.02, f"{value:.4f}", ha="center")
    
    plt.tight_layout()
    plt.savefig("benchmark_results/benchmark_results.png")
    plt.show()
else:
    print("No valid benchmark metrics to visualize.")

# Compare with DeepSeek-V3
print("\nComparing with DeepSeek-V3...")
deepseek_results = {
    "MMLU": 0.80,
    "GSM8K": 0.85,
    "Perplexity": 0.90  # Normalized for visualization
}

# Filter out benchmarks we didn't evaluate
common_benchmarks = set(benchmark_metrics.keys()) & set(deepseek_results.keys())
filtered_our_results = {k: benchmark_metrics[k] for k in common_benchmarks}
filtered_deepseek_results = {k: deepseek_results[k] for k in common_benchmarks}

if filtered_our_results:
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(common_benchmarks))
    width = 0.35
    
    plt.bar(x - width/2, [filtered_our_results[k] for k in common_benchmarks], width, label="Our Model")
    plt.bar(x + width/2, [filtered_deepseek_results[k] for k in common_benchmarks], width, label="DeepSeek-V3")
    
    plt.xlabel("Benchmark")
    plt.ylabel("Score")
    plt.title("Comparison with DeepSeek-V3")
    plt.xticks(x, common_benchmarks)
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels
    for i, benchmark in enumerate(common_benchmarks):
        plt.text(i - width/2, filtered_our_results[benchmark] + 0.02, f"{filtered_our_results[benchmark]:.4f}", ha="center")
        plt.text(i + width/2, filtered_deepseek_results[benchmark] + 0.02, f"{filtered_deepseek_results[benchmark]:.4f}", ha="center")
    
    plt.tight_layout()
    plt.savefig("benchmark_results/comparison_with_deepseek.png")
    plt.show()
    
    # Calculate and print performance gap
    print("Performance Gap Analysis:")
    for benchmark in common_benchmarks:
        gap = filtered_our_results[benchmark] - filtered_deepseek_results[benchmark]
        print(f"  {benchmark}: {gap:.4f} ({'+' if gap >= 0 else ''}{gap * 100:.2f}%)")
else:
    print("No common benchmarks to compare.")

print("\nEvaluation complete! Results have been saved to the benchmark_results directory.")
