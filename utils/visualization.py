"""
Visualization utilities for MoE LLM.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple


def plot_training_loss(
    log_file: str,
    output_dir: str = "plots",
    output_name: str = "training_loss.png",
    window_size: int = 100
) -> str:
    """
    Plot training loss from a log file.
    
    Args:
        log_file: Path to the log file
        output_dir: Directory to save the plot
        output_name: Name of the output file
        window_size: Window size for moving average
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load log file
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    # Extract steps and losses
    steps = []
    losses = []
    
    for log in logs:
        if "step" in log and "loss" in log:
            steps.append(log["step"])
            losses.append(log["loss"])
    
    # Calculate moving average
    if window_size > 0:
        moving_avg = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(losses[start_idx:i+1]))
    else:
        moving_avg = losses
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, alpha=0.3, label="Loss")
    plt.plot(steps, moving_avg, label=f"Moving Average (window={window_size})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_learning_rate(
    log_file: str,
    output_dir: str = "plots",
    output_name: str = "learning_rate.png"
) -> str:
    """
    Plot learning rate from a log file.
    
    Args:
        log_file: Path to the log file
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load log file
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    # Extract steps and learning rates
    steps = []
    learning_rates = []
    
    for log in logs:
        if "step" in log and "learning_rate" in log:
            steps.append(log["step"])
            learning_rates.append(log["learning_rate"])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_evaluation_metrics(
    log_file: str,
    metrics: List[str],
    output_dir: str = "plots",
    output_name: str = "evaluation_metrics.png"
) -> str:
    """
    Plot evaluation metrics from a log file.
    
    Args:
        log_file: Path to the log file
        metrics: List of metrics to plot
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load log file
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    # Extract steps and metrics
    steps = []
    metric_values = {metric: [] for metric in metrics}
    
    for log in logs:
        if "step" in log:
            steps.append(log["step"])
            
            for metric in metrics:
                if metric in log:
                    metric_values[metric].append(log[metric])
                else:
                    metric_values[metric].append(None)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for metric in metrics:
        # Filter out None values
        valid_steps = [step for step, value in zip(steps, metric_values[metric]) if value is not None]
        valid_values = [value for value in metric_values[metric] if value is not None]
        
        if valid_steps and valid_values:
            plt.plot(valid_steps, valid_values, label=metric)
    
    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_benchmark_results(
    results_file: str,
    output_dir: str = "plots",
    output_name: str = "benchmark_results.png"
) -> str:
    """
    Plot benchmark results from a results file.
    
    Args:
        results_file: Path to the results file
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results file
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Extract benchmark names and scores
    benchmarks = []
    scores = []
    
    for key, value in results.items():
        if key.endswith("_accuracy") and not key.startswith("mmlu_") and not key.startswith("math_") and not key.startswith("bbh_"):
            benchmarks.append(key.replace("_accuracy", "").upper())
            scores.append(value)
    
    # Sort benchmarks and scores by score
    benchmarks, scores = zip(*sorted(zip(benchmarks, scores), key=lambda x: x[1]))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.barh(benchmarks, scores)
    plt.xlabel("Accuracy")
    plt.ylabel("Benchmark")
    plt.title("Benchmark Results")
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add score labels
    for i, score in enumerate(scores):
        plt.text(score + 0.01, i, f"{score:.4f}", va="center")
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_router_heatmap(
    router_weights: np.ndarray,
    output_dir: str = "plots",
    output_name: str = "router_heatmap.png"
) -> str:
    """
    Plot a heatmap of router weights.
    
    Args:
        router_weights: Router weights [num_tokens, num_experts]
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.imshow(router_weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Weight")
    plt.xlabel("Expert")
    plt.ylabel("Token")
    plt.title("Router Weights")
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_expert_usage(
    expert_usage: np.ndarray,
    output_dir: str = "plots",
    output_name: str = "expert_usage.png"
) -> str:
    """
    Plot expert usage.
    
    Args:
        expert_usage: Expert usage [num_experts]
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(expert_usage)), expert_usage)
    plt.xlabel("Expert")
    plt.ylabel("Usage")
    plt.title("Expert Usage")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_memory_usage(
    log_file: str,
    output_dir: str = "plots",
    output_name: str = "memory_usage.png"
) -> str:
    """
    Plot memory usage from a log file.
    
    Args:
        log_file: Path to the log file
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load log file
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    # Extract steps and memory usage
    steps = []
    cuda_allocated = []
    cuda_reserved = []
    cpu_memory = []
    
    for log in logs:
        if "step" in log:
            steps.append(log["step"])
            
            if "cuda_memory_allocated_gb" in log:
                cuda_allocated.append(log["cuda_memory_allocated_gb"])
            else:
                cuda_allocated.append(None)
            
            if "cuda_memory_reserved_gb" in log:
                cuda_reserved.append(log["cuda_memory_reserved_gb"])
            else:
                cuda_reserved.append(None)
            
            if "cpu_memory_gb" in log:
                cpu_memory.append(log["cpu_memory_gb"])
            else:
                cpu_memory.append(None)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot CUDA allocated memory
    if any(x is not None for x in cuda_allocated):
        valid_steps = [step for step, value in zip(steps, cuda_allocated) if value is not None]
        valid_values = [value for value in cuda_allocated if value is not None]
        plt.plot(valid_steps, valid_values, label="CUDA Allocated")
    
    # Plot CUDA reserved memory
    if any(x is not None for x in cuda_reserved):
        valid_steps = [step for step, value in zip(steps, cuda_reserved) if value is not None]
        valid_values = [value for value in cuda_reserved if value is not None]
        plt.plot(valid_steps, valid_values, label="CUDA Reserved")
    
    # Plot CPU memory
    if any(x is not None for x in cpu_memory):
        valid_steps = [step for step, value in zip(steps, cpu_memory) if value is not None]
        valid_values = [value for value in cpu_memory if value is not None]
        plt.plot(valid_steps, valid_values, label="CPU Memory")
    
    plt.xlabel("Step")
    plt.ylabel("Memory (GB)")
    plt.title("Memory Usage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def plot_comparison_with_deepseek(
    our_results: Dict[str, float],
    deepseek_results: Dict[str, float],
    output_dir: str = "plots",
    output_name: str = "comparison_with_deepseek.png"
) -> str:
    """
    Plot comparison with DeepSeek-V3.
    
    Args:
        our_results: Our benchmark results
        deepseek_results: DeepSeek-V3 benchmark results
        output_dir: Directory to save the plot
        output_name: Name of the output file
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract common benchmarks
    common_benchmarks = []
    our_scores = []
    deepseek_scores = []
    
    for key in our_results:
        if key in deepseek_results:
            common_benchmarks.append(key.replace("_accuracy", "").upper())
            our_scores.append(our_results[key])
            deepseek_scores.append(deepseek_results[key])
    
    # Sort benchmarks and scores by our score
    common_benchmarks, our_scores, deepseek_scores = zip(*sorted(zip(common_benchmarks, our_scores, deepseek_scores), key=lambda x: x[1]))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(common_benchmarks))
    width = 0.35
    
    plt.barh(x - width/2, our_scores, width, label="Our Model")
    plt.barh(x + width/2, deepseek_scores, width, label="DeepSeek-V3")
    
    plt.xlabel("Accuracy")
    plt.ylabel("Benchmark")
    plt.title("Comparison with DeepSeek-V3")
    plt.yticks(x, common_benchmarks)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add score labels
    for i, score in enumerate(our_scores):
        plt.text(score + 0.01, i - width/2, f"{score:.4f}", va="center")
    
    for i, score in enumerate(deepseek_scores):
        plt.text(score + 0.01, i + width/2, f"{score:.4f}", va="center")
    
    # Save plot
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    
    return output_path
