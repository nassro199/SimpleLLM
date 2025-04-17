"""
Memory optimization utilities for training large models in resource-constrained environments.
"""

import gc
import torch
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get memory statistics for the specified device.
    
    Args:
        device: Device to get memory statistics for
        
    Returns:
        Dictionary of memory statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats = {}
    
    if device.type == "cuda":
        # Get CUDA memory statistics
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
        max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)  # GB
        
        stats = {
            "cuda_memory_allocated_gb": allocated,
            "cuda_memory_reserved_gb": reserved,
            "cuda_max_memory_allocated_gb": max_allocated,
            "cuda_max_memory_reserved_gb": max_reserved
        }
    
    # Get CPU memory statistics
    try:
        import psutil
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024 ** 3)  # GB
        stats["cpu_memory_gb"] = cpu_memory
    except ImportError:
        logger.warning("psutil not installed. CPU memory statistics not available.")
    
    return stats


def print_memory_stats(device: Optional[torch.device] = None) -> None:
    """
    Print memory statistics for the specified device.
    
    Args:
        device: Device to print memory statistics for
    """
    stats = get_memory_stats(device)
    
    logger.info("Memory Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.2f} GB")


def clear_memory(device: Optional[torch.device] = None) -> None:
    """
    Clear memory for the specified device.
    
    Args:
        device: Device to clear memory for
    """
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache if using CUDA
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def optimize_memory_efficiency(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply memory efficiency optimizations to the model.
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model
    """
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "gradient_checkpointing_enable"):
        model.transformer.gradient_checkpointing_enable()
    
    return model


def get_model_size(model: torch.nn.Module) -> Dict[str, Union[int, float]]:
    """
    Get the size of the model in parameters and memory.
    
    Args:
        model: Model to get size for
        
    Returns:
        Dictionary with model size information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate activation memory (rough estimate)
    batch_size = 1
    seq_length = 1024
    hidden_size = 0
    
    for module in model.modules():
        if hasattr(module, "hidden_size"):
            hidden_size = max(hidden_size, module.hidden_size)
    
    if hidden_size > 0:
        # Rough estimate: 4 bytes per float * batch_size * seq_length * hidden_size * num_layers * 4 (for activations)
        activation_size = 4 * batch_size * seq_length * hidden_size * len(list(model.modules())) * 4
    else:
        activation_size = 0
    
    # Total memory
    total_size = param_size + buffer_size + activation_size
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_mb": param_size / (1024 ** 2),
        "buffer_size_mb": buffer_size / (1024 ** 2),
        "estimated_activation_size_mb": activation_size / (1024 ** 2),
        "estimated_total_size_mb": total_size / (1024 ** 2)
    }


def apply_mixed_precision(model: torch.nn.Module, precision: str = "bf16") -> torch.nn.Module:
    """
    Apply mixed precision to the model.
    
    Args:
        model: Model to apply mixed precision to
        precision: Precision to use (fp16, bf16)
        
    Returns:
        Model with mixed precision
    """
    if precision == "fp16":
        # Convert model to fp16
        model = model.half()
    elif precision == "bf16":
        # Convert model to bf16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model = model.bfloat16()
        else:
            logger.warning("BF16 not supported on this device. Using FP32 instead.")
    
    return model


def apply_quantization(model: torch.nn.Module, quantization: str = "int8") -> torch.nn.Module:
    """
    Apply quantization to the model.
    
    Args:
        model: Model to apply quantization to
        quantization: Quantization to use (int8, int4)
        
    Returns:
        Quantized model
    """
    try:
        import bitsandbytes as bnb
        
        if quantization == "int8":
            # Convert linear layers to 8-bit
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None
                    )
                    # Copy weights and bias
                    model._modules[name].weight.data = module.weight.data
                    if module.bias is not None:
                        model._modules[name].bias.data = module.bias.data
        elif quantization == "int4":
            # Convert linear layers to 4-bit
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    model._modules[name] = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        module.bias is not None
                    )
                    # Copy weights and bias
                    model._modules[name].weight.data = module.weight.data
                    if module.bias is not None:
                        model._modules[name].bias.data = module.bias.data
    except ImportError:
        logger.warning("bitsandbytes not installed. Quantization not applied.")
    
    return model
