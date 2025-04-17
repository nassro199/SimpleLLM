"""
Checkpoint utilities for saving and loading model checkpoints.
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    output_dir: str = "checkpoints",
    checkpoint_name: str = "checkpoint",
    save_optimizer: bool = True,
    save_full_model: bool = True
) -> str:
    """
    Save a checkpoint of the model and training state.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        scaler: Gradient scaler to save
        epoch: Current epoch
        global_step: Current global step
        metrics: Metrics to save
        output_dir: Directory to save checkpoint to
        checkpoint_name: Name of the checkpoint
        save_optimizer: Whether to save optimizer state
        save_full_model: Whether to save the full model or just the state dict
        
    Returns:
        Path to the saved checkpoint
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model
    if save_full_model:
        # Save full model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
        else:
            torch.save(model, os.path.join(checkpoint_dir, "model.pt"))
    else:
        # Save model state dict
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_state_dict.pt"))
    
    # Save training state
    training_state = {
        "epoch": epoch,
        "global_step": global_step
    }
    
    # Save optimizer state
    if optimizer is not None and save_optimizer:
        training_state["optimizer_state"] = optimizer.state_dict()
    
    # Save scheduler state
    if scheduler is not None:
        training_state["scheduler_state"] = scheduler.state_dict()
    
    # Save scaler state
    if scaler is not None:
        training_state["scaler_state"] = scaler.state_dict()
    
    # Save metrics
    if metrics is not None:
        training_state["metrics"] = metrics
    
    # Save training state
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    return checkpoint_dir


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    checkpoint_path: str = "checkpoints/checkpoint",
    load_optimizer: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint of the model and training state.
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into
        scheduler: Learning rate scheduler to load checkpoint into
        scaler: Gradient scaler to load checkpoint into
        checkpoint_path: Path to the checkpoint directory
        load_optimizer: Whether to load optimizer state
        device: Device to load checkpoint to
        
    Returns:
        Dictionary of training state
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found")
        return {}
    
    # Load model
    if os.path.exists(os.path.join(checkpoint_path, "model.pt")):
        # Load full model
        model_state = torch.load(os.path.join(checkpoint_path, "model.pt"), map_location=device)
        model.load_state_dict(model_state)
    elif os.path.exists(os.path.join(checkpoint_path, "model_state_dict.pt")):
        # Load model state dict
        model_state = torch.load(os.path.join(checkpoint_path, "model_state_dict.pt"), map_location=device)
        model.load_state_dict(model_state)
    elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        # Load model from HuggingFace format
        model_state = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(model_state)
    else:
        logger.warning(f"No model checkpoint found in {checkpoint_path}")
    
    # Load training state
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device)
        
        # Load optimizer state
        if optimizer is not None and load_optimizer and "optimizer_state" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state"])
        
        # Load scheduler state
        if scheduler is not None and "scheduler_state" in training_state:
            scheduler.load_state_dict(training_state["scheduler_state"])
        
        # Load scaler state
        if scaler is not None and "scaler_state" in training_state:
            scaler.load_state_dict(training_state["scaler_state"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {training_state.get('epoch', 0)}, step {training_state.get('global_step', 0)}")
        
        return training_state
    else:
        logger.warning(f"No training state found in {checkpoint_path}")
        return {}


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """
    Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} not found")
        return None
    
    # Get all checkpoint directories
    checkpoint_dirs = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint")
    ]
    
    if not checkpoint_dirs:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    # Find the latest checkpoint
    latest_checkpoint = None
    latest_step = -1
    
    for checkpoint in checkpoint_dirs:
        # Check if training state exists
        training_state_path = os.path.join(checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            # Load training state
            training_state = torch.load(training_state_path, map_location="cpu")
            
            # Get global step
            global_step = training_state.get("global_step", 0)
            
            # Update latest checkpoint if this one is newer
            if global_step > latest_step:
                latest_step = global_step
                latest_checkpoint = checkpoint
    
    if latest_checkpoint is None:
        logger.warning(f"No valid checkpoints found in {checkpoint_dir}")
        return None
    
    logger.info(f"Found latest checkpoint {latest_checkpoint} at step {latest_step}")
    
    return latest_checkpoint


def save_model_for_inference(
    model: torch.nn.Module,
    tokenizer,
    output_dir: str = "model",
    save_format: str = "pytorch",
    quantization: Optional[str] = None
) -> str:
    """
    Save a model for inference.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save model to
        save_format: Format to save model in (pytorch, safetensors)
        quantization: Quantization to apply (None, int8, int4)
        
    Returns:
        Path to the saved model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply quantization if specified
    if quantization is not None:
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
    
    # Save model
    if hasattr(model, "save_pretrained"):
        # Save using HuggingFace's save_pretrained
        model.save_pretrained(output_dir, safe_serialization=(save_format == "safetensors"))
    else:
        # Save using PyTorch's save
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                
                # Save model state dict
                state_dict = model.state_dict()
                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            except ImportError:
                logger.warning("safetensors not installed. Falling back to PyTorch format.")
                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        else:
            # Save in PyTorch format
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    
    # Save model configuration
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)
    
    # Save model info
    model_info = {
        "model_type": model.__class__.__name__,
        "quantization": quantization,
        "save_format": save_format
    }
    
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Saved model for inference to {output_dir}")
    
    return output_dir
