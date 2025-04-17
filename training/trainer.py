"""
Trainer for MoE LLM with memory optimization techniques.
"""

import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm.auto import tqdm

from ..model.model import MoELLM
from ..model.config import TrainingConfig
from .memory_utils import get_memory_stats, print_memory_stats

logger = logging.getLogger(__name__)


class MoETrainer:
    """
    Trainer for MoE LLM with memory optimization techniques.
    """
    def __init__(
        self,
        model: MoELLM,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LambdaLR] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        use_wandb: bool = False
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up gradient checkpointing if enabled
        if self.config.use_gradient_checkpointing:
            self.model.transformer.gradient_checkpointing_enable()
        
        # Set up mixed precision training if enabled
        self.scaler = None
        if self.config.mixed_precision != "no":
            if self.config.mixed_precision == "fp16":
                self.scaler = torch.cuda.amp.GradScaler()
            elif self.config.mixed_precision == "bf16":
                # BF16 doesn't need a scaler
                pass
            else:
                raise ValueError(f"Unsupported mixed precision mode: {self.config.mixed_precision}")
        
        # Set up wandb if enabled
        if self.use_wandb:
            try:
                import wandb
                if not wandb.api.api_key:
                    logger.warning("Wandb API key not found. Disabling wandb logging.")
                    self.use_wandb = False
                else:
                    wandb.init(
                        project=self.config.wandb_project,
                        name=self.config.wandb_run_name,
                        config=vars(self.config)
                    )
            except ImportError:
                logger.warning("Wandb not installed. Disabling wandb logging.")
                self.use_wandb = False
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.no_improvement_count = 0
    
    def train(self):
        """
        Train the model.
        """
        logger.info("Starting training...")
        
        # Print initial memory stats
        print_memory_stats(self.device)
        
        # Training loop
        self.model.train()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)
        
        # Calculate total steps
        if hasattr(self.train_dataloader, "__len__"):
            total_steps = len(self.train_dataloader) * self.config.max_steps
        else:
            total_steps = self.config.max_steps
        
        # Progress bar
        progress_bar = tqdm(total=total_steps, desc="Training")
        progress_bar.update(self.global_step)
        
        # Accumulate gradients over multiple steps
        accumulated_loss = 0
        
        # Training loop
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.config.mixed_precision != "no":
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] / self.config.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate loss
                accumulated_loss += loss.item()
                
                # Update weights if gradient accumulation is complete
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or batch_idx == len(self.train_dataloader) - 1:
                    # Optimizer step with mixed precision
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    # Update learning rate
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Log metrics
                    if self.global_step % self.config.logging_steps == 0:
                        lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]["lr"]
                        
                        metrics = {
                            "loss": accumulated_loss * self.config.gradient_accumulation_steps,
                            "learning_rate": lr,
                            "epoch": self.epoch,
                            "step": self.global_step
                        }
                        
                        # Add memory stats
                        memory_stats = get_memory_stats(self.device)
                        metrics.update(memory_stats)
                        
                        # Log to console
                        logger.info(f"Step {self.global_step}: loss = {metrics['loss']:.4f}, lr = {lr:.8f}")
                        
                        # Log to wandb
                        if self.use_wandb:
                            import wandb
                            wandb.log(metrics, step=self.global_step)
                        
                        # Reset accumulated loss
                        accumulated_loss = 0
                    
                    # Evaluate model
                    if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        # Log evaluation metrics
                        logger.info(f"Evaluation at step {self.global_step}: loss = {eval_metrics['eval_loss']:.4f}")
                        
                        # Log to wandb
                        if self.use_wandb:
                            import wandb
                            wandb.log(eval_metrics, step=self.global_step)
                        
                        # Check for early stopping
                        if eval_metrics["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self.no_improvement_count = 0
                            
                            # Save best model
                            self._save_checkpoint("best")
                        else:
                            self.no_improvement_count += 1
                            
                            if self.no_improvement_count >= self.config.early_stopping_patience:
                                logger.info(f"Early stopping triggered after {self.global_step} steps")
                                return
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(f"step-{self.global_step}")
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
                    
                    # Check if max steps reached
                    if self.global_step >= self.config.max_steps:
                        break
            
            # Check if max steps reached
            if self.global_step >= self.config.max_steps:
                break
        
        # Save final checkpoint
        self._save_checkpoint("final")
        
        # Close progress bar
        progress_bar.close()
        
        logger.info(f"Training completed after {self.global_step} steps")
    
    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader provided. Skipping evaluation.")
            return {}
        
        logger.info("Evaluating model...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0
        total_samples = 0
        
        # Evaluate model
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Update metrics
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate metrics
        eval_loss = total_loss / total_samples
        
        # Set model back to training mode
        self.model.train()
        
        return {
            "eval_loss": eval_loss,
            "perplexity": math.exp(eval_loss)
        }
    
    def _save_checkpoint(self, tag: str):
        """
        Save a checkpoint of the model and training state.
        
        Args:
            tag: Tag to identify the checkpoint
        """
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_dir = os.path.join("checkpoints", f"checkpoint-{tag}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "no_improvement_count": self.no_improvement_count,
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint of the model and training state.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        # Check if checkpoint exists
        if not os.path.isdir(checkpoint_path):
            logger.warning(f"Checkpoint directory {checkpoint_path} not found. Starting from scratch.")
            return
        
        # Load model
        self.model.load_from_pretrained(checkpoint_path)
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.isfile(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            self.best_eval_loss = training_state["best_eval_loss"]
            self.no_improvement_count = training_state["no_improvement_count"]
            
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            
            if self.lr_scheduler is not None and training_state["lr_scheduler_state"] is not None:
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler_state"])
            
            if self.scaler is not None and training_state["scaler_state"] is not None:
                self.scaler.load_state_dict(training_state["scaler_state"])
            
            logger.info(f"Resumed training from step {self.global_step}")
        else:
            logger.warning(f"Training state not found in {checkpoint_path}. Starting from scratch with loaded model.")


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    optimizer_type: str = "adamw",
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer to use
        betas: Beta parameters for Adam-based optimizers
        eps: Epsilon parameter for Adam-based optimizers
        
    Returns:
        Optimizer for the model
    """
    # Prepare optimizer parameters
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    if optimizer_type == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps
        )
    elif optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=weight_decay,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
        except ImportError:
            logger.warning("Adafactor not available. Falling back to AdamW.")
            from torch.optim import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps
            )
    elif optimizer_type == "8bit-adam":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps
            )
        except ImportError:
            logger.warning("8-bit Adam not available. Falling back to AdamW.")
            from torch.optim import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=betas,
                eps=eps
            )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def create_lr_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    lr_scheduler_type: str = "cosine"
) -> LambdaLR:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        lr_scheduler_type: Type of scheduler to use
        
    Returns:
        Learning rate scheduler
    """
    # Define scheduler function
    if lr_scheduler_type == "linear":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
            )
    elif lr_scheduler_type == "cosine":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    elif lr_scheduler_type == "constant":
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
    else:
        raise ValueError(f"Unsupported scheduler type: {lr_scheduler_type}")
    
    # Create scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return scheduler
