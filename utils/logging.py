"""
Logging utilities for MoE LLM.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Optional, Union, Any

# Configure logging
def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Configure logging for the MoE LLM.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        log_format: Logging format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file is not None:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """
    Logger for TensorBoard.
    """
    def __init__(self, log_dir: str = "logs/tensorboard"):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.initialized = True
        except ImportError:
            logging.warning("TensorBoard not installed. TensorBoard logging disabled.")
            self.initialized = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Tag for the scalar
            value: Scalar value
            step: Global step
        """
        if self.initialized:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """
        Log multiple scalar values to TensorBoard.
        
        Args:
            main_tag: Main tag for the scalars
            tag_scalar_dict: Dictionary of tag-scalar pairs
            step: Global step
        """
        if self.initialized:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """
        Log a histogram to TensorBoard.
        
        Args:
            tag: Tag for the histogram
            values: Values for the histogram
            step: Global step
        """
        if self.initialized:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text_string: str, step: int) -> None:
        """
        Log text to TensorBoard.
        
        Args:
            tag: Tag for the text
            text_string: Text to log
            step: Global step
        """
        if self.initialized:
            self.writer.add_text(tag, text_string, step)
    
    def close(self) -> None:
        """
        Close the TensorBoard writer.
        """
        if self.initialized:
            self.writer.close()


class WandBLogger:
    """
    Logger for Weights & Biases.
    """
    def __init__(
        self,
        project: str = "moe-llm",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Weights & Biases logger.
        
        Args:
            project: Project name
            name: Run name
            config: Configuration dictionary
        """
        self.project = project
        self.name = name or f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.config = config or {}
        
        # Initialize Weights & Biases
        try:
            import wandb
            
            # Check if wandb is already initialized
            if wandb.run is None:
                wandb.init(project=project, name=name, config=config)
            
            self.initialized = True
        except ImportError:
            logging.warning("Weights & Biases not installed. WandB logging disabled.")
            self.initialized = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to Weights & Biases.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step
        """
        if self.initialized:
            import wandb
            wandb.log(metrics, step=step)
    
    def log_artifact(self, artifact_path: str, name: str, type: str) -> None:
        """
        Log an artifact to Weights & Biases.
        
        Args:
            artifact_path: Path to the artifact
            name: Name of the artifact
            type: Type of the artifact
        """
        if self.initialized:
            import wandb
            
            # Create artifact
            artifact = wandb.Artifact(name=name, type=type)
            
            # Add file to artifact
            artifact.add_file(artifact_path)
            
            # Log artifact
            wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """
        Finish the Weights & Biases run.
        """
        if self.initialized:
            import wandb
            wandb.finish()


class JSONLogger:
    """
    Logger for JSON files.
    """
    def __init__(self, log_dir: str = "logs/json"):
        """
        Initialize JSON logger.
        
        Args:
            log_dir: Directory to save JSON logs
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, f"log-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
        
        # Initialize logs
        self.logs = []
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Global step
        """
        # Add step to metrics if provided
        if step is not None:
            metrics["step"] = step
        
        # Add timestamp to metrics
        metrics["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add metrics to logs
        self.logs.append(metrics)
        
        # Write logs to file
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)
    
    def close(self) -> None:
        """
        Close the JSON logger.
        """
        # Write logs to file
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)
