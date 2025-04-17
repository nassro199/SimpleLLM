"""
Perplexity calculation for MoE LLM.
"""

import math
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from tqdm.auto import tqdm

from ..model.model import MoELLM

logger = logging.getLogger(__name__)


def calculate_perplexity(
    model: MoELLM,
    tokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    stride: int = 512,
    max_length: int = 1024
) -> Dict[str, float]:
    """
    Calculate perplexity for a list of texts.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        texts: List of texts to evaluate
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        stride: Stride for sliding window evaluation
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of perplexity metrics
    """
    logger.info(f"Calculating perplexity for {len(texts)} texts...")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Initialize metrics
    total_loss = 0.0
    total_tokens = 0
    
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
        batch_texts = texts[i:i + batch_size]
        
        # Process each text in the batch
        batch_loss = 0.0
        batch_tokens = 0
        
        for text in batch_texts:
            # Tokenize text
            encodings = tokenizer(text, return_tensors="pt")
            
            # Get input IDs
            input_ids = encodings["input_ids"].to(device)
            
            # Calculate perplexity using sliding window approach
            nlls = []
            
            # Process text in chunks with stride
            for i in range(0, input_ids.size(1), stride):
                # Get chunk
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, input_ids.size(1))
                target_len = end_loc - i  # May be different from stride on last iteration
                
                # Get input IDs and labels for this chunk
                input_ids_chunk = input_ids[:, begin_loc:end_loc].to(device)
                target_ids_chunk = input_ids[:, i:end_loc].to(device)
                
                # Create attention mask
                attention_mask = torch.ones_like(input_ids_chunk)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids_chunk,
                        attention_mask=attention_mask,
                        labels=target_ids_chunk
                    )
                
                # Get loss
                neg_log_likelihood = outputs["loss"] * target_len
                
                # Add to list of negative log-likelihoods
                nlls.append(neg_log_likelihood)
            
            # Calculate perplexity for this text
            text_loss = torch.stack(nlls).sum() / input_ids.size(1)
            batch_loss += text_loss.item()
            batch_tokens += input_ids.size(1)
        
        # Update total metrics
        total_loss += batch_loss
        total_tokens += batch_tokens
    
    # Calculate overall perplexity
    perplexity = math.exp(total_loss / total_tokens)
    
    logger.info(f"Perplexity: {perplexity:.4f}")
    
    return {
        "perplexity": perplexity,
        "loss": total_loss / total_tokens
    }


def calculate_perplexity_on_dataset(
    model: MoELLM,
    tokenizer,
    dataset,
    text_column: str = "text",
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    stride: int = 512,
    max_length: int = 1024,
    num_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        dataset: Dataset to evaluate
        text_column: Column name for text data
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        stride: Stride for sliding window evaluation
        max_length: Maximum sequence length
        num_samples: Number of samples to evaluate (None for all)
        
    Returns:
        Dictionary of perplexity metrics
    """
    logger.info(f"Calculating perplexity on dataset...")
    
    # Get texts from dataset
    if num_samples is not None and num_samples < len(dataset):
        # Sample random subset
        import random
        indices = random.sample(range(len(dataset)), num_samples)
        texts = [dataset[i][text_column] for i in indices]
    else:
        # Use all samples
        texts = [sample[text_column] for sample in dataset]
    
    # Calculate perplexity
    return calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        stride=stride,
        max_length=max_length
    )
