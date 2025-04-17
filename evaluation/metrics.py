"""
Evaluation metrics for MoE LLM.
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any


def perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Calculate perplexity from logits and labels.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in labels
        
    Returns:
        Perplexity score
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Calculate perplexity
    return math.exp(loss.item())


def accuracy(predictions: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Calculate accuracy from predictions and labels.
    
    Args:
        predictions: Model predictions [batch_size, seq_len]
        labels: Target labels [batch_size, seq_len]
        ignore_index: Index to ignore in labels
        
    Returns:
        Accuracy score
    """
    # Flatten predictions and labels
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    
    # Create mask for valid labels
    mask = (labels != ignore_index)
    
    # Calculate accuracy
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match score from predictions and references.
    
    Args:
        predictions: Model predictions
        references: Target references
        
    Returns:
        Exact match score
    """
    # Calculate exact matches
    exact_matches = sum(pred == ref for pred, ref in zip(predictions, references))
    
    return exact_matches / len(predictions) if len(predictions) > 0 else 0.0


def f1_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate F1 score from predictions and references.
    
    Args:
        predictions: Model predictions
        references: Target references
        
    Returns:
        F1 score
    """
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Tokenize prediction and reference
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        
        # Calculate precision, recall, and F1
        common_tokens = pred_tokens.intersection(ref_tokens)
        
        precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0


def rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores from predictions and references.
    
    Args:
        predictions: Model predictions
        references: Target references
        
    Returns:
        Dictionary of ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("rouge_score not installed. Run `pip install rouge-score`.")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0
    }
    
    for pred, ref in zip(predictions, references):
        # Calculate ROUGE scores for this pair
        pair_scores = scorer.score(ref, pred)
        
        # Update scores
        for key in scores:
            scores[key] += pair_scores[key].fmeasure
    
    # Average scores
    for key in scores:
        scores[key] /= len(predictions) if len(predictions) > 0 else 1.0
    
    return scores


def bleu_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculate BLEU score from predictions and references.
    
    Args:
        predictions: Model predictions
        references: Target references (multiple references per prediction)
        
    Returns:
        BLEU score
    """
    try:
        from sacrebleu import corpus_bleu
    except ImportError:
        raise ImportError("sacrebleu not installed. Run `pip install sacrebleu`.")
    
    # Calculate BLEU score
    bleu = corpus_bleu(predictions, references)
    
    return bleu.score / 100.0  # Normalize to [0, 1]


def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Calculate multiple metrics from logits, labels, predictions, and references.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        predictions: Model predictions (decoded)
        references: Target references (decoded)
        ignore_index: Index to ignore in labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate perplexity
    metrics["perplexity"] = perplexity(logits, labels, ignore_index)
    
    # Calculate accuracy
    pred_ids = torch.argmax(logits, dim=-1)
    metrics["accuracy"] = accuracy(pred_ids, labels, ignore_index)
    
    # Calculate text-based metrics if predictions and references are provided
    if predictions is not None and references is not None:
        metrics["exact_match"] = exact_match(predictions, references)
        metrics["f1"] = f1_score(predictions, references)
        
        # Calculate ROUGE scores
        rouge_scores = rouge_score(predictions, references)
        metrics.update(rouge_scores)
        
        # Calculate BLEU score if multiple references are provided
        if isinstance(references[0], list):
            metrics["bleu"] = bleu_score(predictions, references)
    
    return metrics
