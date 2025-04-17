"""
Tokenizer implementation for MoE LLM.
"""

import os
import json
from typing import Dict, List, Optional, Union, Any
from transformers import PreTrainedTokenizer, AutoTokenizer


def get_tokenizer(
    tokenizer_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_fast: bool = True,
    revision: str = "main",
    use_auth_token: bool = False,
    **kwargs
) -> PreTrainedTokenizer:
    """
    Load a tokenizer from HuggingFace Hub or local path.
    
    Args:
        tokenizer_name_or_path: Name or path of the tokenizer
        cache_dir: Cache directory for the tokenizer
        use_fast: Whether to use the fast tokenizer implementation
        revision: Revision of the tokenizer
        use_auth_token: Whether to use authentication token for private tokenizers
        **kwargs: Additional arguments to pass to the tokenizer
        
    Returns:
        Loaded tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=cache_dir,
        use_fast=use_fast,
        revision=revision,
        use_auth_token=use_auth_token,
        **kwargs
    )
    
    # Ensure special tokens are set
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>"
    }
    
    # Add special tokens if they don't exist
    special_tokens_dict = {}
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, token_name) is None:
            special_tokens_dict[token_name] = token_value
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    
    return tokenizer


def save_tokenizer(
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    legacy_format: bool = False
) -> None:
    """
    Save a tokenizer to a directory.
    
    Args:
        tokenizer: Tokenizer to save
        output_dir: Directory to save the tokenizer to
        legacy_format: Whether to save in legacy format
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(
        output_dir,
        legacy_format=legacy_format
    )
    
    # Save additional tokenizer info
    tokenizer_info = {
        "name": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "special_tokens": {
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token,
            "unk_token": tokenizer.unk_token
        }
    }
    
    with open(os.path.join(output_dir, "tokenizer_info.json"), "w") as f:
        json.dump(tokenizer_info, f, indent=2)


def train_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: List[str] = ["<pad>", "</s>", "<s>", "<unk>", "<mask>"],
    output_dir: Optional[str] = None
) -> PreTrainedTokenizer:
    """
    Train a new tokenizer on the provided texts.
    
    Args:
        texts: List of texts to train the tokenizer on
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency of a token to be included in the vocabulary
        special_tokens: List of special tokens to add to the tokenizer
        output_dir: Directory to save the tokenizer to
        
    Returns:
        Trained tokenizer
    """
    # Import tokenizers library
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    except ImportError:
        raise ImportError(
            "You need to install the `tokenizers` library to train a tokenizer. "
            "Run `pip install tokenizers`."
        )
    
    # Create a new tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Set pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )
    
    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Set post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Convert to HuggingFace tokenizer
    from transformers import PreTrainedTokenizerFast
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    
    # Save tokenizer if output_dir is provided
    if output_dir is not None:
        save_tokenizer(hf_tokenizer, output_dir)
    
    return hf_tokenizer
