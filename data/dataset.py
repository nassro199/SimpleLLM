"""
Dataset loading and preprocessing for MoE LLM training.
"""

import os
import json
import logging
import torch
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Dataset for language modeling with pre-tokenized data.
    """
    def __init__(
        self,
        tokenized_data: List[Dict],
        max_seq_length: int = 4096,
        input_key: str = "input_ids",
        label_key: str = "labels"
    ):
        self.tokenized_data = tokenized_data
        self.max_seq_length = max_seq_length
        self.input_key = input_key
        self.label_key = label_key
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        
        # Ensure input_ids and labels are within max_seq_length
        input_ids = item[self.input_key][:self.max_seq_length]
        
        # Create labels (shifted input_ids for causal language modeling)
        if self.label_key in item:
            labels = item[self.label_key][:self.max_seq_length]
        else:
            labels = input_ids.copy()
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class IterableTextDataset(IterableDataset):
    """
    Iterable dataset for language modeling with streaming data.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_path: str,
        max_seq_length: int = 4096,
        streaming: bool = True,
        text_column: str = "text",
        buffer_size: int = 1000,
        shuffle: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        
        # Load dataset
        self.dataset = load_dataset(
            dataset_path,
            streaming=streaming,
            split="train"
        )
    
    def __iter__(self):
        buffer = []
        
        for example in self.dataset:
            # Tokenize text
            tokenized = self.tokenizer(
                example[self.text_column],
                truncation=False,
                padding=False,
                return_tensors=None
            )
            
            # Process into chunks of max_seq_length
            input_ids = tokenized["input_ids"]
            
            for i in range(0, len(input_ids), self.max_seq_length):
                chunk = input_ids[i:i + self.max_seq_length]
                
                # Skip chunks that are too small
                if len(chunk) < 32:  # Minimum sequence length
                    continue
                
                # Create attention mask
                attention_mask = [1] * len(chunk)
                
                # Create labels (shifted input_ids for causal language modeling)
                labels = chunk.copy()
                
                item = {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                }
                
                buffer.append(item)
                
                # When buffer is full, shuffle and yield items
                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        import random
                        random.shuffle(buffer)
                    
                    for buffered_item in buffer:
                        yield buffered_item
                    
                    buffer = []
        
        # Yield remaining items in buffer
        if buffer:
            if self.shuffle:
                import random
                random.shuffle(buffer)
            
            for buffered_item in buffer:
                yield buffered_item


def load_and_prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    dataset_paths: List[str],
    max_seq_length: int = 4096,
    streaming: bool = False,
    text_column: str = "text",
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = None,
    preprocessing_num_workers: int = 4,
    use_auth_token: bool = False
) -> Dict[str, Union[Dataset, IterableDataset]]:
    """
    Load and prepare datasets for training, validation, and testing.
    
    Args:
        tokenizer: Tokenizer to use for tokenization
        dataset_paths: List of dataset paths to load
        max_seq_length: Maximum sequence length
        streaming: Whether to use streaming datasets
        text_column: Column name for text data
        train_split: Split to use for training
        validation_split: Split to use for validation
        test_split: Split to use for testing
        preprocessing_num_workers: Number of workers for preprocessing
        use_auth_token: Whether to use authentication token for private datasets
        
    Returns:
        Dictionary containing datasets for training, validation, and testing
    """
    datasets = {}
    
    if streaming:
        # Use iterable datasets for streaming
        for split in [train_split, validation_split, test_split]:
            if split is None:
                continue
            
            datasets[split] = IterableTextDataset(
                tokenizer=tokenizer,
                dataset_path=dataset_paths[0],  # Use first dataset for streaming
                max_seq_length=max_seq_length,
                streaming=True,
                text_column=text_column,
                buffer_size=1000,
                shuffle=(split == train_split)  # Only shuffle training data
            )
    else:
        # Load datasets
        loaded_datasets = []
        for dataset_path in dataset_paths:
            try:
                # Try to load as a local file
                if os.path.isfile(dataset_path):
                    with open(dataset_path, "r") as f:
                        data = json.load(f)
                    
                    # Convert to HuggingFace dataset
                    loaded_dataset = HFDataset.from_dict({"text": data})
                else:
                    # Load from HuggingFace Hub
                    loaded_dataset = load_dataset(
                        dataset_path,
                        use_auth_token=use_auth_token
                    )
                
                loaded_datasets.append(loaded_dataset)
                logger.info(f"Loaded dataset: {dataset_path}")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_path}: {e}")
                continue
        
        if not loaded_datasets:
            raise ValueError("No datasets were successfully loaded.")
        
        # Concatenate datasets if multiple were loaded
        if len(loaded_datasets) > 1:
            # Check if all datasets have the same structure
            all_splits = set()
            for ds in loaded_datasets:
                all_splits.update(ds.keys())
            
            # Concatenate datasets for each split
            concatenated_datasets = {}
            for split in all_splits:
                split_datasets = [ds[split] for ds in loaded_datasets if split in ds]
                if split_datasets:
                    concatenated_datasets[split] = concatenate_datasets(split_datasets)
            
            dataset_dict = concatenated_datasets
        else:
            dataset_dict = loaded_datasets[0]
        
        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                truncation=False,
                padding=False,
                return_tensors=None
            )
        
        tokenized_datasets = {}
        for split, dataset in dataset_dict.items():
            if split not in [train_split, validation_split, test_split] or dataset is None:
                continue
            
            # Tokenize dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=[col for col in dataset.column_names if col != text_column],
                desc=f"Tokenizing {split} dataset"
            )
            
            # Group texts into chunks of max_seq_length
            def group_texts(examples):
                # Concatenate all texts
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                
                # Compute length of concatenated texts
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                
                # We drop the last chunk if it's smaller than max_seq_length
                total_length = (total_length // max_seq_length) * max_seq_length
                
                # Split by chunks of max_seq_length
                result = {
                    k: [t[i:i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                
                # Create labels
                result["labels"] = result["input_ids"].copy()
                
                return result
            
            # Group texts
            grouped_dataset = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=preprocessing_num_workers,
                desc=f"Grouping {split} texts"
            )
            
            # Convert to PyTorch dataset
            datasets[split] = TextDataset(
                tokenized_data=grouped_dataset,
                max_seq_length=max_seq_length
            )
    
    return datasets


def create_dataloaders(
    datasets: Dict[str, Union[Dataset, IterableDataset]],
    batch_size: int,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        datasets: Dictionary containing datasets for training, validation, and testing
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        
    Returns:
        Dictionary containing dataloaders for training, validation, and testing
    """
    dataloaders = {}
    
    for split, dataset in datasets.items():
        is_iterable = isinstance(dataset, IterableDataset)
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(not is_iterable and split == "train"),  # Only shuffle non-iterable training data
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train")  # Only drop last batch for training
        )
    
    return dataloaders
