"""
Main model class for the MoE LLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .transformer import TransformerModel
from .config import MoEConfig


class MTPHead(nn.Module):
    """
    Multi-Token Prediction (MTP) head for predicting multiple tokens at once.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_tokens = config.mtp_num_tokens
        
        # Projection layers for predicting multiple tokens
        self.mtp_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.mtp_num_tokens)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for the MTP head.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            logits: List of [batch_size, seq_len, vocab_size] for each token prediction
        """
        logits = []
        for projection in self.mtp_projections:
            token_logits = projection(hidden_states)
            logits.append(token_logits)
        
        return logits


class MoELLM(nn.Module):
    """
    Mixture of Experts Language Model.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Transformer model
        self.transformer = TransformerModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi-Token Prediction head
        if config.use_mtp:
            self.mtp_head = MTPHead(config)
        else:
            self.mtp_head = None
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_mtp: bool = False
    ) -> Dict:
        """
        Forward pass for the MoE LLM.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            position_ids: Optional position indices [batch_size, seq_len]
            past_key_values: Optional cached key and value states
            labels: Optional labels for language modeling [batch_size, seq_len]
            use_cache: Whether to use cached key and value states
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary or tuple
            use_mtp: Whether to use Multi-Token Prediction
            
        Returns:
            Dictionary containing:
                loss: Optional language modeling loss
                logits: [batch_size, seq_len, vocab_size]
                mtp_logits: Optional list of MTP logits
                past_key_values: Optional cached key and value states
                hidden_states: Optional all hidden states
                attentions: Optional all attention weights
                aux_losses: Optional auxiliary losses from MoE
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Get language modeling logits
        logits = self.lm_head(hidden_states)
        
        # Get MTP logits if requested
        mtp_logits = None
        if use_mtp and self.mtp_head is not None:
            mtp_logits = self.mtp_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Add auxiliary losses if available
            if "aux_losses" in transformer_outputs:
                for key, aux_loss in transformer_outputs["aux_losses"].items():
                    if key == "total":
                        loss += aux_loss
        
        # Prepare outputs
        outputs = {
            "logits": logits,
        }
        
        if loss is not None:
            outputs["loss"] = loss
        
        if mtp_logits is not None:
            outputs["mtp_logits"] = mtp_logits
        
        if use_cache:
            outputs["past_key_values"] = transformer_outputs.get("past_key_values")
        
        if output_hidden_states:
            outputs["hidden_states"] = transformer_outputs.get("hidden_states")
        
        if output_attentions:
            outputs["attentions"] = transformer_outputs.get("attentions")
        
        if "aux_losses" in transformer_outputs:
            outputs["aux_losses"] = transformer_outputs["aux_losses"]
        
        if not return_dict:
            return tuple(outputs.values())
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        min_length: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        use_cache: bool = True,
        use_mtp: bool = False
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            do_sample: Whether to sample from the distribution
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to return
            use_cache: Whether to use cached key and value states
            use_mtp: Whether to use Multi-Token Prediction
            
        Returns:
            generated_ids: [batch_size * num_return_sequences, max_length]
        """
        batch_size = input_ids.shape[0]
        
        # Expand input for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        # Initialize past key values
        past_key_values = None
        
        # Initialize generated sequences with input_ids
        generated_ids = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generate tokens up to max_length
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                use_mtp=use_mtp
            )
            
            # Get next token logits
            if use_mtp and "mtp_logits" in outputs and outputs["mtp_logits"] is not None:
                # Use the first token prediction from MTP
                next_token_logits = outputs["mtp_logits"][0][:, -1, :]
            else:
                next_token_logits = outputs["logits"][:, -1, :]
            
            # Update past key values
            past_key_values = outputs.get("past_key_values")
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_return_sequences):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply min_length constraint
            if generated_ids.shape[1] < min_length:
                # Set probability of EOS token to 0
                next_token_logits[:, self.config.eos_token_id] = -float("inf")
            
            # Sample next token
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, -float("inf"))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_values)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
                
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append next tokens to generated_ids
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)],
                dim=-1
            )
            
            # Check if all sequences have reached EOS
            if (generated_ids[:, -1] == self.config.eos_token_id).all():
                break
        
        return generated_ids
