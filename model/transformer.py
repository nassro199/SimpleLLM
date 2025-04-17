"""
Transformer architecture with Mixture of Experts integration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .moe_layer import MoELayer


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size]
            
        Returns:
            normalized_states: [batch_size, sequence_length, hidden_size]
        """
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position embeddings cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """
        Precompute cos and sin values for positions.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        
        # Compute frequencies for each position
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Compute cos and sin values
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for the given positions.
        
        Args:
            x: [batch_size, seq_len, num_heads, head_dim]
            position_ids: Optional position indices
            
        Returns:
            cos: [1, seq_len, 1, dim]
            sin: [1, seq_len, 1, dim]
        """
        seq_len = x.shape[1]
        
        # If sequence length exceeds cached length, recompute cache
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        if position_ids is None:
            # Use default position ids (0, 1, 2, ...)
            return self.cos_cached[:, :seq_len, ...], self.sin_cached[:, :seq_len, ...]
        else:
            # Use provided position ids
            cos = F.embedding(position_ids, self.cos_cached[0])
            sin = F.embedding(position_ids, self.sin_cached[0])
            return cos.unsqueeze(2), sin.unsqueeze(2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: [batch_size, seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_heads, head_dim]
        cos: [1, seq_len, 1, head_dim]
        sin: [1, seq_len, 1, head_dim]
        
    Returns:
        q_rotated: [batch_size, seq_len, num_heads, head_dim]
        k_rotated: [batch_size, seq_len, num_heads, head_dim]
    """
    # Apply rotation using the rotation matrix
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) as used in DeepSeek-V3.
    
    This is an optimized attention mechanism that reduces computation and memory usage.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mla_dim: int = 128,
        dropout: float = 0.0,
        rotary_dim: int = 128,
        max_position_embeddings: int = 4096
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mla_dim = mla_dim
        self.head_dim = hidden_size // num_heads
        
        # Check if hidden size is divisible by number of heads
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * mla_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * mla_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Rotary embeddings for position encoding
        self.rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            max_position_embeddings=max_position_embeddings
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.mla_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Multi-head Latent Attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Optional position indices [batch_size, seq_len]
            past_key_value: Optional cached key and value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key and value states
            
        Returns:
            attn_output: [batch_size, seq_len, hidden_size]
            attn_weights: Optional attention weights
            past_key_value: Optional cached key and value states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project inputs to queries, keys, and values
        query_states = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * mla_dim]
        key_states = self.k_proj(hidden_states)  # [batch_size, seq_len, num_heads * mla_dim]
        value_states = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.mla_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.mla_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Get cached key and value states if using cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Apply rotary embeddings to queries and keys
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Transpose for batched matrix multiplication
        query_states = query_states.transpose(1, 2)  # [batch_size, num_heads, seq_len, mla_dim]
        key_states = key_states.transpose(1, 2)  # [batch_size, num_heads, kv_seq_len, mla_dim]
        value_states = value_states.transpose(1, 2)  # [batch_size, num_heads, kv_seq_len, head_dim]
        
        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Calculate attention output
        attn_output = torch.matmul(attn_weights, value_states)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Return attention output and optional values
        if output_attentions:
            return attn_output, attn_weights, past_key_value
        else:
            return attn_output, None, past_key_value


class TransformerBlock(nn.Module):
    """
    Transformer block with MoE integration.
    """
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Normalization layers
        if config.use_rms_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention layer
        if config.use_mla:
            self.attention = MultiHeadLatentAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                mla_dim=config.mla_dim,
                dropout=config.attention_dropout_prob,
                rotary_dim=config.rotary_dim,
                max_position_embeddings=config.max_position_embeddings
            )
        else:
            # Standard multi-head attention (not implemented here for brevity)
            raise NotImplementedError("Standard multi-head attention not implemented")
        
        # MoE or FFN layer
        if hasattr(config, "num_experts") and config.num_experts > 1:
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                expert_capacity=config.expert_capacity,
                router_jitter_noise=config.router_jitter_noise,
                router_z_loss_coef=config.router_z_loss_coef,
                router_aux_loss_coef=config.router_aux_loss_coef,
                activation_fn=nn.SiLU()
            )
        else:
            # Standard feed-forward network
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict]]:
        """
        Forward pass for the transformer block.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Optional position indices [batch_size, seq_len]
            past_key_value: Optional cached key and value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key and value states
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
            attn_weights: Optional attention weights
            past_key_value: Optional cached key and value states
            aux_loss: Optional auxiliary losses from MoE
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # MoE or FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Process through MoE or FFN
        aux_loss = None
        if isinstance(self.mlp, MoELayer):
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, past_key_value, aux_loss


class TransformerModel(nn.Module):
    """
    Transformer model with MoE integration.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        if config.use_rms_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False
    ) -> Dict:
        """
        Forward pass for the transformer model.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            position_ids: Optional position indices [batch_size, seq_len]
            past_key_values: Optional cached key and value states
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            use_cache: Whether to use cached key and value states
            
        Returns:
            Dictionary containing:
                last_hidden_state: [batch_size, seq_len, hidden_size]
                past_key_values: Optional cached key and value states
                hidden_states: Optional all hidden states
                attentions: Optional all attention weights
                aux_losses: Optional auxiliary losses from MoE
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        # Convert attention mask to causal mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_aux_losses = []
        new_past_key_values = [] if use_cache else None
        
        # Process through transformer blocks
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            
            hidden_states, attn_weights, past_key_value, aux_loss = layer_outputs
            
            if use_cache:
                new_past_key_values.append(past_key_value)
            
            if output_attentions and attn_weights is not None:
                all_attentions += (attn_weights,)
            
            if aux_loss is not None:
                all_aux_losses.append(aux_loss)
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Prepare outputs
        outputs = {
            "last_hidden_state": hidden_states,
        }
        
        if use_cache:
            outputs["past_key_values"] = new_past_key_values
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        
        if output_attentions:
            outputs["attentions"] = all_attentions
        
        if all_aux_losses:
            # Combine auxiliary losses
            aux_losses = {}
            for key in all_aux_losses[0].keys():
                aux_losses[key] = sum(loss[key] for loss in all_aux_losses if key in loss)
            outputs["aux_losses"] = aux_losses
        
        return outputs
