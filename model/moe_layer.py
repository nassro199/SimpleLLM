"""
Implementation of the Mixture of Experts layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class Router(nn.Module):
    """
    Router module that determines which experts to use for each token.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        jitter_noise: float = 0.0,
        expert_capacity: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.jitter_noise = jitter_noise
        self.expert_capacity = expert_capacity
        
        # Router projection
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize router weights
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_capacity: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Route the hidden states to the experts.
        
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size]
            expert_capacity: Optional override for expert capacity
            
        Returns:
            expert_indices: [num_experts, expert_capacity]
            expert_weights: [num_experts, expert_capacity]
            combine_weights: [batch_size, sequence_length, num_experts_per_token]
            router_logits: [batch_size, sequence_length, num_experts]
            aux_loss: Dictionary of auxiliary losses
        """
        batch_size, sequence_length, hidden_size = hidden_states.shape
        
        # If expert_capacity is not provided, use the one from initialization
        if expert_capacity is None:
            expert_capacity = self.expert_capacity
        
        # If expert_capacity is still None, calculate it based on sequence length
        if expert_capacity is None:
            # Default capacity factor is 1.25 times the expected number of tokens per expert
            tokens_per_expert = (batch_size * sequence_length) / self.num_experts
            expert_capacity = int(tokens_per_expert * 1.25)
        
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch_size, sequence_length, num_experts]
        
        # Add noise to encourage exploration during training
        if self.training and self.jitter_noise > 0:
            router_logits += torch.randn_like(router_logits) * self.jitter_noise
        
        # Calculate routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)  # [batch_size, sequence_length, num_experts]
        
        # Get top-k experts and their weights
        routing_weights_k, indices_k = torch.topk(
            routing_weights, 
            k=self.num_experts_per_token, 
            dim=-1
        )  # [batch_size, sequence_length, num_experts_per_token]
        
        # Normalize the routing weights
        routing_weights_k = routing_weights_k / routing_weights_k.sum(dim=-1, keepdim=True)
        
        # Compute load balancing auxiliary loss
        # This encourages uniform distribution of tokens across experts
        aux_loss = {}
        if self.training:
            # Calculate the fraction of tokens routed to each expert
            router_probs = routing_weights.mean(dim=[0, 1])  # [num_experts]
            
            # Calculate the auxiliary load balancing loss
            # We want each expert to receive an equal fraction of tokens (1/num_experts)
            aux_loss["load_balancing"] = self.num_experts * torch.sum(router_probs * router_probs)
            
            # Calculate the auxiliary z-loss
            # This encourages router logits to stay small
            aux_loss["z_loss"] = torch.mean(torch.square(router_logits))
        
        # Reshape indices and routing weights for scatter operation
        indices_k = indices_k.view(-1, self.num_experts_per_token)  # [batch_size * sequence_length, num_experts_per_token]
        routing_weights_k = routing_weights_k.view(-1, self.num_experts_per_token)  # [batch_size * sequence_length, num_experts_per_token]
        
        # Create a tensor of token indices
        token_indices = torch.arange(batch_size * sequence_length, device=hidden_states.device)
        token_indices = token_indices.unsqueeze(-1).expand(-1, self.num_experts_per_token)  # [batch_size * sequence_length, num_experts_per_token]
        
        # Flatten for dispatching
        flat_hidden_states = hidden_states.view(-1, hidden_size)  # [batch_size * sequence_length, hidden_size]
        
        # Create tensors for expert dispatch
        expert_indices = [[] for _ in range(self.num_experts)]
        expert_weights = [[] for _ in range(self.num_experts)]
        
        # Dispatch tokens to experts
        for i in range(self.num_experts_per_token):
            for j in range(batch_size * sequence_length):
                expert_idx = indices_k[j, i].item()
                if len(expert_indices[expert_idx]) < expert_capacity:
                    expert_indices[expert_idx].append(j)
                    expert_weights[expert_idx].append(routing_weights_k[j, i].item())
        
        # Pad expert indices and weights to expert_capacity
        for i in range(self.num_experts):
            padding_size = expert_capacity - len(expert_indices[i])
            if padding_size > 0:
                # Pad with zeros and set corresponding weights to zero
                expert_indices[i].extend([0] * padding_size)
                expert_weights[i].extend([0.0] * padding_size)
        
        # Convert to tensors
        expert_indices = torch.tensor(expert_indices, device=hidden_states.device)  # [num_experts, expert_capacity]
        expert_weights = torch.tensor(expert_weights, device=hidden_states.device)  # [num_experts, expert_capacity]
        
        return expert_indices, expert_weights, routing_weights_k, router_logits, aux_loss


class ExpertLayer(nn.Module):
    """
    Expert layer that processes tokens routed to it.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, activation_fn: nn.Module = nn.SiLU()):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = activation_fn
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process the hidden states.
        
        Args:
            hidden_states: [batch_size, hidden_size]
            
        Returns:
            processed_states: [batch_size, hidden_size]
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class MoELayer(nn.Module):
    """
    Mixture of Experts layer that routes tokens to different experts.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        expert_capacity: Optional[int] = None,
        router_jitter_noise: float = 0.0,
        router_z_loss_coef: float = 0.001,
        router_aux_loss_coef: float = 0.001,
        activation_fn: nn.Module = nn.SiLU()
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        
        # Create router
        self.router = Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            jitter_noise=router_jitter_noise,
            expert_capacity=expert_capacity
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation_fn=activation_fn
            )
            for _ in range(num_experts)
        ])
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_capacity: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process hidden states through the MoE layer.
        
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size]
            expert_capacity: Optional override for expert capacity
            
        Returns:
            output_states: [batch_size, sequence_length, hidden_size]
            aux_loss: Dictionary of auxiliary losses
        """
        batch_size, sequence_length, hidden_size = hidden_states.shape
        
        # Route tokens to experts
        expert_indices, expert_weights, routing_weights, router_logits, aux_loss = self.router(
            hidden_states, expert_capacity
        )
        
        # Scale auxiliary losses
        if self.training:
            aux_loss["z_loss"] = self.router_z_loss_coef * aux_loss["z_loss"]
            aux_loss["load_balancing"] = self.router_aux_loss_coef * aux_loss["load_balancing"]
            aux_loss["total"] = aux_loss["z_loss"] + aux_loss["load_balancing"]
        
        # Process tokens through experts
        expert_outputs = torch.zeros(
            batch_size * sequence_length, 
            hidden_size, 
            device=hidden_states.device
        )
        
        # Flatten hidden states
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get indices and weights for this expert
            indices = expert_indices[expert_idx]
            weights = expert_weights[expert_idx]
            
            # Skip if no tokens are routed to this expert
            if torch.all(weights == 0):
                continue
            
            # Get hidden states for this expert
            expert_hidden = flat_hidden_states[indices]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_hidden)
            
            # Scale by routing weights
            expert_output = expert_output * weights.unsqueeze(-1)
            
            # Add to output
            expert_outputs.index_add_(0, indices, expert_output)
        
        # Reshape output
        output_states = expert_outputs.view(batch_size, sequence_length, hidden_size)
        
        return output_states, aux_loss
