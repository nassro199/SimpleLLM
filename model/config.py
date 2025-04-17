"""
Configuration for the MoE LLM architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class MoEConfig:
    """
    Configuration class for the Mixture of Experts model.
    """
    # Model size parameters
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    
    # MoE specific parameters
    num_experts: int = 8
    num_experts_per_token: int = 2  # Number of experts to route each token to
    expert_capacity: int = 0  # If 0, capacity is calculated automatically
    router_jitter_noise: float = 0.0  # Add noise to router logits during training
    router_z_loss_coef: float = 0.001  # Coefficient for auxiliary z-loss
    router_aux_loss_coef: float = 0.001  # Coefficient for auxiliary load balancing loss
    
    # Sequence parameters
    max_position_embeddings: int = 4096
    max_sequence_length: int = 4096
    
    # Dropout parameters
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    classifier_dropout: float = 0.1
    
    # Activation function
    hidden_act: str = "silu"
    
    # Layer norm parameters
    layer_norm_eps: float = 1e-5
    use_rms_norm: bool = True  # Use RMSNorm instead of LayerNorm
    
    # Initialization parameters
    initializer_range: float = 0.02
    
    # Positional embedding type
    position_embedding_type: str = "rotary"  # Options: "absolute", "rotary", "alibi"
    rotary_dim: int = 128  # Dimension for rotary embeddings
    
    # Multi-head Latent Attention (MLA) parameters
    use_mla: bool = True  # Whether to use Multi-head Latent Attention
    mla_dim: int = 128  # Dimension for MLA
    
    # Multi-Token Prediction (MTP) parameters
    use_mtp: bool = True  # Whether to use Multi-Token Prediction
    mtp_num_tokens: int = 4  # Number of tokens to predict in MTP
    
    # Training specific parameters
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    # Quantization parameters
    quantization: Optional[str] = None  # Options: None, "int8", "int4", "fp8"
    
    def __post_init__(self):
        """
        Validate and adjust configuration parameters after initialization.
        """
        if self.expert_capacity == 0:
            # Default capacity factor is 1.25 times the expected number of tokens per expert
            tokens_per_expert = 1.0 / self.num_experts_per_token
            self.expert_capacity = int(tokens_per_expert * 1.25)


@dataclass
class TrainingConfig:
    """
    Configuration for training the model.
    """
    # Basic training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Optimizer parameters
    optimizer_type: str = "adamw"  # Options: "adamw", "adafactor", "8bit-adam"
    lr_scheduler_type: str = "cosine"  # Options: "linear", "cosine", "constant"
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # Options: "no", "fp16", "bf16"
    
    # Distributed training
    use_deepspeed: bool = False
    use_fsdp: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Data parameters
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 4
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "moe-llm"
    wandb_run_name: Optional[str] = None


@dataclass
class EvaluationConfig:
    """
    Configuration for model evaluation.
    """
    # Evaluation parameters
    batch_size: int = 16
    max_length: int = 4096
    
    # Benchmarks to evaluate on
    benchmarks: List[str] = field(default_factory=lambda: [
        "mmlu", "gsm8k", "math", "bbh", "arc", "hellaswag", "truthfulqa", "winogrande"
    ])
    
    # Generation parameters
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    num_beams: int = 1
    max_new_tokens: int = 100
