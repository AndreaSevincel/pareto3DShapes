"""
DeepSDF Configuration
=====================
All hyperparameters collected in one place. Tweak these before training.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepSDFConfig:
    # --- Latent code ---
    latent_dim: int = 256          # Dimensionality of per-shape latent vector
    latent_reg_weight: float = 1e-4  # Weight for latent code L2 regularization
    code_init_std: float = 0.01    # Std for initializing latent codes

    # --- Network architecture ---
    hidden_dims: List[int] = field(
        default_factory=lambda: [512, 512, 512, 512, 512, 512, 512, 512]
    )  # 8-layer MLP as in the paper
    skip_connections: List[int] = field(
        default_factory=lambda: [4]
    )  # Layers where input is re-injected (0-indexed)
    dropout_prob: float = 0.0      # Dropout probability (0 = no dropout)
    use_weight_norm: bool = True   # Apply weight normalization to linear layers

    # --- Training ---
    num_epochs: int = 2000
    batch_size: int = 16384        # Number of SDF samples per batch
    learning_rate: float = 5e-4    # Network LR
    latent_lr: float = 1e-3        # Latent code LR (separate, typically higher)
    lr_schedule_step: int = 500    # Step LR every N epochs
    lr_schedule_gamma: float = 0.5
    clamp_dist: float = 0.1        # Clamp SDF targets to [-delta, delta]

    # --- Data ---
    num_sdf_samples: int = 500000  # Samples per shape for training
    subsample_per_shape: int = 16384  # Samples drawn per shape per epoch

    # --- Inference / mesh extraction ---
    mesh_resolution: int = 256     # Marching cubes grid resolution
    num_optim_iters: int = 800     # Iterations to optimize a latent at test time
    optim_lr: float = 5e-3         # LR for test-time latent optimization
