"""
DeepSDF Network
===============
MLP that maps (latent_code, xyz) → signed distance value.

Architecture follows Park et al. 2019:
- 8 fully-connected layers (512 units each by default)
- Skip connection at layer 4: re-inject the input (latent + xyz)
- Weight normalization on all linear layers
- ReLU activations, tanh on the final output
"""

import torch
import torch.nn as nn

from DeepSDF_config import DeepSDFConfig


class DeepSDFNetwork(nn.Module):
    def __init__(self, cfg: DeepSDFConfig):
        super().__init__()
        self.cfg = cfg
        self.skip_connections = cfg.skip_connections

        input_dim = cfg.latent_dim + 3  # latent code + (x, y, z)
        dims = [input_dim] + cfg.hidden_dims + [1]  # final output is 1 scalar

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(dims) - 1):
            in_features = dims[i]

            # At skip-connection layers, the input is concatenated back
            if i in self.skip_connections:
                in_features += input_dim

            out_features = dims[i + 1]

            linear = nn.Linear(in_features, out_features)

            # Apply weight normalization if configured
            if cfg.use_weight_norm:
                linear = nn.utils.parametrizations.weight_norm(linear)

            self.layers.append(linear)

            # All hidden layers get ReLU; the final layer has no activation here
            # (we apply tanh separately in forward)
            if i < len(dims) - 2:
                self.activations.append(nn.ReLU(inplace=True))
            else:
                self.activations.append(None)

        self.dropout = nn.Dropout(cfg.dropout_prob) if cfg.dropout_prob > 0 else None

    def forward(self, latent_code: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_code: (B, latent_dim) — per-shape latent vectors
            xyz:         (B, 3)          — query point coordinates

        Returns:
            sdf: (B, 1) — predicted signed distance values
        """
        # Concatenate latent code with spatial coordinates
        x = torch.cat([latent_code, xyz], dim=-1)
        input_feat = x  # Save for skip connections

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            # Re-inject input at skip-connection layers
            if i in self.skip_connections:
                x = torch.cat([x, input_feat], dim=-1)

            x = layer(x)

            if activation is not None:
                x = activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)

        # tanh squashes output to (-1, 1); actual SDF range is handled by
        # the clamping distance during training
        sdf = torch.tanh(x)
        return sdf
