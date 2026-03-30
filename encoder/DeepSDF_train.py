"""
DeepSDF Training — Auto-Decoder
================================
Joint optimization of:
    1. Network weights θ  (the MLP that predicts SDF)
    2. Latent codes {z_i} (one per training shape, learned directly)

There is NO encoder. Latent codes are free parameters optimized with
gradient descent — this is what makes it an "auto-decoder".

Loss = L1(predicted_sdf, target_sdf) + λ * ||z||²
         ↑ reconstruction                ↑ regularization (keeps codes small)
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DeepSDF_config import DeepSDFConfig
from DeepSDF_model import DeepSDFNetwork
from DeepSDF_dataset import SDFSamplesDataset, collate_sdf_samples


def train(data_dir: str, output_dir: str = "checkpoints"):
    cfg = DeepSDFConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    dataset = SDFSamplesDataset(data_dir, cfg)
    loader = DataLoader(
        dataset,
        batch_size=8,             # 8 shapes per batch (not 8 points!)
        shuffle=True,
        num_workers=4,
        collate_fn=collate_sdf_samples,
        pin_memory=True,
    )

    # ── Network ───────────────────────────────────────────────────────
    network = DeepSDFNetwork(cfg).to(device)

    # ── Latent codes (the auto-decoder part) ──────────────────────────
    # One learnable vector per training shape, initialized from N(0, σ²)
    latent_codes = torch.nn.Embedding(dataset.num_shapes, cfg.latent_dim).to(device)
    nn.init.normal_(latent_codes.weight, mean=0.0, std=cfg.code_init_std)

    # ── Optimizers ────────────────────────────────────────────────────
    # Separate optimizers for network and latent codes (different LRs)
    optimizer_net = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate)
    optimizer_lat = torch.optim.Adam(latent_codes.parameters(), lr=cfg.latent_lr)

    scheduler_net = torch.optim.lr_scheduler.StepLR(
        optimizer_net, step_size=cfg.lr_schedule_step, gamma=cfg.lr_schedule_gamma
    )
    scheduler_lat = torch.optim.lr_scheduler.StepLR(
        optimizer_lat, step_size=cfg.lr_schedule_step, gamma=cfg.lr_schedule_gamma
    )

    # ── Loss ──────────────────────────────────────────────────────────
    loss_l1 = nn.L1Loss(reduction="mean")

    # ── Training loop ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, cfg.num_epochs + 1):
        network.train()
        epoch_loss = 0.0
        num_batches = 0

        for shape_indices, points, sdf_gt in loader:
            shape_indices = shape_indices.to(device)
            points = points.to(device)
            sdf_gt = sdf_gt.to(device)

            # Look up latent codes for each sample's shape
            z = latent_codes(shape_indices)  # (total_points, latent_dim)

            # Forward pass
            sdf_pred = network(z, points)    # (total_points, 1)

            # Clamp predictions the same way targets are clamped
            sdf_pred = torch.clamp(sdf_pred, -cfg.clamp_dist, cfg.clamp_dist)

            # Reconstruction loss
            recon_loss = loss_l1(sdf_pred, sdf_gt)

            # Latent regularization: penalize large codes
            # We regularize only the codes actually used in this batch
            unique_indices = torch.unique(shape_indices)
            used_codes = latent_codes(unique_indices)
            reg_loss = cfg.latent_reg_weight * torch.mean(used_codes.pow(2))

            total_loss = recon_loss + reg_loss

            # Backward + step (both optimizers)
            optimizer_net.zero_grad()
            optimizer_lat.zero_grad()
            total_loss.backward()
            optimizer_net.step()
            optimizer_lat.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        scheduler_net.step()
        scheduler_lat.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        if epoch % 50 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:>5d}/{cfg.num_epochs}]  "
                f"loss={avg_loss:.6f}  "
                f"lr_net={scheduler_net.get_last_lr()[0]:.2e}  "
                f"lr_lat={scheduler_lat.get_last_lr()[0]:.2e}"
            )

        # Periodic checkpoint
        if epoch % 500 == 0:
            ckpt_path = Path(output_dir) / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "network_state_dict": network.state_dict(),
                    "latent_codes": latent_codes.state_dict(),
                    "optimizer_net": optimizer_net.state_dict(),
                    "optimizer_lat": optimizer_lat.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  → Saved checkpoint: {ckpt_path}")

    print("[Train] Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DeepSDF (auto-decoder)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to SDF .npz files")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(args.data_dir, args.output_dir)
