"""
DeepSDF Inference
=================
Two-stage process to reconstruct a mesh for a new (or known) shape:

Stage 1 — Latent code optimization:
    Given observed SDF samples of a target shape, optimize a latent vector z*
    that best explains those samples through the frozen network.
    This is the auto-decoder inference: argmin_z  L1(f(z, x), sdf_gt) + λ||z||²

Stage 2 — Mesh extraction:
    Evaluate the network on a dense 3D grid using z*, then run
    Marching Cubes on the resulting SDF volume to extract the zero-level set.
"""

import numpy as np
import torch
import torch.nn as nn

from DeepSDF_config import DeepSDFConfig
from DeepSDF_model import DeepSDFNetwork


def optimize_latent(
    network: DeepSDFNetwork,
    points: torch.Tensor,
    sdf_gt: torch.Tensor,
    cfg: DeepSDFConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Optimize a latent code to fit observed SDF samples.

    Args:
        network:  Trained DeepSDF network (frozen during this step)
        points:   (N, 3) — observed 3D coordinates
        sdf_gt:   (N, 1) — observed SDF values at those points
        cfg:      Config object
        device:   torch device

    Returns:
        z_star: (1, latent_dim) — the optimized latent code
    """
    network.eval()
    for param in network.parameters():
        param.requires_grad = False

    # Initialize a latent code from a small Gaussian
    z = torch.randn(1, cfg.latent_dim, device=device) * cfg.code_init_std
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=cfg.optim_lr)
    loss_fn = nn.L1Loss()

    points = points.to(device)
    sdf_gt = sdf_gt.to(device)

    for step in range(cfg.num_optim_iters):
        optimizer.zero_grad()

        # Expand z to match all query points: (N, latent_dim)
        z_expanded = z.expand(points.shape[0], -1)

        sdf_pred = network(z_expanded, points)
        sdf_pred = torch.clamp(sdf_pred, -cfg.clamp_dist, cfg.clamp_dist)

        recon_loss = loss_fn(sdf_pred, sdf_gt)
        reg_loss = cfg.latent_reg_weight * torch.mean(z.pow(2))
        total_loss = recon_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(
                f"  [Latent optim step {step + 1}/{cfg.num_optim_iters}]  "
                f"loss={total_loss.item():.6f}"
            )

    return z.detach()


@torch.no_grad()
def extract_mesh(
    network: DeepSDFNetwork,
    latent_code: torch.Tensor,
    cfg: DeepSDFConfig,
    device: torch.device,
    bounds: tuple = (-1.0, 1.0),
):
    """
    Evaluate the SDF on a dense 3D grid and extract the zero-level isosurface.

    Args:
        network:     Trained DeepSDF network
        latent_code: (1, latent_dim)
        cfg:         Config
        device:      torch device
        bounds:      (min, max) spatial extent of the grid

    Returns:
        vertices: (V, 3) numpy array
        faces:    (F, 3) numpy array  (triangle indices)
    """
    try:
        import mcubes  # PyMCubes — install via: pip install PyMCubes
    except ImportError:
        raise ImportError(
            "Mesh extraction requires PyMCubes. Install it with:\n"
            "  pip install PyMCubes"
        )

    network.eval()
    res = cfg.mesh_resolution
    lo, hi = bounds

    # Create a 3D grid of query points
    coords = torch.linspace(lo, hi, res)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)

    # Evaluate in chunks to avoid OOM on large grids
    sdf_values = []
    chunk_size = 64 * 1024  # 65k points per forward pass

    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        pts_chunk = grid_points[start:end]

        z_expanded = latent_code.expand(pts_chunk.shape[0], -1)
        sdf_chunk = network(z_expanded, pts_chunk)
        sdf_values.append(sdf_chunk.cpu())

    sdf_volume = torch.cat(sdf_values, dim=0).numpy().reshape(res, res, res)

    # Marching cubes: extract the zero-level set
    vertices, faces = mcubes.marching_cubes(sdf_volume, 0.0)

    # Rescale vertices from grid indices [0, res-1] back to world coordinates
    vertices = vertices / (res - 1) * (hi - lo) + lo

    return vertices, faces


def save_mesh_as_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Write vertices and faces to a Wavefront .obj file."""
    with open(filepath, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[Mesh] Saved {len(vertices)} vertices, {len(faces)} faces → {filepath}")


def reconstruct_from_checkpoint(
    checkpoint_path: str,
    sdf_samples_path: str,
    output_obj: str = "reconstructed.obj",
):
    """
    Full inference pipeline:
        1. Load trained checkpoint
        2. Optimize a latent code on the given SDF samples
        3. Extract mesh and save as .obj
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    network = DeepSDFNetwork(cfg).to(device)
    network.load_state_dict(ckpt["network_state_dict"])

    # Load target shape's SDF samples
    data = np.load(sdf_samples_path)
    points = torch.tensor(data["points"], dtype=torch.float32)
    sdf_gt = torch.tensor(data["sdf"], dtype=torch.float32).unsqueeze(-1)
    sdf_gt = torch.clamp(sdf_gt, -cfg.clamp_dist, cfg.clamp_dist)

    # Stage 1: optimize latent code
    print("[Inference] Optimizing latent code...")
    z_star = optimize_latent(network, points, sdf_gt, cfg, device)

    # Stage 2: extract mesh
    print("[Inference] Extracting mesh via Marching Cubes...")
    vertices, faces = extract_mesh(network, z_star, cfg, device)
    save_mesh_as_obj(vertices, faces, output_obj)
    print("[Inference] Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepSDF inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sdf_samples", type=str, required=True, help=".npz file")
    parser.add_argument("--output", type=str, default="reconstructed.obj")
    args = parser.parse_args()

    reconstruct_from_checkpoint(args.checkpoint, args.sdf_samples, args.output)
