"""
DeepSDF Dataset
===============
Loads pre-computed SDF samples from .npz files.

Expected data format per shape:
    An .npz file containing:
        - "points":  (N, 3) float32 — xyz coordinates
        - "sdf":     (N,)   float32 — signed distance values

Directory layout:
    data_dir/
        shape_000.npz
        shape_001.npz
        ...

During training we subsample a fixed number of points per shape per epoch
to keep memory usage bounded.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from DeepSDF_config import DeepSDFConfig


class SDFSamplesDataset(Dataset):
    """
    Each __getitem__ returns a batch of SDF samples for a single shape,
    along with the shape index (used to look up its latent code).
    """

    def __init__(self, data_dir: str, cfg: DeepSDFConfig):
        self.cfg = cfg
        self.data_dir = Path(data_dir)

        # Discover all .npz files and sort for deterministic indexing
        self.shape_files = sorted(self.data_dir.glob("*.npz"))
        if len(self.shape_files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {data_dir}. "
                "See the README for the expected data format."
            )

        self.num_shapes = len(self.shape_files)
        print(f"[Dataset] Found {self.num_shapes} shapes in {data_dir}")

    def __len__(self) -> int:
        return self.num_shapes

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Returns:
            shape_idx: int — index into the latent code table
            points:    (subsample, 3) — xyz coordinates
            sdf_vals:  (subsample, 1) — ground-truth signed distances
        """
        data = np.load(self.shape_files[idx])
        points = data["points"]   # (N, 3)
        sdf = data["sdf"]         # (N,)

        # Subsample to a fixed number of points
        n_available = points.shape[0]
        n_sample = min(self.cfg.subsample_per_shape, n_available)
        choice = np.random.choice(n_available, size=n_sample, replace=False)

        points = torch.tensor(points[choice], dtype=torch.float32)
        sdf = torch.tensor(sdf[choice], dtype=torch.float32).unsqueeze(-1)

        # Clamp SDF values to [-delta, delta] as in the paper
        sdf = torch.clamp(sdf, -self.cfg.clamp_dist, self.cfg.clamp_dist)

        return idx, points, sdf


def collate_sdf_samples(batch):
    """
    Custom collate: stack all shapes' subsampled points into one big batch.

    Each element in `batch` is (shape_idx, points, sdf).
    We expand shape_idx to match the number of points so downstream code
    can index latent codes per-sample.

    Returns:
        shape_indices: (total_points,) — which shape each sample belongs to
        all_points:    (total_points, 3)
        all_sdf:       (total_points, 1)
    """
    indices, points, sdfs = [], [], []
    for shape_idx, pts, sdf in batch:
        n = pts.shape[0]
        indices.append(torch.full((n,), shape_idx, dtype=torch.long))
        points.append(pts)
        sdfs.append(sdf)

    return torch.cat(indices), torch.cat(points), torch.cat(sdfs)
