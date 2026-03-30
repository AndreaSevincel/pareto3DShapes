# DeepSDF — step-by-step usage guide

## Step 1: install dependencies

```bash
pip install torch numpy scipy
pip install PyMCubes       # for mesh extraction (marching cubes)
pip install trimesh        # only if your point clouds are .ply files
pip install matplotlib     # only if you want cross-section visualizations
```

If you're on a machine with a CUDA GPU (recommended for training):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2: prepare your training data

DeepSDF expects one `.npz` file per shape, each containing:
- `"points"`: (N, 3) float32 — xyz coordinates of query points in 3D space
- `"sdf"`: (N,) float32 — signed distance value at each query point

These are NOT your raw point clouds. They are sampled query points
spread throughout 3D space, each labeled with their signed distance
to the surface. Your raw point clouds are the surface — the preprocessing
step samples points around (and far from) that surface and computes the SDF.

### Option A: you have point cloud solids (.ply, .xyz, .npy)

Use the `preprocess_pointclouds.py` script (provided separately):

```bash
# Put your point clouds in a folder
mkdir -p data/raw_pointclouds
cp your_shapes/*.ply data/raw_pointclouds/

# Run preprocessing
python preprocess_pointclouds.py \
    --input_dir data/raw_pointclouds \
    --output_dir data/sdf_samples \
    --num_query_points 500000 \
    --k_neighbors 30

# Validate the output (recommended — catches normal estimation bugs)
python validate_preprocessing.py --data_dir data/sdf_samples --export_slices
```

### Option B: you have meshes (.obj, .stl) with faces

Use the `mesh_to_sdf` library:

```bash
pip install mesh-to-sdf
```

Then write a small script:

```python
import numpy as np
import trimesh
import mesh_to_sdf

mesh = trimesh.load("your_shape.obj")
points, sdf = mesh_to_sdf.sample_sdf_near_surface(mesh, number_of_points=500000)
np.savez_compressed("data/sdf_samples/shape_000.npz", points=points, sdf=sdf)
```

### Option C: test with synthetic data first

Use `generate_data.py` (provided separately) to create spheres, tori, and boxes:

```bash
python generate_data.py --output_dir data/sdf_samples --num_shapes 20
```

### Final data layout

After preprocessing, you should have:

```
data/sdf_samples/
    shape_000.npz
    shape_001.npz
    shape_002.npz
    ...
```

---

## Step 3: train

```bash
cd your_project/
python train.py --data_dir data/sdf_samples --output_dir checkpoints
```

### What happens during training

1. The dataset loader (`dataset.py`) discovers all `.npz` files in `data/sdf_samples/`
2. For each shape, it randomly subsamples 16,384 points per epoch
3. The training loop jointly optimizes:
   - The MLP weights (the network that predicts SDF values)
   - One latent code vector per shape (256 dimensions each)
4. Checkpoints are saved every 500 epochs to `checkpoints/`

### What to watch for

The loss should decrease steadily. Typical values:
- Epoch 1: loss ~0.05–0.10
- Epoch 500: loss ~0.005–0.01
- Epoch 2000: loss ~0.001–0.005

If the loss plateaus early or oscillates, possible causes:
- Data issue: run `validate_preprocessing.py` to check
- Learning rate too high: reduce `learning_rate` and `latent_lr` in `config.py`
- Too few shapes: DeepSDF works best with 50+ shapes per class

### Key config parameters you might want to change

Edit `config.py` before training:

```python
num_epochs: int = 2000        # More epochs = better quality, more time
batch_size: int = 16384       # Reduce if you hit OOM errors
learning_rate: float = 5e-4   # Network learning rate
latent_lr: float = 1e-3       # Latent code learning rate (usually higher)
latent_dim: int = 256         # Size of shape code (reduce for few shapes)
clamp_dist: float = 0.1       # SDF clamping range
```

If you have very few shapes (< 20), consider reducing `latent_dim` to 64 or 128
to prevent overfitting.

### Training time estimates

- 20 shapes, CPU: ~2–4 hours for 2000 epochs
- 20 shapes, GPU (RTX 3060+): ~15–30 minutes
- 1000 shapes, GPU: ~2–4 hours

---

## Step 4: inspect results — reconstruct a training shape

After training finishes, test that the model learned correctly by
reconstructing a shape it was trained on:

```bash
python inference.py \
    --checkpoint checkpoints/checkpoint_epoch2000.pt \
    --sdf_samples data/sdf_samples/shape_000.npz \
    --output reconstructed_shape_000.obj
```

This does two things:
1. Optimizes a latent code to fit shape_000's SDF samples (800 iterations)
2. Evaluates the network on a 256³ grid and runs Marching Cubes

The output is a `.obj` mesh file. Open it in any 3D viewer:
- MeshLab (free): meshlab.net
- Blender (free): blender.org
- Online: 3dviewer.net

If the reconstructed mesh looks like the original shape, training worked.

---

## Step 5: reconstruct a NEW shape (not in training set)

This is the auto-decoder inference. Given SDF samples from a shape the
network has never seen, it optimizes a new latent code from scratch:

```bash
# Preprocess the new point cloud first
python preprocess_pointclouds.py \
    --input_dir data/new_shape/ \
    --output_dir data/new_sdf/

# Then reconstruct
python inference.py \
    --checkpoint checkpoints/checkpoint_epoch2000.pt \
    --sdf_samples data/new_sdf/shape_000.npz \
    --output new_shape_reconstructed.obj
```

Quality depends on how similar the new shape is to the training shapes.
If you trained on chairs and try to reconstruct an airplane, it won't work well.

---

## How the files connect

```
config.py          All hyperparameters. Edit this before training.
    ↓
model.py           The 8-layer MLP. Takes (latent_code, xyz) → SDF value.
    ↓               Imported by train.py and inference.py.
dataset.py         Loads .npz files, subsamples points per epoch.
    ↓               Imported by train.py.
train.py           Training loop. Creates latent codes + optimizes everything.
    ↓               Saves checkpoints to disk.
inference.py       Loads a checkpoint, optimizes a latent for a new shape,
                   extracts mesh via Marching Cubes.
```

---

## Common errors and fixes

### `ModuleNotFoundError: No module named 'config'`
You didn't rename the files. See Step 0.

### `ModuleNotFoundError: No module named 'model'`
Same issue. `DeepSDF.py` must be renamed to `model.py`.

### `FileNotFoundError: No .npz files found`
The `--data_dir` path is wrong, or you haven't run preprocessing yet.
Check that the directory contains `shape_*.npz` files.

### `RuntimeError: CUDA out of memory`
Reduce `subsample_per_shape` in `config.py` (try 8192 instead of 16384),
or reduce the DataLoader `batch_size` in `train.py` (try 4 instead of 8).

### `ImportError: PyMCubes`
```bash
pip install PyMCubes
```

### Reconstructed mesh is empty or garbage
- Check preprocessing: `python validate_preprocessing.py --data_dir data/sdf_samples`
- Train for more epochs
- Check that loss decreased during training

### Loss stays flat / doesn't decrease
- Verify your data: the `.npz` files should have both positive and negative SDF values
- Try reducing `learning_rate` to 1e-4
- Check that `clamp_dist` (0.1) is appropriate for your data scale
