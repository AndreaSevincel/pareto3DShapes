# pareto3DShapes

**Neural Discovery and Generative Modeling of Pareto-Optimal 3D Shapes**

A three-stage neural pipeline that autonomously discovers Pareto-optimal 3D shapes, learns a continuous generative model of the Pareto manifold, and enables single-pass inverse design conditioned on target geometric properties.

> Based on the preprint: *Neural Discovery and Generative Modeling of Pareto-Optimal 3D Shapes* — Andrea Emir Sevincel (2026)

---

## Overview

Designing 3D shapes that simultaneously satisfy multiple competing objectives (e.g. minimise volume *and* surface area) is a fundamental challenge in engineering and computational geometry. This project addresses it with a fully differentiable, three-stage pipeline:

```
Stage 1: Pareto Front Discovery
  → Differentiable multi-objective optimisation over DeepSDF neural signed distance functions
  → Outputs a dataset of N Pareto-optimal shapes with property vectors

Stage 2: CVAE Representation Learning
  → Trains a Conditional Variational Autoencoder on point clouds + property vectors
  → Learns a smooth, sampleable latent space over the Pareto manifold

Stage 3: Inverse Design
  → Given a target property vector, samples and decodes novel shapes in a single forward pass
  → Lifts point clouds to watertight meshes via neural SDF fitting + Marching Cubes
```

---

## Repository Structure

```
pareto3DShapes/
└── test/
    ├── stage1_pareto_deepsdf.ipynb      # Stage 1: Pareto front discovery via DeepSDF optimisation
    ├── cvae_stage2.ipynb                # Stage 2: CVAE training on point clouds
    └── stage3_inverse_design.ipynb      # Stage 3: Inverse design and mesh extraction
```

---

## Pipeline Details

### Stage 1 — Differentiable Shape Optimisation (`stage1_pareto_deepsdf.ipynb`)

Each shape is represented as a neural signed distance function (SDF) in the style of [DeepSDF](https://arxiv.org/abs/1901.05103): an 8-layer MLP with 512 hidden units, softplus activations, and a skip connection.

A diverse set of Pareto-optimal shapes is discovered by solving a weighted scalarisation of objectives across N different weight vectors sampled via Das–Dennis decomposition:

$$\mathcal{L}_1(\theta, z, \alpha) = -\sum_i \alpha_i f_i(S_\theta(\cdot; z)) + \lambda_\text{eik}\mathcal{L}_\text{eik} + \lambda_\text{smooth}\mathcal{L}_\text{smooth} + \lambda_\text{lat}\mathcal{L}_\text{lat}$$

**Differentiable objectives implemented:**
- **Volume** — soft sigmoid occupancy approximation with temperature annealing
- **Surface Area** — Gaussian delta approximation via the co-area formula
- **Curvature** — mean curvature via second-order automatic differentiation

**Regularisation terms:**
- Eikonal penalty (enforces `‖∇S‖ = 1`)
- Smoothness penalty (near-surface Laplacian)
- Latent vector L2 regularisation

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate (θ) | 5×10⁻⁴ |
| Learning rate (z) | 10⁻³ |
| Iterations per shape | 5000 |
| Latent dimension | 128 |
| λ_eik | 0.1 |
| λ_smooth | 0.01 |
| σ²_z | 0.01 |

---

### Stage 2 — CVAE Representation Learning (`cvae_stage2.ipynb`)

Each Pareto-optimal SDF is discretised into a point cloud of `N_pts` points sampled near the zero level set. A Conditional VAE is trained on the dataset `{(q⁽ⁱ⁾, P⁽ⁱ⁾)}` where `q` is the property vector and `P` is the point cloud.

**Architecture:**
- **Encoder:** MLP over `[q; flatten(P)]` → Gaussian posterior `(μ, log σ²)`
- **Reparameterisation:** `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`
- **Decoder:** MLP over `[z; q]` → reconstructed point cloud *(used alone at inference)*

**Training objective:**

$$\mathcal{L}_2 = d_\text{CD}(P, \hat{P}) + \beta \mathcal{L}_\text{KL} + \lambda \mathcal{L}_\text{prop}$$

where `d_CD` is the Chamfer distance, `L_KL` is the closed-form KL divergence, and `L_prop` is a property consistency loss from a jointly trained property predictor.

KL annealing (`β`: 0→1 over `T_warm` epochs) is used to prevent posterior collapse.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Optimiser | Adam + cosine annealing |
| Learning rate | 10⁻³ → 10⁻⁵ |
| Epochs | 200 |
| Latent dimension | 64 |
| KL warmup | 50 epochs |
| λ (property consistency) | 0.1 |
| Batch size | 32 shapes |

---

### Stage 3 — Inverse Design (`stage3_inverse_design.ipynb`)

Given a target property vector `q*`, novel shapes are synthesised in a single decoder forward pass:

1. Sample `z ~ N(0, I)`
2. Decode: `P̂ = Dec_ϕ(z, q*)`
3. Fit a new DeepSDF network to `P̂` (minimising on-surface and Eikonal losses)
4. Evaluate on a regular 3D grid and extract the mesh via **Marching Cubes** (resolution R=256)

Multiple samples of `z` at the same `q*` yield geometrically diverse shapes all targeting the same Pareto operating point.

**Evaluation metric — Property Satisfaction Error (PSE):**

$$\text{PSE}(\hat{P}, q^*) = \|\hat{q} - q^*\|_2$$

where `q̂` are the differentiable objective values of the generated shape. Lower PSE = better inverse design fidelity.

---

## Requirements

```bash
pip install torch numpy scipy scikit-learn matplotlib open3d trimesh
```

Optional (for differentiable rendering / shadow objectives):
```bash
pip install nvdiffrast
```

---

## Getting Started

Run the notebooks in order:

```bash
# 1. Discover Pareto-optimal shapes
jupyter notebook test/stage1_pareto_deepsdf.ipynb

# 2. Train the CVAE
jupyter notebook test/cvae_stage2.ipynb

# 3. Run inverse design
jupyter notebook test/stage3_inverse_design.ipynb
```

---

## Citation

If you use this code, please cite the accompanying preprint:

```bibtex
@article{sevincel2026pareto3d,
  title   = {Neural Discovery and Generative Modeling of Pareto-Optimal 3D Shapes},
  author  = {Sevincel, Andrea Emir},
  year    = {2026},
  note    = {Preprint}
}
```

---

## References

- Park et al., *DeepSDF*, CVPR 2019
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014
- Lorensen & Cline, *Marching Cubes*, SIGGRAPH 1987
- Deb et al., *NSGA-II*, IEEE TEC 2002
- Navon et al., *Learning the Pareto Front with Hypernetworks*, ICLR 2021
