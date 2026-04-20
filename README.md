# Pareto3DShapes

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview
**Pareto3DShapes** is a framework dedicated to the multi-objective optimization and inverse design of 3D geometries. By leveraging implicit neural representations and conditional generative models, this project establishes a robust latent space for 3D shapes. This enables the discovery of optimal geometric configurations that satisfy competing design criteria, effectively navigating the Pareto frontier.

This repository provides a complete pipeline from learning continuous signed distance functions to performing inverse design for targeted 3D object generation.

## Research Objectives
* **High-Fidelity 3D Encoding:** Utilize DeepSDF to learn continuous and memory-efficient representations of complex 3D topologies.
* **Generative Modeling:** Model the latent distribution of 3D shapes to allow for smooth interpolation and novel shape synthesis.
* **Inverse Design & Multi-Objective Optimization:** Implement Pareto optimization techniques (e.g., NSGA-II or gradient-based Pareto descent) to generate shapes that meet specific, potentially conflicting, target metrics (e.g., maximizing aerodynamic efficiency while minimizing structural weight).

## Pipeline Architecture

The framework is modularized into three distinct stages to facilitate experimentation and ablation studies.

### Stage 1: Shape Representation (DeepSDF)
Constructs the foundational shape embeddings. Models learn a continuous Signed Distance Function to represent boundary surfaces with high precision.

### Stage 2: Latent Space Modeling (CVAE)
Trains a Conditional Variational Autoencoder on the latent vectors generated in Stage 1. This stage maps the discrete shape embeddings into a continuous, probabilistic latent space, enabling targeted generation.

### Stage 3: Inverse Design & Pareto Optimization
Employs optimization algorithms to traverse the CVAE latent space. It searches for specific latent vectors that decode into 3D shapes satisfying predefined multi-objective criteria, yielding a diverse set of Pareto-optimal designs.

## Installation
