# Transolver & GeoTransolver Tutorial: From Theory to Training + UQ

Welcome to this tutorial series! This hands-on guide walks through the journey from standard Transformer theory to **Transolver** and **GeoTransolver** — state-of-the-art AI surrogate models for predicting physical fields on complex 3D geometries.

Our goal is to bridge the gap between abstract AI concepts and a production-ready simulation pipeline built on **NVIDIA PhysicsNeMo**. By the end of the series, you will understand how to train models that predict aerodynamic surface fields (pressure, wall shear stress, etc.) on arbitrary car bodies — tasks that would classically require hours of CFD simulation.

### Papers

> **Transolver: A Fast Transformer Solver for PDEs on General Geometries**
> Wu et al., 2024 — [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)

> **GeoTransolver** — NVIDIA PhysicsNeMo extension adding Geometry-Aware Layer Ensemble (GALE):
> [`physicsnemo.experimental.models.geotransolver`](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transformer_models)

---

## Series Overview

The series consists of seven notebooks. **Run Notebook 0 once** before the training notebooks — it produces the Zarr dataset and normalization file that Notebooks 3 and 5 both share.

| # | Notebook | Topic |
|---|----------|-------|
| **0** | **Data Preprocessing** | **VTP/STL → Zarr, per-case velocity from info files, normalization** |
| 1 | Understanding Transformers & the Bottleneck | Why standard attention fails at $O(N^2)$ for large meshes |
| 2 | Understanding Transolver | Physics-Attention: Slice → Aggregate → Attend → Deslice |
| 3 | Training Transolver | `TransolverDataPipe` training loop, slice visualization |
| 4 | Understanding GALE & GeoTransolver | Geometry-Aware Layer Ensemble and how it extends Transolver |
| 5 | Training GeoTransolver | Same pipeline as NB3, `broadcast_global_features=False`, GALE |
| 6 | Uncertainty Quantification | MC-Dropout & Concrete Dropout for per-point epistemic uncertainty |

---

### Notebook 0 — Data Preprocessing (run once, shared by NB3 & NB5)

This notebook converts the raw Ahmed body dataset into the Zarr format that both training notebooks consume:

- Reads each case's inlet velocity from its companion info file (velocity differs per simulation — it is not a single global constant).
- Saves `air_density` as a config constant and `stream_velocity` per-case into every Zarr store.
- Saves `stl_coordinates` (raw STL vertices) for GeoTransolver's GALE cross-attention in Notebook 5.
- Computes per-channel normalization statistics with Welford's online algorithm (training split only) and writes `surface_fields_normalization.npz`.

---

### Notebook 1 — The "Why": The Quadratic Bottleneck

We start with the core **attention mechanism** of the standard Transformer. You will discover its critical limitation: **quadratic complexity $O(N^2)$**. We demonstrate why this makes standard Transformers impractical for the millions of mesh points found in real engineering simulations.

### Notebook 2 — The "How": Physics-Attention Mechanics

We build a **NumPy blueprint** of Transolver's solution: **Physics-Attention**. You implement the four-step process — **Slice, Aggregate, Attend, Deslice** — that breaks the $O(N^2)$ wall. The key insight: Transolver groups $N$ mesh points into $M \ll N$ "physics-aware slice tokens," so attention runs in $O(M^2)$ instead.

### Notebook 3 — The "Factory": Training Transolver

We move from theory to a production-grade implementation following the **PhysicsNeMo** `transformer_models/src` workflow. Notebook 0 must be run first to produce the Zarr dataset. This notebook then focuses on:

- **Data pipeline:** `TransolverDataPipe` + `CAEDataset` — reads the Zarr stores from NB0, handling centering, subsampling, and normalization internally.
- **Model selection via YAML:** In production, `conf/model/transolver.yaml` selects the model class through Hydra:
  ```yaml
  _target_: physicsnemo.models.transolver.Transolver
  functional_dim: 2   # air_density + stream_velocity — broadcast to all N mesh points
  embedding_dim:  6   # mesh_centers(3) + normals(3)
  out_dim:        4   # Cp, Cf_x, Cf_y, Cf_z
  ```
  Because Transolver has no dedicated global-conditioning pathway, `broadcast_global_features: true` copies the two flow scalars to every mesh point so they enter the model alongside the local geometry features.
- **Training:** Abbreviated loop using the same `forward_pass` logic as the production `train.py`.
- **Visualization:** Ascription weight maps (learned slice assignments) and Shannon entropy analysis to quantify and spatially map model uncertainty.

### Notebook 4 — Understanding GALE & GeoTransolver

We examine the **Geometry-Aware Layer Ensemble (GALE)** — a cross-attention mechanism that re-injects raw STL geometry at every transformer block to prevent representation drift in deep physics-attention stacks. You will understand:

- Why geometry context fades in deep networks and why GALE is needed
- How GALE cross-attention works mathematically
- The full architectural difference between Transolver and GeoTransolver

### Notebook 6 — Uncertainty Quantification

Adds per-point epistemic confidence estimates to the trained Transolver (or GeoTransolver) without retraining from scratch.

**MC-Dropout:** enable dropout at inference time (`enable_dropout(model)`), run T stochastic forward passes, compute per-point mean and standard deviation. Backed by Gal & Ghahramani's result that dropout networks approximate Bayesian inference over model weights.

**Concrete Dropout:** treat the dropout probability *p* as a learnable parameter. A KL-based regularization term (Bernoulli entropy + weight-norm penalty) is added to the training loss, allowing the model to discover the optimal *p* per layer rather than relying on manual tuning.

The two methods compose: train with Concrete Dropout (learned *p*), then run MC-Dropout at inference with those learned *p* values. The resulting per-point σ map highlights trailing edges, slant junctions, and underbody corners — exactly the regions where CFD verification is most needed.

---

### Notebook 5 — Training GeoTransolver

GeoTransolver training uses the **exact same pipeline classes** as Notebook 3 (`TransolverDataPipe` / `CAEDataset`), differing only in two configuration flags:

| Flag | NB3 — Transolver | NB5 — GeoTransolver |
|------|-----------------|---------------------|
| `broadcast_global_features` | `True` — copies air_density + stream_velocity to all $N$ points; `batch["fx"].shape = (1, N, 2)` | `False` — passes them as a single global vector; `batch["fx"].shape = (1, 1, 2)` |
| `include_geometry` | `False` | `True` — adds `batch["geometry"]` (STL vertices, shape `(1, M, 3)`) for GALE cross-attention |

In production, `conf/model/geotransolver.yaml` selects the model:
```yaml
_target_: physicsnemo.experimental.models.geotransolver.GeoTransolver
functional_dim: 6   # local_embedding: coords(3) + normals(3)
global_dim:     2   # global_embedding: air_density + stream_velocity (routed through GALE)
geometry_dim:   3   # STL vertex coordinates for GALE cross-attention
out_dim:        4
```

---

## Prerequisites

### 1. NVIDIA NGC Account

You need an NGC account to pull the Docker container:
[NVIDIA NGC Docker Setup Guide](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html)

### 2. Ahmed Body Dataset

Download the **Ahmed body surface dataset** from NGC:
[https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset)

After extracting, confirm the directory structure:

```
physicsnemo_ahmed_body_dataset_vv1/dataset/
├── train/
├── train_info/
├── train_stl_files/
├── validation/
├── validation_info/
├── validation_stl_files/
├── test/
├── test_info/
└── test_stl_files/
```

---

## Environment Setup

### Step 1 — Pull the PhysicsNeMo 26.05 Container

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:26.05
```

### Step 2 — Launch the Container

Replace `<path_on_host>` with the absolute path to the directory containing your notebooks and Ahmed body dataset. This directory is mounted as `/workspace` inside the container.

```bash
docker run --gpus 1 --shm-size=2g -p 7008:7008 \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --runtime nvidia \
    -v <path_on_host>:/workspace \
    -it --rm \
    nvcr.io/nvidia/physicsnemo/physicsnemo:26.05
```

### Step 3 — Install Additional Dependencies (Inside Container)

```bash
# System packages
# xvfb provides a virtual framebuffer required by PyVista in headless environments
apt-get update && apt-get install -y rsync xvfb

# Python packages
pip install hydra-core tabulate tensorboard termcolor torchinfo einops \
    "transformer_engine[pytorch]" "zarr>=3.0"
```

### Step 4 — Start Jupyter Lab

Run inside the container to launch Jupyter Lab in the background:

```bash
nohup python3 -m jupyter lab \
    --ip=0.0.0.0 --port=7008 --allow-root --no-browser \
    --NotebookApp.token='' \
    --notebook-dir='/workspace/' \
    --NotebookApp.allow_origin='*' \
    > /dev/null 2>&1 &
```

### Step 5 — Access Jupyter Lab

**Remote host:** Create an SSH tunnel from your local machine (replace `<remote_hostname>` and `<ssh_alias>` as appropriate):

```bash
ssh -L 3030:<remote_hostname>:7008 <ssh_alias>
```

Then open `http://localhost:3030` in your browser.

**Local machine:** Open `http://localhost:7008` directly.

---

## Repository Structure

```
Transolver/
├── Notebook0-Data-Preprocessing.ipynb           ← run first (shared by NB3 & NB5)
├── Notebook1-Understanding-Transformers-and-Bottleneck.ipynb
├── Notebook2-Understanding-Transolver.ipynb
├── Notebook3-Training-Transolver.ipynb
├── Notebook4-Understanding-GALE-GeoTransolver.ipynb
├── Notebook5-Training-GeoTransolver.ipynb
├── Notebook6-Uncertainty-Quantification.ipynb   ← MC-Dropout & Concrete Dropout
├── requirements.txt          # Python dependencies (see Step 3)
├── utils/                    # shared utility modules
└── fig/                      # figures referenced in the notebooks
```

---

## References

- Wu, H., et al. (2024). *Transolver: A Fast Transformer Solver for PDEs on General Geometries.* [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation.* ICML 2016.
- Gal, Y., Hron, J., & Kendall, A. (2017). *Concrete Dropout.* NeurIPS 2017.
- NVIDIA PhysicsNeMo: [https://github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)
- Ahmed Body Example: [`examples/cfd/external_aerodynamics/transformer_models`](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transformer_models)
- PhysicsNeMo 26.05 Container: `nvcr.io/nvidia/physicsnemo/physicsnemo:26.05`
