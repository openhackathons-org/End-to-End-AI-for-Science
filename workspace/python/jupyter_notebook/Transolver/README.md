# Transolver & GeoTransolver Tutorial: From Theory to Training + UQ

Welcome to this tutorial series! This hands-on guide walks through the journey from standard Transformer theory to **Transolver** and **GeoTransolver** — state-of-the-art AI surrogate models for predicting physical fields on complex 3D geometries.

Our goal is to bridge the gap between abstract AI concepts and a production-ready simulation pipeline built on **NVIDIA PhysicsNeMo**. By the end of the series, you will understand how to train models that predict aerodynamic surface fields (pressure, wall shear stress) on arbitrary car bodies — tasks that would classically require hours of CFD simulation.

### Papers

> **Transolver: A Fast Transformer Solver for PDEs on General Geometries**
> Wu et al., 2024 — [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)

> **GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer**
> 2025 — [arXiv:2512.20399](https://arxiv.org/abs/2512.20399)
> Adds **GALE (Geometry-Aware Latent Embeddings)** attention to the Transolver backbone.
> Implementation: [`physicsnemo.experimental.models.geotransolver`](https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/experimental/models/geotransolver)
> Training recipe: [`examples/cfd/external_aerodynamics/transformer_models`](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transformer_models)

---

## Series Overview

Seven notebooks. **Run Notebook 0 once** before the training notebooks — it produces the Zarr dataset and normalization file that Notebooks 3 and 5 both share.

| # | Notebook | Topic | Needs the dataset? |
|---|----------|-------|---|
| **0** | **Data Preprocessing** | VTP/STL → Zarr, per-case velocity, normalization | **yes** |
| 1 | Understanding Transformers & the Bottleneck | Why standard attention fails at $O(N^2)$ | partly (Part 3 only) |
| 2 | Understanding Transolver | Physics-Attention: Slice → Aggregate → Attend → Deslice | no |
| 3 | Training Transolver | `TransolverDataPipe` loop, slice + entropy analysis | yes |
| 4 | Understanding GALE & GeoTransolver | Geometry-Aware Latent Embeddings | optional (Section 5.7 reads a checkpoint) |
| 5 | Training GeoTransolver | `broadcast_global_features=False`, GALE | yes |
| 6 | Uncertainty Quantification | MC-Dropout & Concrete Dropout | Sections 5–6 only |

Notebooks 1 and 2 run anywhere with only NumPy, SciPy, scikit-learn and Matplotlib.

---

### Notebook 0 — Data Preprocessing (run once, shared by NB3 & NB5)

Converts the raw Ahmed body dataset into the Zarr format both training notebooks consume:

- Reads each case's inlet velocity from its companion `<case>_info.txt` file (velocity differs per simulation — it is not a global constant; the dataset spans 20–60 m/s).
- Non-dimensionalises the surface fields by the **kinematic** dynamic pressure $q = \tfrac12 U^2$. The VTP `p` and `wallShearStress` arrays are kinematic (m²/s²) in the OpenFOAM incompressible convention, so no density factor enters. Result: true $C_p$, peaking near $+1$ at stagnation.
- Saves `air_density` (a dataset constant) and per-case `stream_velocity` into every store — `TransolverDataPipe` builds `batch["fx"]` from exactly these two.
- Saves `stl_coordinates` (raw STL vertices) for GeoTransolver's GALE context in Notebook 5.
- Computes per-channel statistics with Welford's algorithm over the **training split only** and writes `surface_fields_normalization.npz`.

> **If you regenerate the data, checkpoints trained on the previous convention become invalid.** The per-case $q$ varies 9× across the velocity range, so the two target spaces are not related by a constant and z-scoring will not absorb the difference. Notebook 5 checks this and refuses to resume across the boundary.

### Notebook 1 — The "Why": The Quadratic Bottleneck

We start with the **attention mechanism** of the standard Transformer and its limitation: **quadratic cost $O(N^2 d_k)$**. Note the real barrier is arithmetic, not memory — FlashAttention-style kernels compute exact attention without materialising the $N \times N$ matrix, but the operation count remains. That is what Transolver removes.

### Notebook 2 — The "How": Physics-Attention Mechanics

A **NumPy blueprint** of Physics-Attention: **Slice, Aggregate, Attend, Deslice**. $N$ mesh points are grouped into $M \ll N$ physics-aware slice tokens, so the *attention matrix* is $M \times M$. The layer as a whole is $O(NMC)$ — linear in $N$, which is the point; the $O(M^2C)$ attention is not the dominant term.

The notebook also measures what this costs: the output has rank $\le M$, so Physics-Attention is a structured **low-rank approximation** of full attention, not an exact reformulation.

### Notebook 3 — The "Factory": Training Transolver

Production-grade implementation following the PhysicsNeMo `transformer_models/src` workflow. Notebook 0 must run first.

- **Data pipeline:** `TransolverDataPipe` + `CAEDataset` — centering, subsampling, normalization.
- **Model selection via YAML:** in production, `conf/model/transolver.yaml` selects the class through Hydra:
  ```yaml
  _target_: physicsnemo.models.transolver.Transolver
  functional_dim: 2   # air_density + stream_velocity — broadcast to all N points
  embedding_dim:  6   # mesh_centers(3) + normals(3)
  out_dim:        4   # Cp, Cf_x, Cf_y, Cf_z
  ```
  Transolver has no dedicated global-conditioning pathway, so `broadcast_global_features: true` copies the two flow scalars to every mesh point.
- **Training:** abbreviated loop mirroring the production `train.py`.
- **Analysis:** ascription weight maps and Shannon entropy — computed **per attention head** and with the model's **learned softmax temperature** applied, both of which materially change the result.

### Notebook 4 — Understanding GALE & GeoTransolver

We examine **GALE (Geometry-Aware Latent Embeddings)** — cross-attention that gives every transformer block access to a persistent geometry context, preventing representation drift in deep stacks.

- Why geometry context fades with depth
- How GALE works: self-attention and cross-attention both operate on the $M$ **slice tokens**, blended by a learned gate
- Section 5.7 reads the learned gate out of a trained checkpoint — a measurement rather than a claim

> Two things worth knowing up front: the gate is **one scalar per block**, so it varies with depth but not position; and the multi-scale ball-query features are concatenated into the hidden state **once at the input**, while it is the tokenized geometry/global context that reaches every block.

### Notebook 5 — Training GeoTransolver

Same pipeline classes as Notebook 3, differing in configuration:

| Flag | NB3 — Transolver | NB5 — GeoTransolver |
|------|-----------------|---------------------|
| `broadcast_global_features` | `True` — `batch["fx"].shape = (1, N, 2)` | `False` — `(1, 1, 2)` |
| `include_geometry` | `False` | `True` — adds `batch["geometry"]` `(1, M, 3)` |
| `scale_invariance` | not used | **`True`** with `reference_scale=[1.35, 0.26, 0.44]` |

> `scale_invariance` is **required**, not an enhancement. The ball-query `radii` are radii *in the scaled frame*, so omitting it makes every local-feature shell sample the wrong neighbourhood — and the checkpoint still loads without complaint.

In production, `conf/model/geotransolver.yaml` selects the model:
```yaml
_target_: physicsnemo.experimental.models.geotransolver.GeoTransolver
functional_dim: 6   # local_embedding: coords(3) + normals(3)
global_dim:     2   # global_embedding: air_density + stream_velocity
geometry_dim:   3   # STL vertex coordinates for GALE context
out_dim:        4
```

### Notebook 6 — Uncertainty Quantification

Adds per-point epistemic confidence to a trained model without retraining from scratch.

**MC-Dropout** — keep dropout active at inference, run $T$ stochastic passes, take the per-point mean and standard deviation. Backed by Gal & Ghahramani's result that dropout networks approximate Bayesian inference over weights.

**Concrete Dropout** — treat $p$ as learnable, with a KL-based regularizer added to the training loss. PhysicsNeMo supports this natively via `model.concrete_dropout=true` and `training.lambda_reg`.

The two **compose rather than compete**: Concrete Dropout supplies at training time the $p$ that MC-Dropout consumes at inference. Concrete Dropout makes no claim to improve accuracy — it removes $p$ as a hyperparameter you have to guess.

The notebook also covers calibration properly: correlation alone does not establish that σ is trustworthy, so it reports **z-RMS**, **coverage**, and a per-tercile breakdown distinguishing a scale error (a single factor fixes it) from a shape error (nothing scalar will).

> Notebook 6 keeps one deliberate **negative result**: on a 1-D toy problem MC-Dropout fails to widen σ across a data gap, at every dropout rate and both activations. Knowing where a method does not fire is more useful than a demo tuned to flatter it.

---

## Prerequisites

### 1. NVIDIA NGC Account

Required to pull the container: [NGC Docker Setup Guide](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html)

### 2. Ahmed Body Dataset

Download from NGC:
[physicsnemo_ahmed_body_dataset](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset)

After extracting, confirm the structure:

```
physicsnemo_ahmed_body_dataset_vv1/dataset/
├── train/                 (408 cases)   ├── validation/   (50)   ├── test/   (50)
├── train_info/                          ├── validation_info/     ├── test_info/
└── train_stl_files/                     └── validation_stl_files/└── test_stl_files/
```

Notebook 0 writes its Zarr output and `surface_fields_normalization.npz` under this tree by default. Notebooks 0, 3 and 6 accept a `ZARR_DIR` environment variable if you prefer to keep generated data elsewhere.

---

## Environment Setup

### Step 1 — Pull the PhysicsNeMo Container

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:26.06
```

> PhysicsNeMo **25.11 or newer** is required: earlier releases lack the experimental GeoTransolver namespace (NB4/5/6) and the `concrete_dropout` flag (NB6).

### Step 2 — Launch the Container

Replace `<path_on_host>` with the absolute path to the directory holding your notebooks and the dataset. It is mounted at `/workspace`.

```bash
docker run --gpus 1 --shm-size=2g -p 7008:7008 \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --runtime nvidia \
    -v <path_on_host>:/workspace \
    -it --rm \
    nvcr.io/nvidia/physicsnemo/physicsnemo:26.06
```

### Step 3 — Install Additional Dependencies (Inside Container)

```bash
# xvfb provides a virtual framebuffer for PyVista in headless environments
apt-get update && apt-get install -y rsync xvfb

# Python packages — see requirements.txt for the full pinned list
pip install -r requirements.txt
```

> Do not install `transformer_engine` unless you need it. Every notebook here sets `use_te=False`, and TE generally installs only inside NGC containers.

### Step 4 — Start Jupyter Lab

```bash
nohup python3 -m jupyter lab \
    --ip=0.0.0.0 --port=7008 --allow-root --no-browser \
    --NotebookApp.token='' \
    --notebook-dir='/workspace/' \
    --NotebookApp.allow_origin='*' \
    > /dev/null 2>&1 &
```

Start Jupyter from the directory containing the notebooks — several use relative paths (e.g. `checkpoints_geotransolver`) that resolve against the kernel's working directory.

### Step 5 — Access Jupyter Lab

**Remote host:** create an SSH tunnel from your local machine:

```bash
ssh -L 3030:<remote_hostname>:7008 <ssh_alias>
```

Then open `http://localhost:3030`.

**Local machine:** open `http://localhost:7008`.

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
├── requirements.txt                             ← Python dependencies
├── fig/                                         ← figures referenced in the notebooks
├── checkpoints_geotransolver/                   ← created by NB5
└── logs_geotransolver/                          ← created by NB5
```

---

## References

- Wu, H., et al. (2024). *Transolver: A Fast Transformer Solver for PDEs on General Geometries.* [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
- *GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer* (2025). [arXiv:2512.20399](https://arxiv.org/abs/2512.20399)
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation.* ICML 2016.
- Gal, Y., Hron, J., & Kendall, A. (2017). *Concrete Dropout.* NeurIPS 2017.
- NVIDIA PhysicsNeMo: [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)
- External aerodynamics recipe: [`examples/cfd/external_aerodynamics/transformer_models`](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/transformer_models)
