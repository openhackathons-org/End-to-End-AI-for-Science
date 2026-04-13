# AI Surrogates for Reactor Physics

This repository provides the code that accompanies [this technical blog](). This work demonstrates a practical workflow for developers and engineers in the nuclear industry to build AI surrogates using [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/main) and integrate them into their design processes. We focus on a relatively simple pin cell example, where jointly predicting the neutron flux field and the absorption cross-section map - and then computing the homogenised cross-section - yields substantially higher accuracy than directly predicting the homogenised cross-section from a set of scalar descriptors.

| Notebook | Details |
| -------- | ------- |
| [1. Dataset generation](1-dataset-generation.ipynb) | Employ Latin hypercube sampling to generate a diverse dataset of reactor pin cells for model training. Sampled designs are simulated using [`OpenMC`](https://openmc.org/), and the corresponding neutron flux field and homogenised cross-section value is recorded as the output. |
| [2. Feature-based regression](2-feature-based-regression.ipynb) | Establish a **baseline predictive model**: a feature-based regression model - specifically, a gradient boosting regressor. The baseline model directly predicts the scalar homogenised cross-section from a set of scalar descriptors capturing the key geometric features and material parameters of a pin-cell. |
| [3. Fourier neural operator](3-fourier-neural-operator.ipynb) | Train and evaluate a Fourier Neural Operator (FNO) using [`physicsnemo`](https://github.com/NVIDIA/physicsnemo). Demonstrate improved performance over the baseline model through a two-step, physics-aligned approach: first predict the neutron flux field using the FNO, then compute the homogenised cross-section from the predicted flux. |

## Getting started

`OpenMC` must be installed in a `conda` environment; `environment.yml` defines the base project environment.

### 1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate ai-for-science
```

### 2. Install PhysicsNeMo

`nvidia-physicsnemo` is installed separately with `pip`:

```bash
pip install nvidia-physicsnemo
```

Separating the `pip` install helps reduce dependency conflicts.

If dependency issues persist, the recommended approach for running the [Fourier neural operator](3-fourier-neural-operator.ipynb) notebook is to use the [PhysicsNeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo?version=26.03).