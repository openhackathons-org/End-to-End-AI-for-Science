# Transolver and GeoTransolver: from attention to surface prediction

Learn how attention models predict pressure and wall shear over a three-dimensional body using NVIDIA PhysicsNeMo. This series connects the ideas behind Transformers, Transolver and GeoTransolver to an Ahmed-body CFD workflow, then introduces uncertainty quantification (UQ).

The notebooks combine small examples you can run with saved training and prediction examples you can study. Explanations sit beside the code, figures and tables. Start with the concepts, follow the data through a model, and learn how to interpret its predictions and uncertainty intervals.

## What you will learn

- Prepare surface meshes, geometry and case information for a neural model.
- Understand why ordinary attention becomes expensive on large meshes.
- Follow Transolver's learned grouping of surface points into a smaller set of tokens.
- Explore how GeoTransolver uses a separate geometry context throughout its layers.
- Read training curves, compare predicted fields with CFD, and calculate drag from surface fields.
- Distinguish prediction accuracy from uncertainty, and understand interval calibration.

The detailed experiment results belong inside the notebooks, next to the explanations needed to interpret them. This README explains the learning path and how to get started.

## Series Overview

| Notebook | What happens in the lesson | What you need to run it |
|---|---|---|
| [0 — Data Preprocessing](Notebook0-Data-Preprocessing.ipynb) | Match VTP surfaces, STL geometry and info files; read each case's inlet speed; understand geometry scaling and field normalization. | Read the saved walkthrough directly. Supply raw CFD files to execute preprocessing. |
| [1 — Transformers and the Bottleneck](Notebook1-Understanding-Transformers-and-Bottleneck.ipynb) | Build attention from a small example, examine its cost, and connect it to a surface mesh. | The illustrative examples run without CFD data. A raw VTP is optional. |
| [2 — Understanding Transolver](Notebook2-Understanding-Transolver.ipynb) | Follow Slice → Aggregate → Attend → Deslice, with arrays and diagrams showing each step. | NumPy-based examples; no trained model or CFD dataset. |
| [3 — Training Transolver](Notebook3-Training-Transolver.ipynb) | Prepare model inputs, follow training and checkpoint restoration, compare fields and drag, and inspect learned point groupings. | Saved examples are included. Live CFD cells require the matching external data and experiment support. |
| [4 — Understanding GALE and GeoTransolver](Notebook4-Understanding-GALE-GeoTransolver.ipynb) | Use small examples to understand the geometry-context path, then inspect learned mixing coefficients. | PyTorch for the small examples. Matching external model files for checkpoint inspection. |
| [5 — Training GeoTransolver](Notebook5-Training-GeoTransolver.ipynb) | Connect geometry inputs to the trained model, surface predictions, drag and GALE routing. | Saved examples are included. Live CFD cells require CUDA and the matching experiment files. |
| [6 — Uncertainty Quantification](Notebook6-Uncertainty-Quantification.ipynb) | Repeat predictions, calibrate interval widths, and interpret pressure and whole-body drag uncertainty. | The small regression example runs without CFD data. Ahmed-body calculations require the matching experiment files. |

For the full story, read in numbered order. You can also begin with Notebooks 1 and 2 to learn attention, then return to Notebook 0 before the CFD exercises. Notebook 6 builds on the prediction and drag explanations in Notebooks 3 and 5.

## Prerequisites

### Background knowledge

Basic Python, NumPy arrays and neural-network training are helpful. You do not need to know Transolver, GALE or uncertainty calibration in advance: the notebooks introduce these terms when they first appear.

Some CFD vocabulary is useful. A **surface field** assigns values to locations on a body. **Pressure** and **wall shear** describe the surface loading, while **drag** combines their contributions along the flow direction. Notebook 0 explains the stored arrays and their relationship to the geometry.

### Computing environment

Choose the level of execution you need:

| Activity | Software and hardware |
|---|---|
| Read the lessons and saved figures | A browser or notebook viewer, with the accompanying `fig/` folder available. |
| Run the small conceptual examples | Python, the notebook requirements, and compatible PyTorch for the lessons that use it. These examples can run on a CPU. |
| Run the complete CFD exercises | A compatible PhysicsNeMo environment, CUDA GPU access, the dataset, and the matching experiment files described below. |

The environment used for the recorded workflow is **Python 3.12 with the NVIDIA PhysicsNeMo 26.08 container**. Using this image gives the clearest starting point for the full exercises. The [PhysicsNeMo installation guide](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html) explains supported installation options.

For Docker on an NVIDIA GPU machine, install Docker Engine, a compatible NVIDIA driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On a managed cluster, use its approved container and GPU workflow instead.

### NVIDIA NGC access

The container and Ahmed-body dataset are distributed through NVIDIA NGC. Follow the [NGC User Guide](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html) for account and API-key setup when authentication is required.

If the container registry requests credentials, run this on the machine that will pull the image:

```bash
docker login nvcr.io --username '$oauthtoken'
```

The username is the literal text `$oauthtoken`; the single quotes keep the shell from interpreting it as a variable. Paste your NGC API key at the password prompt. Then continue with the container setup below.

### Ahmed-body dataset

Download the [Ahmed-body dataset from NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset) when you are ready to execute the CFD data exercises. Keep the extracted files together:

```text
dataset/
├── train/                  Surface VTP files
├── train_info/             Matching case information
├── train_stl_files/        Matching STL geometry
├── validation/
├── validation_info/
├── validation_stl_files/
├── test/
├── test_info/
└── test_stl_files/
```

Match companions by case name. A surface such as `case894.vtp` needs its corresponding STL and info file. The directory labels locate source files; the notebooks explain how cases are assigned roles in the recorded experiment.

The recorded models use each case's inlet velocity as their scalar flow input. Shape enters through surface positions and normals, with an additional STL geometry context for GeoTransolver. Notebook 0 also explains how a shared coordinate scale preserves size differences between bodies, and why field-normalization statistics are fitted on the training data.

The extracted archive may have an extra enclosing directory. Point the notebook at the inner `dataset/` directory containing these source folders.

### Files for the trained-model exercises

The saved figures can be read with this tutorial folder alone. To execute the recorded model workflow, you additionally need the **matching experiment bundle**: model checkpoints, field normalization, geometry reference, experiment support code, and associated evaluation and calibration records.

These files are external to this teaching folder. Use the bundle and mount layout supplied by your instructor or experiment owner. The helpers read its records automatically; students following the normal learning route do not need to open or edit those records.

Notebook 0 also includes a recorded comparison of metadata with measured geometry. Recomputing that audit needs the instructor-provided geometry-audit files. With only the raw dataset, you can execute the conversion and study the supplied audit figure; the audit cell is separate from the conversion.

**A checkpoint must use its matching preprocessing.** Notebook 0's conversion exercise writes teaching data with its stated field scaling. The saved model examples use their own native-field stores and normalization. Notebook 0 explains the difference. Completing that conversion alone does not recreate the saved model experiment.

## Repository Structure

```text
Transolver/
├── README.md
├── requirements.txt
├── utils.py
├── Notebook0-Data-Preprocessing.ipynb
├── Notebook1-Understanding-Transformers-and-Bottleneck.ipynb
├── Notebook2-Understanding-Transolver.ipynb
├── Notebook3-Training-Transolver.ipynb
├── Notebook4-Understanding-GALE-GeoTransolver.ipynb
├── Notebook5-Training-GeoTransolver.ipynb
├── Notebook6-Uncertainty-Quantification.ipynb
└── fig/                    Saved figures and teaching diagrams
```

`utils.py` contains shared plotting and data helpers, keeping the notebook cells focused on the lesson. **Copy or publish `fig/` together with the notebooks.** Their image links are relative to this folder, so the same layout is needed in GitLab, GitHub and local Jupyter sessions.

## Environment Setup

### Step 1 — Organize a working directory

Choose an absolute path on the host for your learning workspace. Keep the tutorial folder inside it. If you download data or receive experiment files, place them in the same workspace or mount their locations separately.

```text
learning_workspace/
├── Transolver/             This tutorial folder
├── data/                   Downloaded CFD data, when needed
├── experiment/             Matching experiment bundle, when supplied
├── geometry_audit/         Geometry-audit files, when supplied
├── tutorial_data/          Your generated preprocessing output
└── notebook_outputs/       Your new figures and other outputs
```

The data, experiment and output directories are examples of locations you can use; they are not additional files you must download to read the lessons.

### Step 2 — Pull the PhysicsNeMo container

On the host machine:

```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:26.08
```

Keep the image tag consistent across setup and execution. This guide uses the recorded workflow's environment rather than assuming that every later release has the same interfaces.

### Step 3 — Launch the container

Replace the example host path, then run:

```bash
TRANSOLVER_WORKSPACE="/absolute/path/to/learning_workspace"

docker run --name transolver-tutorial --rm -it --gpus all --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 \
  -v "${TRANSOLVER_WORKSPACE}:/workspace" \
  -w /workspace/Transolver \
  nvcr.io/nvidia/physicsnemo/physicsnemo:26.08 bash
```

The mount makes your host workspace available as `/workspace` inside the container. The working-directory option opens the shell beside the notebooks, `utils.py` and `fig/`. The GPU option exposes NVIDIA GPUs, and the port mapping connects the host's local Jupyter port to the container.

Files written under the mounted workspace remain on the host after the container exits. Packages installed only inside this disposable container must be installed again when you start a fresh container.

### Step 4 — Install the notebook dependencies

Run the following **inside the container**, from `/workspace/Transolver`:

```bash
python -m pip install -r requirements.txt
```

The image already provides the PyTorch and PhysicsNeMo stack. The requirements file adds the tutorial's notebook, array, storage and plotting dependencies. It deliberately does not replace PyTorch or PhysicsNeMo. The notebook configurations also disable Transformer Engine; installing it separately is not part of this setup.

### Step 5 — Check the environment and register a kernel

Run this short check inside the container:

```bash
python - <<'PY'
from pathlib import Path
import sys
import numpy
import torch
import physicsnemo

print("Python:", sys.version.split()[0])
print("Python executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("PhysicsNeMo:", getattr(physicsnemo, "__version__", "imported"))
print("CUDA available:", torch.cuda.is_available())
print("Working directory:", Path.cwd())
print("utils.py found:", Path("utils.py").is_file())
print("fig/ found:", Path("fig").is_dir())
PY
```

The last two checks should be `True`. CUDA should be available for the full GPU exercises. A CPU environment can still run the small examples.

Give this Python environment a recognizable kernel name:

```bash
python -m ipykernel install --user \
  --name transolver-tutorial \
  --display-name "Python (Transolver tutorial)"
```

### Step 6 — Start JupyterLab

From the tutorial folder inside the container:

```bash
python -m jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root
```

Keep this terminal open. Jupyter prints an address containing a login token. The `--allow-root` option supports containers that run as root; a normal local user does not need it. Jupyter's default token authentication remains enabled.

If you need to find the address again, open another terminal on the Docker host and run:

```bash
docker exec transolver-tutorial jupyter server list
```

This queries the named container created above. In a non-Docker environment, run `jupyter server list` in the environment hosting Jupyter. See the [Jupyter Server authentication guide](https://jupyter-server.readthedocs.io/en/latest/operators/security.html) for details.

### Step 7 — Open Jupyter locally or remotely

**When Docker runs on your own machine:** open `http://127.0.0.1:8888` and use the token printed by Jupyter.

**When Docker runs on a directly reachable remote machine:** open a terminal on your laptop and create a tunnel to that machine:

```bash
ssh -N -L 8888:127.0.0.1:8888 username@remote-host
```

Replace the username and host, leave the tunnel running, then open `http://127.0.0.1:8888` on your laptop and enter the remote Jupyter token. Here the browser's local address reaches the remote server through the tunnel.

On a cluster with separate login and compute nodes, follow the institution's approved container allocation and Jupyter forwarding instructions. The tunnel must reach the machine running Jupyter; a generic connection to the login node is not sufficient in every cluster layout.

### Step 8 — Select the kernel and run a lesson

Open a notebook and select **Python (Transolver tutorial)**. Run cells from the top down. Use **Restart Kernel and Run All Cells** when beginning a lesson again, so it does not depend on variables left over from an earlier run.

Leave `RUN_WITH_DATA = False` for the first pass. This runs the small examples and skips code that requires external CFD data or trained models. The saved Ahmed-body figures and tables remain visible. Notebooks focused on the recorded CFD workflow can therefore be studied before enabling their data cells.

## Local Python Alternative for the Small Examples

If you want to explore the conceptual examples without Docker, start in the tutorial folder and create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install PyTorch using the command selected for your operating system and CPU or CUDA platform in the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the notebook dependencies and register the environment:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name transolver-local \
  --display-name "Python (Transolver local)"
python -m jupyter lab
```

Select **Python (Transolver local)** in Jupyter. Keep `RUN_WITH_DATA = False`. This route is for the small examples and saved lessons; use the matching PhysicsNeMo environment and experiment setup for the full CFD exercises.

## Enable the CFD Data Exercises

Configure paths **inside the environment that runs Jupyter**, using paths visible inside the container. Set environment variables before starting Jupyter so its kernels inherit them. If you change configuration in a notebook, restart and rerun its setup cells in order.

If Jupyter is already running when you change shell exports, stop and restart the server from that configured terminal. A new kernel launched by an older server inherits the older server's environment.

### Preprocessing your dataset

The following example matches the workspace layout above:

```bash
export AHMED_DATA_DIR=/workspace/data/ahmed_body/dataset
export ZARR_DIR=/workspace/tutorial_data/ahmed_body_zarr
export NORM_FILE="${ZARR_DIR}/surface_fields_normalization.npz"
export NOTEBOOK_OUTPUT_DIR=/workspace/notebook_outputs
```

| Setting | Purpose |
|---|---|
| `AHMED_DATA_DIR` | Raw dataset root containing the source split directories and their companion files. |
| `ZARR_DIR` | Separate destination for the teaching conversion's Zarr stores. |
| `NORM_FILE` | Normalization file loaded by Notebook 3's scratch lesson. Notebook 0 writes its file under `ZARR_DIR`. |
| `PREPROCESS_MAX_CASES` | Optional limit for a short conversion exercise; zero processes all available cases. |
| `AHMED_FIELD_UNITS` | `kinematic` or `Pa`, as explained in Notebook 0. Confirm the CFD export convention; `kinematic` is the notebook's default assumption. |
| `NOTEBOOK_OUTPUT_DIR` | Destination used by the model/UQ helpers. Some conceptual plots also save files in the current working directory. |
| `UQ_CURATION_REPORT_DIR` | Directory of instructor-provided geometry-audit files, needed to recompute Notebook 0's metadata-audit cell. |

If you have the geometry-audit files, set their location before starting Jupyter:

```bash
export UQ_CURATION_REPORT_DIR=/workspace/geometry_audit
```

Open Notebook 0 and change its setup cell to `RUN_WITH_DATA = True`. With both raw data and the audit files configured, run from the beginning. With raw data alone, run the setup and conversion cells but skip the cell calling `plot_metadata_curation_audit()` under “Compare the geometry descriptions with the meshes”; study its saved figure instead. The audit files are not part of the raw NGC dataset.

Notebook 0 writes `surface_fields_normalization.npz` inside `ZARR_DIR`. Check the source paths and field convention before converting data. Follow its explanation of inlet velocity, coordinate scaling and field normalization as separate steps.

### Using the matching trained experiment

After obtaining the complete compatible experiment bundle, configure its location. For a bundle placed at `/workspace/experiment`, the relevant settings are:

```bash
export FULL_DATA_UQ_ROOT=/workspace/experiment
export VELOCITY_NOTEBOOK_SUPPORT="${FULL_DATA_UQ_ROOT}/notebook_support.py"
export SURFACE_GEOMETRY_REFERENCE="${FULL_DATA_UQ_ROOT}/geometry_reference.json"
export UQ_ONLY_RESULTS_DIR="${FULL_DATA_UQ_ROOT}/reports"
export AHMED_DATASET_ROOT=/workspace/data/ahmed_body/dataset
```

| Setting | Purpose |
|---|---|
| `FULL_DATA_UQ_ROOT` | Completed experiment directory, including its matching training and evaluation artifacts. |
| `VELOCITY_NOTEBOOK_SUPPORT` | The experiment's support code for loading the models and running the teaching continuations. |
| `SURFACE_GEOMETRY_REFERENCE` | The same shared coordinate reference used to train the checkpoints. |
| `UQ_ONLY_RESULTS_DIR` | The completed experiment's pressure and drag uncertainty reports. |
| `AHMED_DATASET_ROOT` | Original CFD surfaces used for mesh-based plots and integration. |

Use the bundle's documented source and data mount layout as well as these paths. These settings locate an existing experiment; they do not generate missing checkpoints, support code or evaluation files.

With the matching files available, enable `RUN_WITH_DATA` in the notebook you are using and rerun from its first cell. Notebook 3's compact scratch option also uses the experiment support code and geometry reference in this edition. Notebook 5's trained-model route requires CUDA and the matching completed GeoTransolver experiment.

The notebooks explain which files each step reads and what it produces. Their separate teaching continuations illustrate training updates while the saved prediction examples remain tied to their documented model state.

## Optional: Rendering New CFD Surfaces on a Remote Server

The supplied figures are ordinary PNG images; reading them needs no graphics backend. Creating new PyVista/VTK surface renders on a server needs an available offscreen OpenGL backend.

Start with:

```bash
export PYVISTA_OFF_SCREEN=true
```

If rendering still fails, use the graphics setup supported by your container or cluster. The [PyVista installation guide](https://docs.pyvista.org/getting-started/installation.html#running-on-remote-servers) describes remote-server dependencies. Some environments use EGL; others use a Mesa/OSMesa software backend. For an environment already configured with OSMesa, the recorded rendering setup used:

```bash
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow
export LIBGL_ALWAYS_SOFTWARE=1
```

Those backend settings require their corresponding runtime libraries. Use the [VTK backend documentation](https://docs.vtk.org/en/v9.5.2/advanced/runtime_settings.html) or your instructor's environment setup to select the backend available on your machine.

## Troubleshooting

| What you see | What to check |
|---|---|
| Figures are missing in GitLab or GitHub | Publish the entire `fig/` folder next to the notebooks. Check that filenames and capitalization match the image links. |
| `ModuleNotFoundError: utils` | Start Jupyter in the tutorial folder and check `Path.cwd()` in the kernel. `utils.py` must be beside the notebook. |
| A package installs but the notebook cannot import it | Check the selected kernel and `sys.executable`; install dependencies using that environment's `python -m pip`. |
| PyTorch or PhysicsNeMo cannot be imported | Check whether you are inside the intended container. For the local route, install platform-compatible PyTorch before the requirements. |
| CUDA is unavailable | CPU is sufficient for the small examples. For the full GPU exercises, check the GPU allocation and container GPU access. |
| `RUN_WITH_DATA = True` reports missing support code or experiment files | Check the matching experiment bundle and its container-visible paths. Use the default setting while studying the saved lesson. |
| Notebook 0 reports missing curation/audit files | Configure `UQ_CURATION_REPORT_DIR` with the supplied audit files, or skip that audit cell while running the conversion with raw data alone. |
| Dataset files cannot be found | Point `AHMED_DATA_DIR` at the extracted `dataset/` root, then check matching VTP, STL and info filenames. |
| Jupyter asks for a token | Use the token printed at startup, or recover the server address with `jupyter server list`. |
| New surface rendering fails | Follow the optional offscreen-rendering section. The supplied PNG figures can still be read directly. |
| A cell depends on an undefined variable | Restart the kernel and run the setup and calculation cells from the top. |

## Further Reading

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)
- [GeoTransolver: Learning Physics on Irregular Domains Using Multi-scale Geometry Aware Physics Attention Transformer](https://arxiv.org/abs/2512.20399)
- [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) and [Concrete Dropout](https://arxiv.org/abs/1705.07832)
- [PhysicsNeMo transformer example at the source revision used for these lessons](https://github.com/NVIDIA/physicsnemo/tree/94b355108cfd10e253d3d41cacdc0c0112b078e4/examples/cfd/external_aerodynamics/transformer_models)
- [PhysicsNeMo installation guide](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html)
