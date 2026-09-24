# End-to-End AI for Science — Crash Surrogate Modeling

This three-notebook series follows a real bumper-beam crash-surrogate workflow using
[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) and GeoTransolver. It connects
the native simulation data, model architecture, training process, validation evidence, and
full-field predictions.

All three notebooks are already executed and include their outputs. Notebook0 and Notebook1
provide the data and architecture background. Notebook2 contains the short executable workflow.

Notebook0 opens with the bumper-beam impact sequence before examining the corresponding mesh and
time-dependent fields.

## Notebook workflow

**Notebook0: data → Notebook1: architecture → Notebook2: two-epoch one-shot workflow**

The notebooks should be read in this order:

| Step | Notebook | Main question |
|---:|---|---|
| 1 | [Notebook0 — Native VTP Data Contract and Preprocessing](Notebook0_Crash-Data-Preprocessing.ipynb) | What is stored in each crash simulation, and what tensors enter the model? |
| 2 | [Notebook1 — GeoTransolver Architecture and Concepts](Notebook1_Crash-Architecture-and-Concepts.ipynb) | How does GeoTransolver process the mesh, conditioning values, and time? |
| 3 | [Notebook2 — Two-Epoch One-Shot Workflow](Notebook2_Crash-Training-Integration-Comparison.ipynb) | How do training, validation, checkpointing, and inference connect in the PhysicsNeMo crash recipe? |

## The case study

New bumper-beam simulation datasets can be generated with PhysicsNeMo's
[OpenRadioss dataset-generation recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/openradioss_dataset_gen),
using its
[bumper-beam flow](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/openradioss_dataset_gen/bumper_beam).
The flow creates `d3plot` runs and run-level metadata that PhysicsNeMo-Curator can convert to
training-ready VTP data. The default generator sweep
contains 135 design cases, while the supplied, curated notebook dataset contains 134 usable
simulations in its training and validation splits.

The supplied dataset is already curated as native **VTP (VTK XML PolyData)**. One VTP file
represents one complete simulation and stores the reference mesh, displacement history,
topology, effective plastic strain, and von-Mises stress. A JSON file stores the run-level
conditions.

| Item | Verified value |
|---|---:|
| Training simulations | 129 |
| Validation simulations | 5 |
| Points per mesh | 13,675 |
| Quad cells per mesh | 13,626 |
| Source states | 51, from 0.000 to 0.250 s |
| Prediction targets | 50, from 0.005 to 0.250 s |
| Time interval | 0.005 s |
| Global conditions | `velocity_x`, `thickness_scale`, `rwall_origin_y` |

The runnable Notebook2 demonstration predicts node position only, with target shape
`[N, 50, 3]`. Effective plastic strain and von-Mises stress remain available in the source
VTP files, but they are not prediction targets in this demonstration.

> The five files under `test/` are byte-identical to the five validation files. Results in
> this series are therefore validation results, not an independent test estimate.

## What each notebook contains

### Notebook0 — understand the data

[Open Notebook0](Notebook0_Crash-Data-Preprocessing.ipynb)

Notebook0 audits the supplied dataset without modifying it. It covers:

- split membership and `global_features.json` integrity;
- VTP reference coordinates, topology, and 51 synchronized field states;
- reconstruction of absolute positions from reference coordinates and displacement;
- cell-to-point conversion for plastic strain and stress;
- the PhysicsNeMo VTP reader and training datapipe;
- exact one-shot and time-conditioned tensor contracts;
- full-mesh ground-truth deformation and physical-field visualizations.

The schema audit, field-history plots, and ground-truth animation are embedded in the executed
notebook.

### Notebook1 — understand the architecture

[Open Notebook1](Notebook1_Crash-Architecture-and-Concepts.ipynb)

Notebook1 follows real bumper-beam tensors through the model. It covers:

- VTP trajectory to `SimSample` conversion;
- GeoTransolver input, context, and output shapes;
- Geometry-Aware Latent Embeddings (GALE);
- geometry and global-context cross-attention;
- one-shot, time-conditioned, and autoregressive temporal wrappers;
- position-only contracts derived from the public recipe configuration;
- embedded error figures and full-mesh animation from separate full recipe runs.

Notebook1 does not locate scheduler job directories, load checkpoints, or expose a live inference
switch. It is an architecture reference; model execution is confined to Notebook2's short
one-shot workflow.

### Notebook2 — run the two-epoch one-shot workflow

[Open Notebook2](Notebook2_Crash-Training-Integration-Comparison.ipynb)

Notebook2 provides an executable walkthrough of the one-shot PhysicsNeMo crash workflow. It
covers:

- the one-shot configuration supplied by the crash recipe;
- two epochs of full-mesh training on all supplied training simulations;
- validation on all five validation simulations after each epoch;
- TensorBoard metrics and the paired epoch-2 checkpoint;
- loading that checkpoint for full-mesh inference;
- VTP export and comparison with one validation trajectory.

The saved cell outputs demonstrate that training, validation, checkpointing, and inference
complete successfully. Two epochs are not sufficient for convergence or accuracy claims. For
longer, converged, distributed, or production training, run the
[PhysicsNeMo crash recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash)
directly rather than increasing the notebook workload.

## Three ways to represent time

All three methods use the same position target and full 13,675-point mesh.

| Method | Model call | Output contract | Main behavior |
|---|---|---|---|
| One-shot | One call per simulation | `[N, 50, 3]` | Predicts all future positions together; no predicted state is fed back. |
| Time-conditioned | One call per requested time | `[N, 3]` per query | Uses shared weights for 50 independent time queries. |
| Autoregressive | Repeated calls | `[N, 3]` per step | Predicts acceleration, updates the state, and feeds the prediction into the next step. |

The completed autoregressive configuration uses the experimental setting `initial_vel=0.0`.
Its measured result should not be treated as a general conclusion about every autoregressive
model.

## Embedded reference comparison

Notebook2 also contains figures and animations from separate full recipe runs. They compare
three methods across five validation simulations and 50 predicted frames. These are embedded
reference results; Notebook2 does not locate scheduler job directories or load their checkpoints.

| Method | Best validation epoch | Epoch-500 validation MSE | Trajectory position RMSE |
|---|---:|---:|---:|
| Time-conditioned | 215 | 2.894 × 10⁻⁴ | 2.996 |
| One-shot | 334 | 8.721 × 10⁻⁴ | 6.360 |
| Autoregressive | 464 | 4.714 × 10⁻³ | 32.592 |

Validation MSE is measured in normalized coordinate space. Position RMSE is measured from
the exported VTP coordinates in the dataset's native length units; the VTP metadata do not
declare a physical unit.

The reference section includes convergence curves, horizon error, per-run variation,
full-mesh animations, and spatial-error analysis.

## Required locations

| Purpose | Location |
|---|---|
| Project | A checkout containing the `physicsnemo/` source tree |
| Notebooks | This directory |
| Crash recipe | `physicsnemo/examples/structural_mechanics/crash` inside the checkout |
| Dataset | A VTP dataset root with `global_features.json`, `train/`, and `validation/` |
| Demonstration output | A writable directory selected by Notebook2 |

The notebooks search from Jupyter's current working directory. `CRASH_PROJECT_ROOT`,
`BUMPER_BEAM_DATA_ROOT`, `PHYSICSNEMO_CRASH_RECIPE`, and `NOTEBOOK2_DEMO_DIR` can override
these locations. No completed benchmark directory or scheduler-specific run ID is required.

Notebook0 and Notebook1 likewise require no historical run directory, scheduler job ID, or
completed checkpoint.

Expected dataset layout:

```text
bumper_beam/
├── global_features.json
├── train/          # 129 VTP files
├── validation/     # 5 VTP files
└── test/           # duplicate of validation
```

## Quick start

Use a PhysicsNeMo container with the project and dataset mounted, then start Jupyter from this
directory.

```bash
cd <path-to-checkout>/11-09-2026/pr_ready_crash_notebooks
jupyter lab --ip=0.0.0.0 --no-browser
```

Read Notebook0 and Notebook1 for the data and architecture background. Notebook2 is the only
notebook intended to execute a model workflow in this release.

- Notebook0 computes its data audit and train-split statistics without a GPU.
- Notebook1 inspects recipe source and tensor contracts without loading a model checkpoint.
- Notebook2 needs GPUs only when rerunning the two-epoch training and inference workflow.
- Saved outputs remain visible when live execution is disabled.

The PhysicsNeMo container supplies PyTorch, PhysicsNeMo, VTK/PyVista, Hydra, TensorBoard,
NumPy, pandas, Matplotlib, and Pillow. The notebooks do not install packages or clone another
copy of the recipe.

## Optional live Notebook2 demonstration

Live execution defaults to off so opening the notebook does not launch GPU work.

| Environment variable | Action |
|---|---|
| `RUN_NOTEBOOK2_DEMO=1` | Train the one-shot model for two epochs, validate after each epoch, save the epoch-2 checkpoint, and run validation inference from that checkpoint. |

For a fresh run, select a new writable output directory and set the switch before starting
Jupyter:

```bash
export NOTEBOOK2_DEMO_DIR="$PWD/notebook2_output"
export RUN_NOTEBOOK2_DEMO=1
jupyter lab --ip=0.0.0.0 --no-browser
```

Notebook2 writes only isolated demonstration artifacts. It does not require or modify completed
benchmark runs. Use the PhysicsNeMo crash recipe directly for real training.

## Interpretation limits

- The two-epoch run verifies the workflow only; its losses and predictions are not evidence of convergence or production accuracy.
- Training, validation, and inference use the full mesh; only rendering may subsample points.
- The position-only checkpoints do not establish stress or plastic-strain prediction accuracy.
- TensorBoard validation MSE and exported position RMSE use different scales and should not be
  compared numerically.
- `frame_000` is the first target at 0.005 s, and `frame_049` is the final target at 0.250 s.
- OOD warnings are review signals. They indicate input drift and are not estimates of prediction
  error.
- Final engineering decisions still require suitable CAE evidence, engineering review, and
  physical validation.

## Repository contents

```text
notebooks/
├── README.md
├── Notebook0_Crash-Data-Preprocessing.ipynb
├── Notebook1_Crash-Architecture-and-Concepts.ipynb
└── Notebook2_Crash-Training-Integration-Comparison.ipynb
```

All diagrams, figures, animations, and saved two-epoch demonstration outputs are contained in
the notebooks. No separate display assets are required.

## References

- [PhysicsNeMo crash recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/crash)
- [PhysicsNeMo crash README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/README.md)
- [PhysicsNeMo OpenRadioss dataset-generation recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/openradioss_dataset_gen)
- [PhysicsNeMo OpenRadioss bumper-beam flow](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics/openradioss_dataset_gen/bumper_beam)
- [PhysicsNeMo structural-mechanics blog](https://nvidia.github.io/physicsnemo/blog/2026/03/17/structural-mechanics/)
- [Automotive Crash Dynamics Modeling Accelerated with Machine Learning](https://arxiv.org/abs/2510.15201)
- [High-Fidelity Industrial Crash Dynamics Prediction via Geometry-Aware Operator Learning with Memory-Efficient Low-Rank Attention](https://arxiv.org/abs/2605.27758)
