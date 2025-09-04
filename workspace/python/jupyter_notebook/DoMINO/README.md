# PhysicsNeMo External Aerodynamics DLI

This Deep Learning Institute (DLI) course introduces participants to physics-informed machine learning techniques applied to external aerodynamics. Using NVIDIA's Physics NeMo and NeMo Framework, the course walks through the full pipeline — from preprocessing CFD data to training and deploying advanced geometric deep learning models.

## Introduction

Participants will work with the **Ahmed body** benchmark — a canonical problem in automotive aerodynamics — to build a surrogate model. The course blends classical CFD data handling with modern AI-based modeling. 
For the sake of education here we will use only **Ahmed Body Surface data** for traning and not volume data. 

## Quick Start Guide

### 1. Setting your environment to pull the PhysicsNeMo container from NGC:

Please refer to the following link for instructions on setting up your environment to pull the PhysicsNeMo container from NGC:
https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html


### Download Ahmed Body surface dataset

The complete Ahmed body surface dataset is hosted on NGC and accessible from the following link:

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset

Then unzip the data and copy them where you can acccess it from the cotainer,e.g. `/workspace/data/`

```bash
.
├── test
├── test_info
├── train
├── train_info
├── validation
└── validation_info
```

Please note that the dataset contains VTP files, but training DoMINO and X-MeshGraphNet also requires STL files. Therefore, in the `domino-data-preprocessing.ipynb` notebook, STL files are extracted from the available VTP data.

### Runing PhysicsNeMo container using dokcer command

Pull the PhysicsNeMo container with the following command:
```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.08
 ```

To launch the PhysicsNeMo container using docker, use the following command:

```bash
docker run --gpus 1  --shm-size=2g -p 7008:7008 --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -v <path_on_host>:/workspace -it --rm $(docker images | grep 25.08| awk '{print $3}')
 
```

Make sure to replace <path_on_host> with the absolute path to the directory on the host system that contains your Jupyter notebooks and Ahmed Body surface data. This path will be mounted as `/workspace` inside the container, providing access to your data and scripts during the session

### start Jupyter Lab

From the terminal inside the container run the following command to start Jupyter Lab in the background:

```bash
nohup python3 -m jupyter lab --ip=0.0.0.0 --port=7008 --allow-root  --no-browser --NotebookApp.token='' --notebook-dir='/workspace/' --NotebookApp.allow_origin='*' > /dev/null 2>&1 &
```

Then from your labtop start a SSH tunnel using the host which your job is runing and the port which you assigned above `--port=3030`: 

```bash
ssh -L 3030:eos0311:7008 eos
```
Access Jupyter Lab using `http://localhost:3030` in your browser. 

### Dependencies

Please install the following in the container.
```bash
pip install numpy pyvista vtk matplotlib tqdm numpy-stl
apt update
apt install  xvfb
```

### Working with Jupyter Notebooks

1. In Jupyter Lab, there are two notebooks
2. Each notebook contains:
   - Theory sections explaining the concepts
   - Code cells with detailed comments
   - Interactive visualizations
   - Progress checkpoints to verify your understanding

## Course Modules

### 1. [Preprocessing Ahmed body dataset for training with DoMINO](domino-data-preprocessing.ipynb)
- Loading and understanding VTK simulation data
- Data normalization techniques for ML
- Extracting key physical quantities
- Creating standardized datasets for training

### 2. [Training and Inferencing DoMINO Model](domino-training-test.ipynb)
- Understanding DoMINO architecture and physics-informed learning
- Training process
- Load Model Checkpoint & Run Inference
- Visualizing the predicted results

### Hardware Requirements
- NVIDIA GPU with CUDA support
- 4GB+ RAM
- 100GB+ disk space

### Software Requirements
- Docker with NVIDIA Container Toolkit
- NVIDIA Drivers (version 545 or later)
- SLURM workload manager

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Monitor GPU memory: `nvidia-smi`
   - Check system memory: `free -h`

2. **GPU Support**

   - Run the following command to verify your container runtime supports NVIDIA GPUs:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Expected output should show your GPU information, for example:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-PCIE-40GB          Off |   00000000:41:00.0 Off |                    0 |
| N/A   37C    P0             35W /  250W |   40423MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

## Additional Resources

- [NVIDIA Physics NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo-physics/user-guide)
- [PyVista Documentation](https://docs.pyvista.org/) for 3D visualization
- [Ahmed Body Benchmark](https://www.cfd-online.com/Wiki/Ahmed_body) for background
- [Physics-Informed Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) paper
- [Graph Neural Networks](https://arxiv.org/abs/2106.10943) for scientific computing
- [Neural Operators](https://arxiv.org/abs/2108.08481) for PDEs
- [DGL Documentation](https://www.dgl.ai/) for graph neural networks
- [Triton Inference Server](https://github.com/triton-inference-server/server) for deployment
