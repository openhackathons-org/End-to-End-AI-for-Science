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

Then navigate to the directory `/workspace/physicsnemo_ahmed_body_dataset_vv1/dataset` and confirm that the data has been downloaded successfully.

```bash
physicsnemo_ahmed_body_dataset_vv1/dataset
├── train/
├── train_info/
├── train_stl_files/
├── validation/
├── validation_info/
├── validation_stl_files/
├── test/
├── test_info/
├── test_stl_files/

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

### Start Jupyter Lab


To launch Jupyter Lab inside the container in the background, run the following command from the terminal inside of the container


```bash
nohup python3 -m jupyter lab --ip=0.0.0.0 --port=7008 --allow-root  --no-browser --NotebookApp.token='' --notebook-dir='/workspace/' --NotebookApp.allow_origin='*' > /dev/null 2>&1 &
```
### Accessing Jupyter Lab
- If you're running the container on a remote host:
    You need to set up SSH tunneling to access the Jupyter Lab interface from your local machine.
    Assuming your container is running on host eos0311 and you used `--port=7008`, run the following command from your local laptop:
    ```bash
    ssh -L 3030:eos0311:7008 eos
    ```
    This creates a tunnel from your local port 3030 to port 7008 on the remote host.
    After establishing the tunnel, open your browser and go to `http://localhost:3030` to access Jupyter Lab.

- If the container is running on your local machine:
    You can access Jupyter Lab directly in your browser without SSH tunneling uisng `http://localhost:7008`

### Cloning the Repository Containing the Notebooks

At this point, you've launched the physicsNeMo container and started Jupyter Lab. Next, you need to clone the **End-to-End AI for Science** repository to access the notebooks.

From the terminal inside the container, run the following command:

```bash
git clone https://github.com/openhackathons-org/End-to-End-AI-for-Science.git
```
Once cloned, navigate to the following directory to access the notebooks:

```bash
End-to-End-AI-for-Science/workspace/python/jupyter_notebook/DoMINO
```
You can now open the notebooks in Jupyter Lab and start exploring.

### Dependencies

Please install the following in the container.
```bash
pip install numpy pyvista vtk matplotlib tqdm numpy-stl
apt update
apt install  xvfb
```

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

- [NVIDIA Physics NeMo Documentation](https://docs.nvidia.com/physicsnemo/index.html)
- [PyVista Documentation](https://docs.pyvista.org/) for 3D visualization
- [Ahmed Body Benchmark](https://www.cfd-online.com/Wiki/Ahmed_body) for background
- [Neural Operators](https://arxiv.org/abs/2108.08481) for PDEs
- [DoMINO Paper](https://arxiv.org/abs/2501.13350) for DoMINO paper
- [DGL Documentation](https://www.dgl.ai/) for graph neural networks
- [Triton Inference Server](https://github.com/triton-inference-server/server) for deployment
