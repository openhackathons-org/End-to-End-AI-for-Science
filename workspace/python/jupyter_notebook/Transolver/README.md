# Transolver Tutorial: From Theory to Training

Welcome to this tutorial series!

In this series, we’ll go from Transformer theory to Transolver, a new approach for large-scale physics simulations. The goal is to build a complete, end-to-end understanding, from the fundamental concepts to a real, working training pipeline.

This series is broken into three notebooks, each building on the last:

* **Notebook 1 (The "Why"):** We will start by exploring the core "attention" mechanism of a standard Transformer. We will then discover its critical flaw: the "quadratic complexity" ($O(N^2)$). You will see why this "quadratic bottleneck" makes it "computationally prohibited" for the "large-scale meshes" used in real-world physics.

* **Notebook 2 (The "How"):** We'll build a `NumPy` "blueprint" of the Transolver solution: **Physics-Attention**. We will implement the 4-step mathematical process (Slice, Aggregate, Attend, Deslice) that breaks this bottleneck. This notebook will explain the *internal mechanics* of how Transolver learns "intrinsic physical states" by grouping mesh points into "learnable slices" and performing attention on a few "physics-aware tokens".

* **Notebook 3 (The "Factory"):** We'll move from blueprint to reality. We will use the official **Physics-NEMO** library and your provided Python scripts (`datapipe.py`, `preprocess.py`, etc.) to build a complete, professional-grade training pipeline. We will preprocess your 508 VTP/STL files into the Zarr format, calculate the necessary normalization statistics, and run an (abbreviated) training loop for a few epochs to see it all work.

---

## 1. Prerequisites

Before you begin, you will need two things:

1.  **NVIDIA NGC Account:** You must have an account to pull the required Docker container. You can find setup instructions here:
    * [NVIDIA NGC Docker Setup Guide](https://docs.nvidia.com/launchpad/ai/base-command-coe/latest/bc-coe-docker-basics-step-02.html)

2.  **Ahmed Body Dataset:** You must download the "Ahmed body surface dataset" from NGC.
    * **Link:** [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_ahmed_body_dataset)
    * Download and extract this dataset. We will mount it into our container.
    * Then navigate to the directory `/workspace/physicsnemo_ahmed_body_dataset_vv1/dataset` and confirm that the data has been downloaded successfully.

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

---

## 2. Environment Setup

This guide will walk you through setting up the exact environment used for this project.

### Runing PhysicsNeMo container using dokcer command

Pull the PhysicsNeMo container with the following command:
```bash
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.11
 ```

To launch the PhysicsNeMo container using docker, use the following command:

```bash
docker run --gpus 1  --shm-size=2g -p 7008:7008 --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -v <path_on_host>:/workspace -it --rm $(docker images | grep 25.08| awk '{print $3}')
 
```

Make sure to replace <path_on_host> with the absolute path to the directory on the host system that contains your Jupyter notebooks and Ahmed Body surface data. This path will be mounted as `/workspace` inside the container, providing access to your data and scripts during the session

### Install Dependencies (Inside Container)
```bash
# Install system dependencies
# (xvfb is a virtual screen buffer required for PyVista in a headless server)
apt update
apt install rsync xvfb -y

# Install required Python packages
pip install hydra-core tabulate tensorboard termcolor torchinfo einops transformer_engine[pytorch] zarr>=3.0 zarrs

```
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








