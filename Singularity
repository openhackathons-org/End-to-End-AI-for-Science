# Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
# To build this : $ singularity build --fakeroot --sandbox End-to-End-AI-for-Science.sif Singularity
# To Run this : $ singularity run --writable --nv End-to-End-AI-for-Science.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python

Bootstrap: docker
FROM: nvcr.io/nvidia/modulus/modulus:22.09

%environment
%post
    pip3 install gdown
    pip3 install wandb ruamel.yaml netCDF4 mpi4py
    pip3 install --upgrade nbconvert
    python3 /workspace/python/source_code/dataset.py    
    python3 /workspace/python/source_code/fourcastnet/decompress.py
    rm -rf /workspace/python/source_code/fourcastnet/pre_data
    
    apt update
    apt install -y gdb unzip libatomic1 ca-certificates libglu1-mesa libsm6 libegl1 libgomp1 python3 gcc g++ make binutils libxrandr-dev ttf-ubuntu-font-family
    apt install -y libnvidia-gl-525
    apt install -y ffmpeg
    bash /workspace/python/source_code/omniverse/get-kit.sh
    chmod +x /workspace/python/source_code/omniverse/*.sh
    
%files
    workspace/* /workspace/

%runscript
    "$@"

%labels
    AUTHOR aswkumar
