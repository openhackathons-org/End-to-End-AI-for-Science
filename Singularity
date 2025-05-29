# Copyright (c) 2024 NVIDIA Corporation.  All rights reserved.
# To build this : $ singularity build --fakeroot --sandbox End-to-End-AI-for-Science.sif Singularity
# To Run this : $ singularity run --writable --nv End-to-End-AI-for-Science.sif jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python

Bootstrap: docker
FROM: nvcr.io/nvidia/physicsnemo/physicsnemo:25.03

%environment
%post
    pip3 install gdown ipympl cdsapi
    pip3 install --upgrade nbconvert
    python3 /workspace/python/source_code/dataset.py    
    python3 /workspace/python/source_code/fourcastnet/decompress.py
    rm -rf /workspace/python/source_code/fourcastnet/pre_data
    
    apt update && apt install ffmpeg -y
    pip install --no-cache-dir --no-deps -e git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0#egg=makani  
    pip install --no-cache-dir "earth2studio==0.5.0"
    pip install --no-cache-dir cartopy mlflow
    
%files
    workspace/* /workspace/

%runscript
    "$@"

%labels
    AUTHOR aswkumar
