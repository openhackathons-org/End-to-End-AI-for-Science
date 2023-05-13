# Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t openhackathons:ai-for-science .
# To run: $ sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/workspace/python/source_code/extension:/workspace/python/source_code/extension -p 8011:8011  -p 8888:8888 -p 8899:8899 -it --rm openhackathons:ai-for-science
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/modulus/modulus:22.09

# Install required python packages
RUN pip3 install gdown ipympl cdsapi
RUN pip3 install wandb ruamel.yaml netCDF4 mpi4py 
RUN pip3 install --upgrade nbconvert

# TO COPY the data 
COPY workspace/ /workspace/

# This Installs All the Dataset
RUN python3 /workspace/python/source_code/dataset.py

# This decompresses the Dataset for usage 
RUN python3 /workspace/python/source_code/fourcastnet/decompress.py

# Remove Compressed files
RUN rm -rf /workspace/python/source_code/fourcastnet/pre_data

###### Install Omniverse 
# Install dependencies
RUN apt update
RUN apt install -y gdb unzip libatomic1 ca-certificates libglu1-mesa libsm6 libegl1 libgomp1 python3 gcc g++ make binutils libxrandr-dev ttf-ubuntu-font-family
RUN apt install -y libnvidia-gl-525
RUN apt install -y ffmpeg

# Install Omniverse
RUN cd /workspace/python/source_code/omniverse && bash get-kit.sh
RUN chmod +x /workspace/python/source_code/omniverse/*.sh

## Uncomment this line to run Jupyter notebook by default
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python/
