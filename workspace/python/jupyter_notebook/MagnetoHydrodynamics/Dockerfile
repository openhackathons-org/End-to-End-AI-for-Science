FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y apt-file \
                        vim \
                        wget \
                        git \
                        software-properties-common \
                        make \
                        g++ \
                        gcc \
                        gpg-agent

# create a user
RUN groupadd -g 1111 mhd && \
    useradd -m -u 1111 -g mhd mhduser

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

# set up a conda env (only used when generating data)
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda create -n env python=3.12

# put conda in path so we can use conda activate
ENV PATH=/opt/conda/envs/env/bin:$PATH

# these env variables are recommended by dedalus package 
RUN conda run -n env conda env config vars set OMP_NUM_THREADS=1
RUN conda run -n env conda env config vars set NUMEXPR_MAX_THREADS=1
# RUN conda install -c conda-forge dedalus
RUN conda run -n env conda install -c conda-forge dedalus

# install nvidia-physicsnemo
RUN pip install nvidia-physicsnemo

# copy requirements and install
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# install nvidia-physicsnemo sym
RUN pip install nvidia-physicsnemo.sym --no-build-isolation

WORKDIR /workspace
USER mhduser
