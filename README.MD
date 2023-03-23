# End to End AI for Science

This repository contains materials for End to End AI for Science bootcamp. The objective of this lab is to give an introduction to application of Artificial Intelligence (AI) algorithms in Science ( High Performance Computing (HPC) Simulations ). This Bootcamp will provide introduction to fundamentals of using Physics Informed Neural Network and how they can be applied to real world scientific domains using NVIDIA Modulus and uses NVIDIA Omniverse for visualisation.

## Target Audience:

The target audience for this bootcamp are researchers/graduate students and developers who are new to field of Artifical Intelligence and interested in learning difference between Data and Physics Informed Neural Network approach applied to Simulation domains. Basic Python programming knowledge is required. 

## Tutorial Duration

The overall bootcamp will take approximately 12 hours. 

## Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container

To build a docker container, run: 
`sudo docker build -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t openhackathons:ai-for-science .`

and to run the container, run:
`sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8011:8011  -p 8888:8888 -p 8899:8899 -v ${PWD}/workspace/python/source_code/extension:/workspace/python/source_code/extension -it --rm openhackathons:ai-for-science`

The container launches jupyter lab and runs on port 8888
`jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Singularity Container

To build the singularity container, run: 
`singularity build --fakeroot --sandbox End-to-End-AI-for-Science.sif Singularity`

Then, run the container:
`singularity run --writable --nv End-to-End-AI-for-Science.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

## Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/aswkumar99/End-to-End-AI-for-Science/issues).
