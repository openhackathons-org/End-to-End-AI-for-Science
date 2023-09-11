# End-to-End AI for Science Deployment Guide 

The End-to-End AI for Science Bootcamp provides a step-by-step overview of the fundamentals of deep neural networks, walks attendees through the hands-on experience of building and improving deep learning models using a framework that uses the fundamental laws of physics to model the behavior of complex systems, and enables attendees to visualize the physically accurate outputs of the trained model in near real-time.

<p align="center">
  <img width="600" height="400" src="https://github.com/openhackathons-org/End-to-End-AI-for-Science/blob/d403086ce59c49b26be430bbea0056c37bd4d5f6/workspace/python/jupyter_notebook/omniverse/images/tcwv.gif">
</p>

## Deploying the materials

### Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU, a RTX enabled GPU is necessary for working with the NVIDIA Omniverse™.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

The material is also tested to be working with NVIDIA A100 and A800 GPUs, kindly contact us if you are looking for instructions to deploy it in A100/A800 environment.

### Tested environment

This materials was tested with both Docker and Singularity on an NVIDIA RTX 3080Ti GPU in an x86-64 platform installed with a driver version of `535.104.05`. 

### Deploying with container 

This material can be deployed with either Docker or Singularity container, refer to the respective sections for the instructions.

#### Docker Container

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

#### Singularity Container

To build the singularity container, run: 
`singularity build --fakeroot --sandbox End-to-End-AI-for-Science.sif Singularity`

Then, run the container:
`singularity run --writable --nv End-to-End-AI-for-Science.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/aswkumar99/End-to-End-AI-for-Science/issues).