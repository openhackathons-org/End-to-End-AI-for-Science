# End-to-End AI for Science Deployment Guide 

This Bootcamp will provide researchers hands-on approaches on how to use NVIDIA PhysicsNeMo, a framework that combines physics and partial differential equations (PDEs) with artificial intelligence (AI) to build robust models. Participants will also learn about the differences between Physics-driven and Data-driven approaches to AI. In addition, the Bootcamp will provide hands-on experience with visualizing the results of physics simulations using ParaView. This Lab will also introduce you to Earth2Studio where you get to try out different workflows for various Weather forecasting models!

<p align="center">
  <img width="600" height="400" src="https://github.com/openhackathons-org/End-to-End-AI-for-Science/blob/d403086ce59c49b26be430bbea0056c37bd4d5f6/workspace/python/jupyter_notebook/omniverse/images/tcwv.gif">
</p>

## Deploying the materials

### Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU with atleast 20GB of GPU memory for training large models as part of this Bootcamp.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

- Internet is needed during Container building and runtime to download and setup the required files.

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
`sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 8888:8888 -p 8889:8889 -it --rm openhackathons:ai-for-science`

The container launches jupyter lab and runs on port 8888
`jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

#### Singularity Container

To build the singularity container, run: 
`singularity build --fakeroot --sandbox End-to-End-AI-for-Science.sif Singularity`

Then, run the container:
`singularity run --writable --nv End-to-End-AI-for-Science.sif jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/aswkumar99/End-to-End-AI-for-Science/issues).
