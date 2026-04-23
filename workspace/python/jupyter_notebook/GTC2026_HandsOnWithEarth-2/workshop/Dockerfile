FROM nvcr.io/nvidia/physicsnemo/physicsnemo:25.11

ENV PYTHONPATH="/physicsnemo:${PYTHONPATH}"
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

# Install Jupyter - pin to new version as needed
RUN pip install --no-cache-dir --root-user-action ignore \
    jupyterlab==4.5.4 jupyterlab-widgets jupyter-archive jupyter-resource-usage jupytext \
    && rm -rf /root/.jupyter \
    && jupyter labextension disable "@jupyterlab/apputils-extension:announcements" 

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglx-mesa0 libegl1 xvfb git \
    && rm -rf /var/lib/apt/lists/*

# Install uv if not already available and create venv with uv
# Install PhysicsNeMo from source with model extras (includes dependencies for StormCast)
# Install Earth2Studio for HRRR/ERA5 data access
# Install additional training dependencies
# Install ipykernel and register the kernel
RUN pip install uv \
    && cd /root && uv init --python=3.12 \
    && uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git@0.12.1" --extra sfno \
    && uv add torch-optimi pyproj matplotlib cartopy omegaconf hydra-core wandb \
    nvidia-ml-py jupyter ipywidgets \
    tensorboard nvtx termcolor scipy setuptools==81.0.0 \
    nvidia-physicsnemo==1.3.0 \
    && uv pip install nvidia-ml-py && uv pip uninstall pynvml \
    && uv add ipykernel && uv run python -m ipykernel install --user --name=physicsnemo 

WORKDIR /dli/workshop

EXPOSE 8888

ADD entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh"]
