# Hands-On with Earth-2: Building Local and National Weather Resilience with AI

This workshop on using Earth-2 for high-resolution weather modeling has been presented at GTC 2026.

**Workshop recording:** [DLIT81485](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81485/)

The training pipeline is based on the [StormCast example](<(https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/stormcast)>) in the PhysicsNeMo GitHub repository.

## Instructions

- Build your environment using `workshop/Dockerfile`:

```
docker build -t earth2workshop:25:06 -f workshop/Dockerfile workshop
```

- Start your container on a system with an NVIDIA GPU:

```
docker run --gpus all --ipc host --pid host --shm-size 16g \
    -v /path/to/workshop:/dli/workshop \
    -v /path/to/data:/data \
    -t earth2workshop:25:06
```

- Navigate to `localhost:8888` in a web browser.
- Follow the notebooks in `workshop/notebooks`.

## Resources

- [Original workshop recording](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81485/)
- [Example in the PhysicsNeMo GitHub repository](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/stormcast)
- [Earth2Studio GitHub repository](https://github.com/NVIDIA/earth2studio)
- [Earth2Studio DLI workshop](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-31+V1)
